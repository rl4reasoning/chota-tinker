"""GEPA optimize_anything: optimize a system prompt for LCB coding problems.

Each evaluator call: solver LLM → extract ```python → run tests → return 0-1 reward.
GEPA iterates, proposing improved system prompts based on failures.

Usage:
  python optimize_anything_coding.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --n-problems 12 --reflection-minibatch-size 4 \
    --proposer-model anthropic/claude-sonnet-4-6 \
    --output-dir ./run_002 --max-metric-calls 120

  # Fair comparison run (30B model, single-turn):
  python optimize_anything_coding.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --n-problems 12 --reflection-minibatch-size 4 \
    --max-tokens 8192 \
    --proposer-model anthropic/claude-sonnet-4-6 \
    --output-dir ./run_single_30b --max-metric-calls 168
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import threading
from typing import Any

import hashlib

import litellm
import requests
from datasets import load_dataset
from openai import OpenAI

import gepa.optimize_anything as oa
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig
from utils.fast_eval import _evaluate_code, _extract_answer_code
from utils.vllm_multi_gpu import (
    build_vllm_server_urls,
    launch_vllm_servers,
    register_vllm_shutdown,
    resolve_vllm_gpu_ids,
    wait_for_vllm_servers,
)


class BudgetLogger:
    """Sequential log (budget_log.txt) showing how the evaluation budget is used.

    Lines in arrival order:
      >> NEW SEED / CANDIDATE   — first time a candidate hash is seen
      METRIC N/max              — each evaluator call (problem, score)
      PROPOSER #N               — each proposer call (tokens, budget, problems shown)
    """

    def __init__(self, log_path: str, max_metric_calls: int, n_problems: int, append: bool = False) -> None:
        self._path = log_path
        self._max = max_metric_calls
        self._lock = threading.Lock()
        self._metric_count = 0
        self._proposer_count = 0
        implied = max_metric_calls // n_problems if n_problems else "?"
        if append:
            with open(log_path, "a") as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"RESUMED at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"  max_metric_calls : {max_metric_calls}\n")
                f.write(f"{'='*70}\n\n")
        else:
            with open(log_path, "w") as f:
                f.write("BUDGET LOG\n")
                f.write(f"  max_metric_calls : {max_metric_calls}\n")
                f.write(f"  n_problems       : {n_problems}\n")
                f.write(f"  implies          : ~{implied} candidate evaluations\n")
                f.write("=" * 70 + "\n\n")

    @staticmethod
    def _ts() -> str:
        return datetime.datetime.now().strftime("%H:%M:%S")

    def _append(self, line: str) -> None:
        with self._lock:
            with open(self._path, "a") as f:
                f.write(line + "\n")

    def log_new_candidate(self, candidate_id: int, cand_hash: str) -> None:
        label = "SEED" if candidate_id == 1 else f"CANDIDATE #{candidate_id}"
        self._append(f"[{self._ts()}]  >> NEW {label} ({cand_hash})")

    def log_metric(self, cand_hash: str, candidate_id: int, prob_idx: Any, score: float, extra: str = "") -> int:
        """Log one evaluator call. Returns the 1-based call number."""
        with self._lock:
            self._metric_count += 1
            call_num = self._metric_count
        label = "seed" if candidate_id == 1 else f"cand#{candidate_id}"
        self._append(
            f"[{self._ts()}]  METRIC  {call_num:4d}/{self._max}"
            f"  {label}({cand_hash})  prob={prob_idx}  score={score:.4f}{extra}"
        )
        return call_num

    def log_proposer(self, in_tokens: int | None, out_tokens: int | None, prob_indices: list[int]) -> None:
        with self._lock:
            self._proposer_count += 1
            call_num = self._proposer_count
            current_evals = self._metric_count
        remaining = self._max - current_evals
        tok_str = f"in={in_tokens}tok  out={out_tokens}tok" if in_tokens is not None else "tokens=unknown"
        idx_str = str(sorted(prob_indices)) if prob_indices else "unknown"
        self._append(
            f"[{self._ts()}]  PROPOSER #{call_num}  {tok_str}"
            f"  [budget: {current_evals}/{self._max}  ({remaining} remaining)]"
            f"  shown_probs={idx_str}"
        )


DEFAULT_DATASET = "bicycleman15/intellect_3_code_very_hard"
DEFAULT_MAX_TESTS = 10
DEFAULT_TIMEOUT_S = 5.0
DEFAULT_PROPOSER_MODEL = "openai/gpt-4o-mini"
DEFAULT_N_PROBLEMS = 10
DEFAULT_MAX_METRIC_CALLS = None  # auto: n_problems * 10

SEED_SYSTEM_PROMPT = """You are a helpful coding assistant.
Solve the given programming problem and provide your solution.

First, think about the problem step by step.
Then, provide your final solution wrapped in ```python``` code blocks.
"""

SEED_CODE = """\
```python
import sys
input = sys.stdin.readline

def main():
    # TODO: implement solution
    pass

main()
```
"""


def install_proposer_logger(output_dir: str, budget_logger: "BudgetLogger | None" = None) -> None:
    """Log proposer queries to proposer_queries.jsonl; optionally update budget_log.txt."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "proposer_queries.jsonl")
    _lock = threading.Lock()
    _counter = [0]
    _orig = litellm.completion

    def _logged(model, messages, **kwargs):
        with _lock:
            _counter[0] += 1
            idx = _counter[0]

        completion = _orig(model=model, messages=messages, **kwargs)

        usage = getattr(completion, "usage", None)
        in_tokens = getattr(usage, "prompt_tokens", None)
        out_tokens = getattr(usage, "completion_tokens", None)
        with _lock:
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "call": idx,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "model": model,
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens,
                    "messages": messages,
                }) + "\n")

        if budget_logger is not None:
            full_text = "\n".join(m.get("content", "") or "" for m in messages if isinstance(m, dict))
            prob_indices = [int(x) for x in re.findall(r"PROBLEM_INDEX:\s*(-?\d+)", full_text) if int(x) >= 0]
            budget_logger.log_proposer(in_tokens, out_tokens, prob_indices)

        return completion

    litellm.completion = _logged
    print(f"[proposer logger] → {log_path}")


def launch_vllm_engine(args) -> tuple[str, str]:
    gpu_ids = resolve_vllm_gpu_ids(args)
    server_urls = build_vllm_server_urls(args, gpu_ids)
    print(f"Launching vLLM: {args.model}  GPUs={','.join(gpu_ids)}")
    processes = launch_vllm_servers(args, gpu_ids)
    register_vllm_shutdown(processes)
    print(f"Waiting for vLLM (timeout={args.vllm_startup_timeout_s}s) ...")
    wait_for_vllm_servers(server_urls, args.vllm_startup_timeout_s)
    print("vLLM ready.\n")
    return server_urls[0].rstrip("/") + "/v1", args.model


def connect_vllm_server(vllm_url: str, solver_model: str | None) -> tuple[str, str]:
    openai_base_url = vllm_url.rstrip("/") + "/v1"
    if solver_model:
        return openai_base_url, solver_model
    try:
        resp = requests.get(openai_base_url + "/models", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if not models:
            raise RuntimeError("/v1/models returned an empty list")
        return openai_base_url, models[0]["id"]
    except Exception as e:
        raise RuntimeError(
            f"Could not fetch model from {openai_base_url}: {e}\n"
            "Is the vLLM server running? Pass --solver-model to skip auto-detection."
        ) from e


def _parse_tests(row: dict) -> dict[str, Any]:
    def _load(v):
        return json.loads(v) if isinstance(v, str) else v
    if row.get("info") is not None:
        tests = _load(row["info"]).get("tests")
        if tests is not None:
            return _load(tests)
    if row.get("tests") is not None:
        return _load(row["tests"])
    raise KeyError(f"tests not found in row: {list(row.keys())}")


def load_problems(dataset_name: str, start: int = 0, end: int | None = None) -> list[dict[str, Any]]:
    ds = load_dataset(dataset_name, split="train")
    if end is None:
        end = len(ds)
    ds = ds.select(range(start, min(end, len(ds))))
    return [{"question": row["question"], "tests": _parse_tests(row), "_index": i} for i, row in enumerate(ds)]


def _eval_code_response(
    code_response: str,
    tests: dict[str, Any],
    max_tests: int,
    timeout_s: float,
    question: str = "",
    prob_idx: Any = -1,
) -> float:
    """Extract code, run tests, log ASI (plain-text to avoid breaking proposer's ``` fence), return reward."""
    code = _extract_answer_code(code_response)
    if code is None:
        oa.log("No ```python ... ``` code block found in response.")
        return 0.0

    n_available = min(len(tests.get("inputs", [])), max_tests)
    reward, timeout_count, _ = _evaluate_code(
        code=code, tests=tests, max_tests=max_tests, timeout_s=timeout_s,
        timeout_record_limit=5, require_solution_class=True,
    )
    passed = round(reward * n_available)

    sep = "=" * 40
    log_parts = [sep, f"RESULT: {passed}/{n_available} tests passed | TLE: {timeout_count} | PROBLEM_INDEX: {prob_idx}", sep]
    if question:
        log_parts += ["", "QUESTION:", question]
    log_parts += ["", "LLM RESPONSE:", code_response]
    oa.log("\n".join(log_parts))
    return reward


def run_prompt_optimization(
    problems: list[dict[str, Any]],
    solver_model: str,
    proposer_model: str,
    max_tests: int,
    timeout_s: float,
    max_tokens: int,
    objective: str | None,
    base_url: str,
    api_key: str,
    reflection_minibatch_size: int | None = None,
    max_metric_calls: int = DEFAULT_MAX_METRIC_CALLS,
    budget_logger: "BudgetLogger | None" = None,
    output_dir: str | None = None,
) -> Any:
    client = OpenAI(api_key=api_key, base_url=base_url)
    _cand_lock = threading.Lock()
    _state = {"candidates": {}, "cand_id": 0}

    def evaluator(system_prompt: str, example: dict[str, Any]) -> float:
        cand_hash = hashlib.md5(system_prompt.encode()).hexdigest()[:8]
        with _cand_lock:
            is_new = cand_hash not in _state["candidates"]
            if is_new:
                _state["cand_id"] += 1
                _state["candidates"][cand_hash] = _state["cand_id"]
            candidate_id = _state["candidates"][cand_hash]

        if is_new and budget_logger is not None:
            budget_logger.log_new_candidate(candidate_id, cand_hash)

        prob_idx = example.get("_index", "?")
        cand_type = "seed" if candidate_id == 1 else f"cand#{candidate_id}"

        try:
            response = client.chat.completions.create(
                model=solver_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": example["question"]},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            code_response = response.choices[0].message.content or ""
        except Exception as e:
            oa.log(f"LLM call failed: {e}")
            print(f"[metric] {cand_type}({cand_hash}) prob={prob_idx} → LLM FAILED: {e}")
            return 0.0

        reward = _eval_code_response(
            code_response, example["tests"], max_tests, timeout_s,
            example.get("question", ""), prob_idx,
        )
        n_available = min(len(example["tests"].get("inputs", [])), max_tests)
        passed = round(reward * n_available)

        call_num = budget_logger.log_metric(cand_hash, candidate_id, prob_idx, reward) if budget_logger else "?"
        print(f"[metric #{call_num}] {cand_type}({cand_hash}) prob={prob_idx} → score={reward:.4f} ({passed}/{n_available} tests)")
        return reward

    gepa_config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=max_metric_calls,
            parallel=True,
            max_workers=len(problems),
            run_dir=output_dir,
        ),
        reflection=ReflectionConfig(
            reflection_lm=proposer_model,
            reflection_minibatch_size=reflection_minibatch_size,
        ),
    )
    return oa.optimize_anything(
        seed_candidate=SEED_SYSTEM_PROMPT,
        evaluator=evaluator,
        dataset=problems,
        config=gepa_config,
        objective=objective or (
            "Maximize the fraction of test cases passed across all problems. "
            "The system prompt should instruct the LLM to write correct, efficient Python "
            "that reads from stdin and writes to stdout, enclosed in ```python blocks."
        ),
        background=(
            "We are optimizing a system prompt for an LLM that solves competitive "
            "programming problems in Python. The LLM receives the problem statement as "
            "the user message and should output a complete Python solution."
        ),
    )


def run_single_task(
    problem: dict[str, Any],
    proposer_model: str,
    max_tests: int,
    timeout_s: float,
    objective: str | None,
    max_metric_calls: int = DEFAULT_MAX_METRIC_CALLS,
    output_dir: str | None = None,
) -> Any:
    def evaluator(candidate: str) -> float:
        return _eval_code_response(candidate, problem["tests"], max_tests, timeout_s)

    gepa_config = GEPAConfig(
        engine=EngineConfig(max_metric_calls=max_metric_calls, run_dir=output_dir),
        reflection=ReflectionConfig(reflection_lm=proposer_model),
    )
    return oa.optimize_anything(
        seed_candidate=SEED_CODE,
        evaluator=evaluator,
        config=gepa_config,
        objective=objective or (
            "Maximize the fraction of test cases passed. "
            "Write correct, efficient Python that reads from stdin and writes to stdout."
        ),
        background=f"Coding problem to solve:\n\n{problem['question']}",
    )


def dump_evolution(result: Any, output_dir: str, mode: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "evolution.json"), "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    key = result._str_candidate_key
    def _get_text(cand_dict: dict) -> str:
        return cand_dict[key] if key and key in cand_dict else json.dumps(cand_dict, indent=2)

    indices = sorted(range(result.num_candidates), key=lambda i: result.discovery_eval_counts[i])

    lines = [
        "=" * 70,
        f"CANDIDATE EVOLUTION  ({result.num_candidates} candidates, mode={mode})",
        f"Total metric calls: {result.total_metric_calls}",
        f"Best candidate idx: {result.best_idx}  (score={result.val_aggregate_scores[result.best_idx]:.4f})",
        "=" * 70,
    ]
    for idx in indices:
        score = result.val_aggregate_scores[idx]
        parent_str = ", ".join(str(p) for p in result.parents[idx] if p is not None) or "seed"
        is_best = "  *** BEST ***" if idx == result.best_idx else ""
        text = _get_text(result.candidates[idx])
        preview = text[:600] + ("\n[... truncated ...]" if len(text) > 600 else "")
        lines += [
            f"\n--- Candidate #{idx}  (found at eval_call={result.discovery_eval_counts[idx]}){is_best}",
            f"    Score : {score:.4f}",
            f"    Parent: {parent_str}",
            f"    Text  :\n{preview}",
        ]
    lines.append(f"\n{'='*70}")

    with open(os.path.join(output_dir, "evolution.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    with open(os.path.join(output_dir, "candidates_scores.jsonl"), "w") as f:
        for idx in indices:
            f.write(json.dumps({
                "idx": idx,
                "score": result.val_aggregate_scores[idx],
                "discovery_eval_count": result.discovery_eval_counts[idx],
                "parents": result.parents[idx],
                "is_best": idx == result.best_idx,
                "candidate": _get_text(result.candidates[idx]),
            }) + "\n")

    print(f"Evolution saved to {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GEPA optimize_anything: find the best system prompt for LCB coding",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--mode", choices=["prompt", "single"], default="prompt")

    vllm_group = parser.add_mutually_exclusive_group(required=True)
    vllm_group.add_argument("--model", help="HuggingFace model to serve via vLLM")
    vllm_group.add_argument("--vllm-url", help="URL of a running vLLM server")

    parser.add_argument("--vllm-port", type=int, default=8000, dest="vllm_server_base_port")
    parser.add_argument("--vllm-host", default="127.0.0.1", dest="vllm_server_host")
    parser.add_argument("--gpu-ids", default=None, dest="vllm_gpu_ids")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--startup-timeout", type=float, default=300.0, dest="vllm_startup_timeout_s")
    parser.add_argument("--solver-model", default=None)
    parser.add_argument("--proposer-model", default=DEFAULT_PROPOSER_MODEL)
    parser.add_argument("--api-key", default="token")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--n-problems", type=int, default=DEFAULT_N_PROBLEMS)
    parser.add_argument("--problem-id", type=int, default=0)
    parser.add_argument("--max-tests", type=int, default=DEFAULT_MAX_TESTS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--objective", default=None)
    parser.add_argument("--max-metric-calls", type=int, default=None, dest="max_metric_calls",
                        help="Total evaluator calls budget; default = n_problems * 10")
    parser.add_argument("--reflection-minibatch-size", type=int, default=None, dest="reflection_minibatch_size",
                        help="Problems shown to proposer per reflection step (default: 3)")
    parser.add_argument("--output-dir", default=None)

    args = parser.parse_args()

    if args.max_metric_calls is None:
        args.max_metric_calls = args.n_problems * 10
        print(f"[auto] max-metric-calls = {args.max_metric_calls}")

    if args.model:
        base_url, solver_model = launch_vllm_engine(args)
    else:
        base_url, solver_model = connect_vllm_server(args.vllm_url, args.solver_model)
    print(f"Solver: {solver_model}  ({base_url})\n")

    budget_logger = None
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        gepa_state_path = os.path.join(args.output_dir, "gepa_state.bin")
        is_resuming = os.path.exists(gepa_state_path)
        if is_resuming:
            print(f"[resume] Checkpoint found in {args.output_dir!r} — resuming optimization.")
        else:
            with open(os.path.join(args.output_dir, "args.json"), "w") as f:
                json.dump(vars(args), f, indent=2)
        n_problems_for_log = args.n_problems if args.mode == "prompt" else 1
        budget_log_path = os.path.join(args.output_dir, "budget_log.txt")
        budget_logger = BudgetLogger(budget_log_path, args.max_metric_calls, n_problems_for_log, append=is_resuming)
        print(f"[budget logger] → {budget_log_path}")
        install_proposer_logger(args.output_dir, budget_logger)

    if args.mode == "prompt":
        print(f"Loading {args.n_problems} problems from {args.dataset} ...")
        problems = load_problems(args.dataset, start=0, end=args.n_problems)
        if not problems:
            print("ERROR: no problems loaded", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded {len(problems)} problems.\n")

        result = run_prompt_optimization(
            problems=problems,
            solver_model=solver_model,
            proposer_model=args.proposer_model,
            max_tests=args.max_tests,
            timeout_s=args.timeout,
            max_tokens=args.max_tokens,
            objective=args.objective,
            base_url=base_url,
            api_key=args.api_key,
            reflection_minibatch_size=args.reflection_minibatch_size,
            max_metric_calls=args.max_metric_calls,
            budget_logger=budget_logger,
            output_dir=args.output_dir,
        )
    else:
        print(f"Loading problem {args.problem_id} from {args.dataset} ...")
        problems = load_problems(args.dataset, start=args.problem_id, end=args.problem_id + 1)
        if not problems:
            print(f"ERROR: no problem at index {args.problem_id}", file=sys.stderr)
            sys.exit(1)
        print(f"Problem (first 500 chars):\n{problems[0]['question'][:500]}\n")

        result = run_single_task(
            problem=problems[0],
            proposer_model=args.proposer_model,
            max_tests=args.max_tests,
            timeout_s=args.timeout,
            objective=args.objective,
            max_metric_calls=args.max_metric_calls,
            output_dir=args.output_dir,
        )

    best_score = result.val_aggregate_scores[result.best_idx]
    print(f"\n{'='*70}")
    print(f"DONE  candidates={result.num_candidates}  metric_calls={result.total_metric_calls}"
          f"  best_idx={result.best_idx}  score={best_score:.4f}")
    print(f"{'='*70}\n")
    print("BEST CANDIDATE" if args.mode == "prompt" else "BEST CODE")
    print("─" * 70)
    print(result.best_candidate)
    print("─" * 70)

    if args.output_dir:
        dump_evolution(result, args.output_dir, args.mode)


if __name__ == "__main__":
    main()
