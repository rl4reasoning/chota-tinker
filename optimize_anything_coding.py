"""
Use GEPA's optimize_anything framework to optimize a system prompt that boosts
an LLM's coding accuracy across LCB problems.

The candidate is a system prompt (plain text).  For each problem the evaluator:
  1. Calls the solver LLM with  (system=candidate, user=problem_question).
  2. Extracts the ```python block from the response.
  3. Runs the code against test cases and returns a 0-1 reward.
  4. Logs the generated code + failures as ASI so GEPA can read them.

GEPA then iterates, proposing improved system prompts based on what went wrong.

Usage:
  # Full example: 20 problems, auto budget (~10 candidate evaluations), save evolution
    python optimize_anything_coding.py --model Qwen/Qwen3-4B-Instruct-2507 \
    --n-problems 12 --reflection-minibatch-size 4 \
    --proposer-model anthropic/claude-sonnet-4-6 --output-dir ./run_002 --max-metric-calls 60


Prerequisites:
  pip install litellm           # GEPA uses litellm for its internal proposer LLM

  # Set the API key matching your --proposer-model:
  export OPENAI_API_KEY=sk-...              # for openai/gpt-4o, openai/gpt-4o-mini, etc.
  export ANTHROPIC_API_KEY=sk-ant-...       # for anthropic/claude-opus-4-6, etc.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import threading
from typing import Any

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

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

DEFAULT_DATASET = "bicycleman15/intellect_3_code_very_hard"
DEFAULT_MAX_TESTS = 10
DEFAULT_TIMEOUT_S = 5.0
DEFAULT_PROPOSER_MODEL = "openai/gpt-4o-mini"  # litellm model string for GEPA's internal proposer
DEFAULT_N_PROBLEMS = 10
DEFAULT_MAX_METRIC_CALLS = None  # None → auto: n_problems * 10 (i.e. ~10 candidate evaluations)

# Seed system prompt — a plain, naive starting point.
# GEPA will iteratively improve it based on failures.
SEED_SYSTEM_PROMPT = """You are a helpful coding assistant.
Solve the given programming problem and provide your solution.

First, think about the problem step by step.
Then, provide your final solution wrapped in ```python``` code blocks.
"""

# Seed code candidate for single-task mode (direct code search).
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


# --------------------------------------------------------------------------- #
# Proposer query logger
# --------------------------------------------------------------------------- #

def install_proposer_logger(output_dir: str) -> None:
    """Monkey-patch litellm.completion to log every proposer query.

    Appends one JSON object per line to <output_dir>/proposer_queries.jsonl:
      {"call": N, "timestamp": "...", "model": "...", "messages": [...]}

    Safe to call from multi-threaded GEPA (all writes are serialized via a lock).
    """
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
        entry = {
            "call": idx,
            "timestamp": datetime.datetime.now().isoformat(),
            "model": model,
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "messages": messages,
        }
        with _lock:
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        return completion

    litellm.completion = _logged
    print(f"[proposer logger] Proposer queries will be logged to: {log_path}")


# --------------------------------------------------------------------------- #
# vLLM engine
# --------------------------------------------------------------------------- #

def launch_vllm_engine(args) -> tuple[str, str]:
    """Launch a vLLM server subprocess and return (openai_base_url, model_name).

    Uses the same launch/wait/shutdown utilities as collect_trajectories.py.
    The server is automatically shut down when the process exits.
    """
    gpu_ids = resolve_vllm_gpu_ids(args)
    server_urls = build_vllm_server_urls(args, gpu_ids)

    print(f"Launching vLLM server: {args.model}")
    print(f"  GPUs : {', '.join(gpu_ids)}")
    print(f"  URL(s): {', '.join(server_urls)}")

    processes = launch_vllm_servers(args, gpu_ids)
    register_vllm_shutdown(processes)

    print(f"Waiting for vLLM to be ready (timeout: {args.vllm_startup_timeout_s}s) ...")
    wait_for_vllm_servers(server_urls, args.vllm_startup_timeout_s)
    print("vLLM server is ready.\n")

    # Return the first server's OpenAI-compatible URL and the model name.
    openai_base_url = server_urls[0].rstrip("/") + "/v1"
    return openai_base_url, args.model


def connect_vllm_server(vllm_url: str, solver_model: str | None) -> tuple[str, str]:
    """Connect to an already-running vLLM server.

    Returns (openai_base_url, model_name).  If solver_model is None, the model
    name is auto-detected from the server's /v1/models endpoint.
    """
    openai_base_url = vllm_url.rstrip("/") + "/v1"
    if solver_model:
        return openai_base_url, solver_model

    url = openai_base_url + "/models"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if not models:
            raise RuntimeError("/v1/models returned an empty list")
        model_name = models[0]["id"]
    except Exception as e:
        raise RuntimeError(
            f"Could not fetch model list from {url}: {e}\n"
            "Is the vLLM server running? Pass --solver-model to skip auto-detection."
        ) from e

    return openai_base_url, model_name


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #

def _parse_tests(row: dict) -> dict[str, Any]:
    info_raw = row.get("info")
    if info_raw is not None:
        info = json.loads(info_raw) if isinstance(info_raw, str) else info_raw
        tests_raw = info.get("tests")
        if tests_raw is not None:
            return json.loads(tests_raw) if isinstance(tests_raw, str) else tests_raw
    tests_raw = row.get("tests")
    if tests_raw is not None:
        return json.loads(tests_raw) if isinstance(tests_raw, str) else tests_raw
    raise KeyError(f"Cannot find tests in row with keys: {list(row.keys())}")


def load_problems(
    dataset_name: str,
    start: int = 0,
    end: int | None = None,
) -> list[dict[str, Any]]:
    ds = load_dataset(dataset_name, split="train")
    if end is None:
        end = len(ds)
    ds = ds.select(range(start, min(end, len(ds))))
    return [{"question": row["question"], "tests": _parse_tests(row)} for row in ds]


# --------------------------------------------------------------------------- #
# Core evaluation helper (shared by both modes)
# --------------------------------------------------------------------------- #

def _eval_code_response(
    code_response: str,
    tests: dict[str, Any],
    max_tests: int,
    timeout_s: float,
    question: str = "",
) -> float:
    """Extract code from an LLM response, run it against tests, log ASI, return reward."""
    code = _extract_answer_code(code_response)
    if code is None:
        oa.log("No ```python ... ``` code block found in response.")
        return 0.0

    n_available = min(len(tests.get("inputs", [])), max_tests)
    reward, timeout_count, timeout_indices = _evaluate_code(
        code=code,
        tests=tests,
        max_tests=max_tests,
        timeout_s=timeout_s,
        timeout_record_limit=5,
        require_solution_class=True,
    )

    passed = round(reward * n_available)

    # Log diagnostic info as ASI so GEPA can read it when proposing improvements.
    asi_parts = [f"Test results: {passed}/{n_available} passed. TLE: {timeout_count}."]
    if question:
        asi_parts.append(f"\nQuestion:\n{question}")
    asi_parts.append(f"\nFull LLM response:\n{code_response}")
    oa.log("\n".join(asi_parts))

    return reward


# --------------------------------------------------------------------------- #
# Prompt optimization mode
# --------------------------------------------------------------------------- #

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
) -> Any:
    """Optimize a system prompt across N coding problems."""
    client = OpenAI(api_key=api_key, base_url=base_url)

    def evaluator(system_prompt: str, example: dict[str, Any]) -> float:
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
            return 0.0

        return _eval_code_response(code_response, example["tests"], max_tests, timeout_s, example.get("question", ""))

    gepa_config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=max_metric_calls,
            parallel=True,
            max_workers=len(problems),
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


# --------------------------------------------------------------------------- #
# Single-task mode (direct code search for one problem)
# --------------------------------------------------------------------------- #

def run_single_task(
    problem: dict[str, Any],
    proposer_model: str,
    max_tests: int,
    timeout_s: float,
    objective: str | None,
    max_metric_calls: int = DEFAULT_MAX_METRIC_CALLS,
) -> Any:
    """Directly search for code that solves one specific problem."""

    def evaluator(candidate: str) -> float:
        return _eval_code_response(candidate, problem["tests"], max_tests, timeout_s)

    gepa_config = GEPAConfig(
        engine=EngineConfig(max_metric_calls=max_metric_calls),
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


# --------------------------------------------------------------------------- #
# Evolution dump / visualization
# --------------------------------------------------------------------------- #

def dump_evolution(result: Any, output_dir: str, mode: str) -> None:
    """Write candidate evolution to output_dir after optimization completes.

    Creates two files:
      evolution.json  — full result serialized via result.to_dict()
      evolution.txt   — human-readable report: candidates in discovery order
                        with score, parent, and truncated text
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- JSON dump (full, round-trippable) ----------------------------------
    json_path = os.path.join(output_dir, "evolution.json")
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"Evolution JSON written to {json_path}")

    # ---- Text report --------------------------------------------------------
    key = result._str_candidate_key  # unwrap str candidates stored as {key: text}

    def _get_text(cand_dict: dict) -> str:
        if key and key in cand_dict:
            return cand_dict[key]
        return json.dumps(cand_dict, indent=2)

    # Sort by discovery_eval_counts (order they were found during search)
    indices = sorted(range(result.num_candidates), key=lambda i: result.discovery_eval_counts[i])

    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"CANDIDATE EVOLUTION  ({result.num_candidates} candidates, mode={mode})")
    lines.append(f"Total metric calls: {result.total_metric_calls}")
    lines.append(f"Best candidate idx: {result.best_idx}  (score={result.val_aggregate_scores[result.best_idx]:.4f})")
    lines.append(f"{'='*70}")

    for idx in indices:
        score = result.val_aggregate_scores[idx]
        eval_count = result.discovery_eval_counts[idx]
        parents = result.parents[idx]
        parent_str = ", ".join(str(p) for p in parents if p is not None) or "seed"
        is_best = "  *** BEST ***" if idx == result.best_idx else ""
        text = _get_text(result.candidates[idx])
        preview = text[:600] + ("\n[... truncated ...]" if len(text) > 600 else "")

        lines.append(f"\n--- Candidate #{idx}  (found at eval_call={eval_count}){is_best}")
        lines.append(f"    Score : {score:.4f}")
        lines.append(f"    Parent: {parent_str}")
        lines.append(f"    Text  :\n{preview}")

    lines.append(f"\n{'='*70}")

    txt_path = os.path.join(output_dir, "evolution.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Evolution report written to {txt_path}")

    # ---- JSONL per-candidate dump (full text, machine-readable) --------------
    jsonl_path = os.path.join(output_dir, "candidates_scores.jsonl")
    with open(jsonl_path, "w") as f:
        for idx in indices:
            entry = {
                "idx": idx,
                "score": result.val_aggregate_scores[idx],
                "discovery_eval_count": result.discovery_eval_counts[idx],
                "parents": result.parents[idx],
                "is_best": idx == result.best_idx,
                "candidate": _get_text(result.candidates[idx]),
            }
            f.write(json.dumps(entry) + "\n")
    print(f"Candidates JSONL written to {jsonl_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GEPA optimize_anything: find the best system prompt for LCB coding problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="HuggingFace dataset (default: %(default)s)")
    parser.add_argument("--mode", choices=["prompt", "single"], default="prompt",
                        help="'prompt' optimizes a system prompt across N problems (default); "
                             "'single' directly searches for code for one problem")

    # ---- vLLM engine --------------------------------------------------------
    vllm_group = parser.add_mutually_exclusive_group(required=True)
    vllm_group.add_argument("--model",
                            help="HuggingFace model to load and serve via vLLM "
                                 "(launches a vLLM subprocess)")
    vllm_group.add_argument("--vllm-url",
                            help="Connect to an already-running vLLM server "
                                 "(e.g. http://localhost:8000)")

    # vLLM server settings — only relevant when --model is used
    parser.add_argument("--vllm-port", type=int, default=8000,
                        dest="vllm_server_base_port",
                        help="Port for the vLLM server (default: 8000)")
    parser.add_argument("--vllm-host", default="127.0.0.1",
                        dest="vllm_server_host",
                        help="Host to bind the vLLM server (default: 127.0.0.1)")
    parser.add_argument("--gpu-ids", default=None,
                        dest="vllm_gpu_ids",
                        help="Comma-separated GPU IDs for vLLM (default: all visible GPUs)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="vLLM GPU memory utilization (default: 0.9)")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="vLLM max sequence length (default: model default)")
    parser.add_argument("--startup-timeout", type=float, default=300.0,
                        dest="vllm_startup_timeout_s",
                        help="Seconds to wait for vLLM to be ready (default: 300)")

    # ---- solver / proposer --------------------------------------------------
    parser.add_argument("--solver-model", default=None,
                        help="Override solver model name (auto-detected from vLLM if not set)")
    parser.add_argument("--proposer-model", default=DEFAULT_PROPOSER_MODEL,
                        help="litellm model string for GEPA's internal proposer "
                             "(default: %(default)s)")
    parser.add_argument("--api-key", default="token",
                        help="API key for the solver server (default: 'token', fine for vLLM)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens for solver LLM response (default: %(default)s)")

    # ---- task ---------------------------------------------------------------
    parser.add_argument("--n-problems", type=int, default=DEFAULT_N_PROBLEMS,
                        help="Problems to evaluate per iteration in prompt mode (default: %(default)s)")
    parser.add_argument("--problem-id", type=int, default=0,
                        help="Problem index for single-task mode (default: %(default)s)")
    parser.add_argument("--max-tests", type=int, default=DEFAULT_MAX_TESTS,
                        help="Max test cases per problem per evaluation (default: %(default)s)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                        help="Per-test-case timeout in seconds (default: %(default)s)")
    parser.add_argument("--objective", default=None,
                        help="Override the optimization objective text")
    parser.add_argument("--max-metric-calls", type=int, default=None,
                        dest="max_metric_calls",
                        help="Total evaluator calls budget (stopping condition). "
                             "With N problems this gives max_metric_calls/N candidate evaluations. "
                             "Default: n_problems * 10 (i.e. ~10 candidate evaluations).")
    parser.add_argument("--reflection-minibatch-size", type=int, default=None,
                        dest="reflection_minibatch_size",
                        help="Problems shown to proposer per reflection step "
                             "(default: 3). Lower = faster but less context; "
                             "higher = richer feedback but bigger prompts.")
    parser.add_argument("--output-dir", default=None,
                        help="If set, dump evolution.json and evolution.txt here after optimization")

    args = parser.parse_args()

    # Auto-scale max_metric_calls if not explicitly set.
    # Need at least n_problems calls just for the seed; budget = n_problems * 10
    # gives ~10 candidate evaluations (seed + 9 proposed).
    if args.max_metric_calls is None:
        args.max_metric_calls = args.n_problems * 10
        print(f"[auto] max-metric-calls = {args.max_metric_calls} ({args.n_problems} problems × 10)")

    # ---- Resolve solver endpoint -------------------------------------------
    if args.model:
        base_url, solver_model = launch_vllm_engine(args)
    else:
        base_url, solver_model = connect_vllm_server(args.vllm_url, args.solver_model)

    print(f"Solver: {solver_model}  ({base_url})\n")

    # ---- Install proposer logger (if output dir specified) -----------------
    if args.output_dir:
        install_proposer_logger(args.output_dir)

    # ---- Run optimization --------------------------------------------------
    if args.mode == "prompt":
        print(f"[prompt mode] Loading {args.n_problems} problem(s) from {args.dataset} ...")
        problems = load_problems(args.dataset, start=0, end=args.n_problems)
        if not problems:
            print("ERROR: no problems loaded", file=sys.stderr)
            sys.exit(1)
        print(f"Loaded {len(problems)} problems.")
        print(f"Seed system prompt:\n{'-'*50}\n{SEED_SYSTEM_PROMPT}{'-'*50}\n")

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
        )

    else:  # single
        print(f"[single mode] Loading problem {args.problem_id} from {args.dataset} ...")
        problems = load_problems(args.dataset, start=args.problem_id, end=args.problem_id + 1)
        if not problems:
            print(f"ERROR: no problem at index {args.problem_id}", file=sys.stderr)
            sys.exit(1)
        problem = problems[0]
        n_tests = len(problem["tests"].get("inputs", []))
        print(f"Problem statement (first 500 chars):\n{problem['question'][:500]}")
        print(f"\nTest cases available: {n_tests}  (running up to {args.max_tests})\n")

        result = run_single_task(
            problem=problem,
            proposer_model=args.proposer_model,
            max_tests=args.max_tests,
            timeout_s=args.timeout,
            objective=args.objective,
            max_metric_calls=args.max_metric_calls,
        )

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    best_score = result.val_aggregate_scores[result.best_idx]
    print(f"Candidates explored : {result.num_candidates}")
    print(f"Total metric calls  : {result.total_metric_calls}")
    print(f"Best candidate idx  : {result.best_idx}")
    print(f"Best score          : {best_score:.4f}")

    label = "FINAL SYSTEM PROMPT" if args.mode == "prompt" else "FINAL CODE"
    print(f"\n{'─' * 70}")
    print(label)
    print(f"{'─' * 70}")
    print(result.best_candidate)
    print(f"{'─' * 70}")

    # ---- Print all candidates summary to stdout ----------------------------
    key = result._str_candidate_key
    indices = sorted(range(result.num_candidates),
                     key=lambda i: result.discovery_eval_counts[i])
    print(f"\n{'─' * 70}")
    print(f"ALL CANDIDATES  ({result.num_candidates} total, sorted by discovery order)")
    print(f"{'─' * 70}")
    for idx in indices:
        score = result.val_aggregate_scores[idx]
        parents = result.parents[idx]
        parent_str = ", ".join(str(p) for p in parents if p is not None) or "seed"
        is_best = "  ★ BEST" if idx == result.best_idx else ""
        cand = result.candidates[idx]
        text = cand[key] if key and key in cand else json.dumps(cand)
        preview = text[:120].replace("\n", " ↵ ")
        print(f"  #{idx:<2d}  score={score:.4f}  parent={parent_str:<6s}{is_best}")
        print(f"       {preview}...")
    print(f"{'─' * 70}")

    if args.output_dir:
        dump_evolution(result, args.output_dir, args.mode)


if __name__ == "__main__":
    main()
