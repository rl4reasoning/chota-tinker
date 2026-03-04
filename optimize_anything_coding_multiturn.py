"""GEPA optimize_anything: optimize a system prompt for multi-turn LCB coding.

Each evaluator runs a multi-turn loop with IntellectCodeEnv: solver LLM uses
<interact></interact> blocks for code execution, then submits a final ```python solution.

Usage:
  python optimize_anything_coding_multiturn.py \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --n-problems 8 --max-turns 5 --reflection-minibatch-size 4 \
    --proposer-model anthropic/claude-sonnet-4-6 \
    --output-dir ./run_multiturn_003 --max-metric-calls 80

  # Fair comparison run (30B model, multi-turn):
  python optimize_anything_coding_multiturn.py \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --n-problems 12 --max-turns 5 --reflection-minibatch-size 4 \
    --max-tokens 4096 \
    --proposer-model anthropic/claude-sonnet-4-6 \
    --output-dir ./run_multiturn_30b --max-metric-calls 240
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
from intellect_env import IntellectCodeEnv
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
      METRIC N/max              — each evaluator call (problem, score, turns)
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
DEFAULT_MAX_TURNS = 5
DEFAULT_INTERACT_TIMEOUT_S = 10.0
DEFAULT_EVAL_TIMEOUT_S = 5.0
DEFAULT_PROPOSER_MODEL = "openai/gpt-4o-mini"
DEFAULT_N_PROBLEMS = 10
DEFAULT_MAX_METRIC_CALLS = None  # auto: n_problems * 10

SEED_SYSTEM_PROMPT = """You are a helpful coding assistant.

You have access to a Python interpreter.
To execute code, wrap it inside <interact></interact>. I will run it and return stdout/stderr in a subsequent turn.

IMPORTANT CONTEXT:
- This is a multi-turn conversation.
- The Python interpreter is a tool for gathering evidence: testing hypotheses, validating assumptions, checking edge cases, and falsifying incorrect reasoning.
- You should interact only when doing so provides information that can change your understanding, reasoning, or final decision.

────────────────────────
HARD RULES (NON-NEGOTIABLE)
────────────────────────
- BEFORE you output ANY final solution code in a ```python``` block, you MUST have completed at least one successful <interact></interact> execution in an earlier turn.
- If you have NOT yet completed a successful <interact></interact>, you are FORBIDDEN from outputting any ```python``` code block (even partial solutions).
- In the FIRST assistant response after receiving a new coding problem, you MUST perform an <interact></interact> intended to test, validate, or falsify some part of your reasoning.
- Interactions performed solely to satisfy this requirement (without testing a hypothesis or reducing uncertainty) are INVALID.

────────────────────────
EXECUTION ENVIRONMENT (CRITICAL)
────────────────────────
- The execution environment does NOT take input from stdin. You MUST hardcode inputs in your code.
- The execution environment shows ONLY what you PRINT to stdout.
- EVERY <interact></interact> MUST include explicit print(...) statements.
- Do NOT rely on REPL-style expression outputs or implicit returns.

────────────────────────
DEFINITION OF "SUCCESSFUL <interact>"
────────────────────────
An interaction is successful ONLY if ALL of the following hold:
- The code executes without exceptions, AND
- It prints at least 2 lines of task-relevant evidence, AND
- At least one printed line is a newly computed result (not already given in the prompt), AND
- The subsequent assistant message explicitly uses this evidence to confirm, revise, or reject a stated hypothesis.

────────────────────────
MANDATORY INTERACTION STRUCTURE
────────────────────────
Before each <interact></interact>, you MUST clearly state:
- The specific hypothesis, assumption, or uncertainty being tested
- Why this cannot be fully resolved by reasoning alone
- What outcome you expect if the hypothesis is correct vs incorrect

After receiving the output, you MUST clearly state:
- What the output shows (summarize or quote key lines)
- Whether the hypothesis was confirmed, weakened, or falsified
- What (if anything) changed in your approach

────────────────────────
SOLUTION STRESS TEST (CRITICAL)
────────────────────────
- For algorithmic correctness problems, you MUST run at least one interaction that attempts to falsify your proposed solution.
- This interaction MUST compare your approach against a correct reference implementation using:
  (a) brute force / exhaustive checking for small inputs (e.g., n ≤ 6–8), OR
  (b) randomized testing against a slower but correct oracle.
- This interaction MUST print either:
  • "No counterexample found in K tests" (K ≥ 100), OR
  • A concrete counterexample where your approach disagrees with the oracle.
- If a counterexample is found, you MUST revise your approach and repeat the oracle test.

DO NOT overfit to the examples provided in the prompt.
Testing only the examples provided in the prompt does NOT count as validation or falsification.

────────────────────────
ANTI-THRASHING RULE
────────────────────────
- If an <interact></interact> produces no output, insufficient output, or redundant output, your NEXT interaction MUST fix this and MUST NOT repeat the same interaction pattern.

────────────────────────
ITERATIVE WORKFLOW
────────────────────────
1. State your approach and any assumptions or uncertainties.
2. Use <interact></interact> to gather evidence addressing those uncertainties.
3. Update your reasoning based on the evidence.
4. Repeat steps 2–3 if meaningful uncertainty remains.
5. ONLY when no critical uncertainty remains, produce the final solution.

────────────────────────
FINAL CODE REQUIREMENTS
────────────────────────
- The final code MUST be inside a ```python``` code block.
- The final code MUST read inputs from stdin and MUST NOT hardcode inputs.
- The final answer MUST clearly depend on interaction-generated evidence.
- Do NOT include <interact></interact> blocks after the final code.
- Make sure that the final code runs efficiently for the problem sizes described in the question.
"""

FINAL_PROMPT = """STOP. Do NOT use <interact> anymore. Your interaction budget is exhausted.

You MUST now output your final solution code wrapped in ```python``` code blocks.

Based on all the information and debugging you have done so far, write your best solution now.

Output ONLY the final ```python``` code block. No more <interact> blocks allowed."""


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


def postprocess_response(response: str) -> str:
    """Close unclosed <interact> tags cut off by the stop sequence."""
    if "<interact>" in response and "</interact>" not in response:
        response += "</interact>"
    return response


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


def load_problems_multiturn(
    dataset_name: str,
    start: int = 0,
    end: int | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    if dataset_name.startswith("bicycleman15/") or dataset_name.startswith("anirudhb11/intellect_"):
        full_ds = load_dataset(dataset_name, split="train")
    elif "lcb" in dataset_name:
        full_ds = load_dataset(dataset_name, split="test")
    else:
        full_ds = load_dataset(dataset_name, "code", split="train")

    if end is None:
        end = len(full_ds)
    sliced = full_ds.select(range(start, min(end, len(full_ds))))
    problems = [{"question": sliced[i]["question"], "_index": i} for i in range(len(sliced))]
    return sliced, problems


def run_prompt_optimization(
    hf_dataset: Any,
    problems: list[dict[str, Any]],
    dataset_name: str,
    solver_model: str,
    proposer_model: str,
    max_turns: int,
    interact_timeout_s: float,
    eval_timeout_s: float,
    max_tokens: int,
    objective: str | None,
    base_url: str,
    api_key: str,
    reflection_minibatch_size: int | None = None,
    max_metric_calls: int | None = DEFAULT_MAX_METRIC_CALLS,
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

        env = IntellectCodeEnv(
            system_prompt="",
            dataset_name=dataset_name,
            problem_index=example["_index"],
            max_turns=max_turns,
            dataset=hf_dataset,
            interaction_timeout_s=interact_timeout_s,
            eval_timeout_s=eval_timeout_s,
            interaction_mode=False,
        )
        try:
            obs, _ = env.reset()
        except Exception as e:
            oa.log(f"env.reset() failed: {e}")
            return 0.0

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        total_reward = 0.0
        interact_count = 0
        tle_count = 0
        all_assistant_responses: list[str] = []
        all_obs: list[str] = []
        terminated = False

        for turn in range(max_turns):
            is_last_turn = (turn == max_turns - 1)
            user_content = f"{obs}\n\n{FINAL_PROMPT}" if is_last_turn else obs
            all_obs.append(user_content)
            messages.append({"role": "user", "content": user_content})

            try:
                resp = client.chat.completions.create(
                    model=solver_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    stop=["</interact>"],
                )
                response = resp.choices[0].message.content or ""
            except Exception as e:
                oa.log(f"LLM call failed on turn {turn}: {e}")
                messages.append({"role": "assistant", "content": ""})
                break

            response = postprocess_response(response)
            messages.append({"role": "assistant", "content": response})
            all_assistant_responses.append(response)

            if "<interact>" in response:
                interact_count += 1

            obs, reward, terminated, truncated, info = env.step(response)
            total_reward = reward

            if info.get("final"):
                tle_count = info.get("eval_timeout_count", 0)

            if terminated or truncated:
                break

        n_tests = len(env.tests.get("inputs", []))
        passed = round(total_reward * n_tests)

        # Build conversation log: interleave assistant responses with interpreter outputs.
        turn_parts = []
        for i, assistant_resp in enumerate(all_assistant_responses):
            turn_parts.append(f"[Turn {i + 1} - Assistant]\n{assistant_resp}")
            if i < len(all_obs) - 1:
                turn_parts.append(f"[Turn {i + 1} - Interpreter output]\n{all_obs[i + 1]}")
        full_conversation = "\n\n---\n\n".join(turn_parts)

        # Log diagnostic info as ASI. Plain-text delimiters avoid breaking the proposer's ``` fence.
        sep = "=" * 40
        oa.log("\n".join([
            sep,
            f"RESULT: {passed}/{n_tests} tests passed | TLE: {tle_count} | Turns: {env.current_turn}/{max_turns} | Interacts: {interact_count} | PROBLEM_INDEX: {prob_idx}",
            sep,
            "",
            "QUESTION:",
            example["question"],
            "",
            "FULL CONVERSATION:",
            full_conversation,
        ]))

        turns_used = env.current_turn
        early_stop = terminated and turns_used < max_turns
        cand_type = "seed" if candidate_id == 1 else f"cand#{candidate_id}"
        extra = (
            f"  turns={turns_used}/{max_turns}  interacts={interact_count}  TLE={tle_count}"
            + (" [early-stop]" if early_stop else "")
        )

        call_num = budget_logger.log_metric(cand_hash, candidate_id, prob_idx, total_reward, extra) if budget_logger else "?"
        print(
            f"[metric #{call_num}] {cand_type}({cand_hash}) prob={prob_idx} → "
            f"score={total_reward:.4f} ({passed}/{n_tests}) turns={turns_used}/{max_turns} "
            f"interacts={interact_count} TLE={tle_count}" + (" [early-stop]" if early_stop else "")
        )
        return total_reward

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
            "Maximize the average final reward across all problems. "
            "The system prompt should guide the LLM to: (1) use <interact></interact> to test "
            "hypotheses and validate logic before submitting a final answer, (2) write correct, "
            "efficient Python that reads from stdin, enclosed in ```python``` blocks."
        ),
        background=(
            "We are optimizing a system prompt for a multi-turn coding agent that solves "
            "competitive programming problems in Python. The agent receives the problem statement "
            "and can execute code interactively via <interact></interact> blocks before submitting "
            "a final ```python``` solution that reads from stdin and writes to stdout."
        ),
    )


def dump_evolution(result: Any, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "evolution.json"), "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    key = result._str_candidate_key
    def _get_text(cand_dict: dict) -> str:
        return cand_dict[key] if key and key in cand_dict else json.dumps(cand_dict, indent=2)

    indices = sorted(range(result.num_candidates), key=lambda i: result.discovery_eval_counts[i])

    lines = [
        "=" * 70,
        f"CANDIDATE EVOLUTION  ({result.num_candidates} candidates)",
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
        description="GEPA optimize_anything: find the best multi-turn system prompt for LCB coding",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)

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
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--interact-timeout", type=float, default=DEFAULT_INTERACT_TIMEOUT_S, dest="interact_timeout_s")
    parser.add_argument("--eval-timeout", type=float, default=DEFAULT_EVAL_TIMEOUT_S, dest="eval_timeout_s")
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
        budget_log_path = os.path.join(args.output_dir, "budget_log.txt")
        budget_logger = BudgetLogger(budget_log_path, args.max_metric_calls, args.n_problems, append=is_resuming)
        print(f"[budget logger] → {budget_log_path}")
        install_proposer_logger(args.output_dir, budget_logger)

    print(f"Loading {args.n_problems} problems from {args.dataset} ...")
    hf_dataset, problems = load_problems_multiturn(args.dataset, start=0, end=args.n_problems)
    if not problems:
        print("ERROR: no problems loaded", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(problems)} problems.  Max turns: {args.max_turns}\n")

    result = run_prompt_optimization(
        hf_dataset=hf_dataset,
        problems=problems,
        dataset_name=args.dataset,
        solver_model=solver_model,
        proposer_model=args.proposer_model,
        max_turns=args.max_turns,
        interact_timeout_s=args.interact_timeout_s,
        eval_timeout_s=args.eval_timeout_s,
        max_tokens=args.max_tokens,
        objective=args.objective,
        base_url=base_url,
        api_key=args.api_key,
        reflection_minibatch_size=args.reflection_minibatch_size,
        max_metric_calls=args.max_metric_calls,
        budget_logger=budget_logger,
        output_dir=args.output_dir,
    )

    best_score = result.val_aggregate_scores[result.best_idx]
    print(f"\n{'='*70}")
    print(f"DONE  candidates={result.num_candidates}  metric_calls={result.total_metric_calls}"
          f"  best_idx={result.best_idx}  score={best_score:.4f}")
    print(f"{'='*70}\n")
    print("BEST SYSTEM PROMPT")
    print("─" * 70)
    print(result.best_candidate)
    print("─" * 70)

    if args.output_dir:
        dump_evolution(result, args.output_dir)


if __name__ == "__main__":
    main()
