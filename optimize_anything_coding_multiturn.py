"""
Use GEPA's optimize_anything framework to optimize a system prompt that boosts
an LLM's multi-turn coding accuracy across LCB problems.

Like optimize_anything_coding.py but uses IntellectCodeEnv for multi-turn
<interact></interact>-based code execution trajectories.

The candidate is a system prompt.  For each problem the evaluator:
  1. Resets IntellectCodeEnv for that problem.
  2. Runs a multi-turn loop (up to --max-turns steps):
     a. Calls the solver LLM with stop=["</interact>"].
     b. Postprocesses the response (closes unclosed <interact> tags).
     c. Calls env.step(response).
     d. On the last turn, appends FINAL_PROMPT to force final code output.
  3. Returns the final reward from the env (0-1 fraction of test cases passed).
  4. Logs turn count, interact count, reward, and code preview as ASI.

GEPA then iterates, proposing improved system prompts based on what went wrong.

Usage:
  python optimize_anything_coding_multiturn.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --n-problems 8 --max-turns 5 \
    --reflection-minibatch-size 4 \
    --proposer-model anthropic/claude-sonnet-4-6 \
    --output-dir ./run_multiturn_003 \
    --max-metric-calls 80

Prerequisites:
  pip install litellm           # GEPA uses litellm for its internal proposer LLM

  # Set the API key matching your --proposer-model:
  export ANTHROPIC_API_KEY=sk-ant-...       # for anthropic/claude-sonnet-4-6, etc.
  export OPENAI_API_KEY=sk-...              # for openai/gpt-4o, openai/gpt-4o-mini, etc.
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
from intellect_env import IntellectCodeEnv
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
DEFAULT_MAX_TURNS = 5
DEFAULT_INTERACT_TIMEOUT_S = 10.0
DEFAULT_EVAL_TIMEOUT_S = 5.0
DEFAULT_PROPOSER_MODEL = "openai/gpt-4o-mini"
DEFAULT_N_PROBLEMS = 10
DEFAULT_MAX_METRIC_CALLS = None  # None → auto: n_problems * 10

# Seed system prompt — the full multi-turn system prompt from collect_trajectories.py.
# GEPA will iteratively improve it based on failures.
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
# Response postprocessing
# --------------------------------------------------------------------------- #

def postprocess_response(response: str) -> str:
    """Close unclosed <interact> tags (the stop sequence cuts off the closing tag)."""
    if "<interact>" in response and "</interact>" not in response:
        response += "</interact>"
    return response


# --------------------------------------------------------------------------- #
# vLLM engine  (same helpers as optimize_anything_coding.py)
# --------------------------------------------------------------------------- #

def launch_vllm_engine(args) -> tuple[str, str]:
    """Launch a vLLM server subprocess and return (openai_base_url, model_name)."""
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

    openai_base_url = server_urls[0].rstrip("/") + "/v1"
    return openai_base_url, args.model


def connect_vllm_server(vllm_url: str, solver_model: str | None) -> tuple[str, str]:
    """Connect to an already-running vLLM server and return (openai_base_url, model_name)."""
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

def load_problems_multiturn(
    dataset_name: str,
    start: int = 0,
    end: int | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Load dataset slice.

    Returns (hf_dataset_slice, problems) where problems is a list of
    {"question": ..., "_index": i} dicts and _index is the 0-based index
    within the returned slice (used to initialize IntellectCodeEnv).
    """
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


# --------------------------------------------------------------------------- #
# Prompt optimization
# --------------------------------------------------------------------------- #

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
) -> Any:
    """Optimize a system prompt across N problems using multi-turn trajectories."""
    client = OpenAI(api_key=api_key, base_url=base_url)

    def evaluator(system_prompt: str, example: dict[str, Any]) -> float:
        env = IntellectCodeEnv(
            system_prompt="",         # system prompt lives in messages history
            dataset_name=dataset_name,
            problem_index=example["_index"],
            max_turns=max_turns,
            dataset=hf_dataset,
            interaction_timeout_s=interact_timeout_s,
            eval_timeout_s=eval_timeout_s,
            interaction_mode=False,   # system prompt already contains interaction rules
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

        for turn in range(max_turns):
            # On the last turn append FINAL_PROMPT to force final code submission.
            is_last_turn = (turn == max_turns - 1)
            user_content = f"{obs}\n\n{FINAL_PROMPT}" if is_last_turn else obs
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

        # Compute passed count from reward and total test cases.
        n_tests = len(env.tests.get("inputs", []))
        passed = round(total_reward * n_tests)

        # Log diagnostic info as ASI so GEPA can read it when proposing improvements.
        full_conversation = "\n\n---\n\n".join(
            f"[Turn {i + 1}]\n{r}" for i, r in enumerate(all_assistant_responses)
        )
        asi_parts = [
            f"Turns used: {env.current_turn}/{max_turns}",
            f"Interact blocks: {interact_count}",
            f"Test results: {passed}/{n_tests} passed. TLE: {tle_count}.",
            f"\nQuestion:\n{example['question']}",
            f"\nFull LLM responses:\n{full_conversation}",
        ]
        oa.log("\n".join(asi_parts))

        return total_reward

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


# --------------------------------------------------------------------------- #
# Evolution dump
# --------------------------------------------------------------------------- #

def dump_evolution(result: Any, output_dir: str) -> None:
    """Write candidate evolution to output_dir after optimization completes."""
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "evolution.json")
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"Evolution JSON written to {json_path}")

    key = result._str_candidate_key

    def _get_text(cand_dict: dict) -> str:
        if key and key in cand_dict:
            return cand_dict[key]
        return json.dumps(cand_dict, indent=2)

    indices = sorted(range(result.num_candidates), key=lambda i: result.discovery_eval_counts[i])

    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"CANDIDATE EVOLUTION  ({result.num_candidates} candidates)")
    lines.append(f"Total metric calls: {result.total_metric_calls}")
    lines.append(
        f"Best candidate idx: {result.best_idx}  "
        f"(score={result.val_aggregate_scores[result.best_idx]:.4f})"
    )
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
        description="GEPA optimize_anything: find the best multi-turn system prompt for LCB coding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="HuggingFace dataset (default: %(default)s)")

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
                        help="Max tokens per LLM call per turn (default: %(default)s)")

    # ---- task ---------------------------------------------------------------
    parser.add_argument("--n-problems", type=int, default=DEFAULT_N_PROBLEMS,
                        help="Number of problems to evaluate per candidate (default: %(default)s)")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS,
                        help="Max interaction turns per problem per evaluation (default: %(default)s)")
    parser.add_argument("--interact-timeout", type=float, default=DEFAULT_INTERACT_TIMEOUT_S,
                        dest="interact_timeout_s",
                        help="Timeout in seconds for each <interact> code execution (default: %(default)s)")
    parser.add_argument("--eval-timeout", type=float, default=DEFAULT_EVAL_TIMEOUT_S,
                        dest="eval_timeout_s",
                        help="Per-test-case timeout for final evaluation (default: %(default)s)")
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
                             "(default: 3). Lower = faster; higher = richer feedback.")
    parser.add_argument("--output-dir", default=None,
                        help="If set, dump evolution.json and evolution.txt here after optimization")

    args = parser.parse_args()

    # Auto-scale max_metric_calls if not explicitly set.
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

    # ---- Load dataset -------------------------------------------------------
    print(f"Loading {args.n_problems} problem(s) from {args.dataset} ...")
    hf_dataset, problems = load_problems_multiturn(args.dataset, start=0, end=args.n_problems)
    if not problems:
        print("ERROR: no problems loaded", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(problems)} problems.")
    print(f"Max turns per trajectory: {args.max_turns}")
    print(f"\nSeed system prompt (first 200 chars):\n{'-'*50}")
    print(SEED_SYSTEM_PROMPT[:200] + "...")
    print(f"{'-'*50}\n")

    # ---- Run optimization --------------------------------------------------
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
    )

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    best_score = result.val_aggregate_scores[result.best_idx]
    print(f"Candidates explored : {result.num_candidates}")
    print(f"Total metric calls  : {result.total_metric_calls}")
    print(f"Best candidate idx  : {result.best_idx}")
    print(f"Best score          : {best_score:.4f}")

    print(f"\n{'─' * 70}")
    print("FINAL SYSTEM PROMPT")
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
        dump_evolution(result, args.output_dir)


if __name__ == "__main__":
    main()
