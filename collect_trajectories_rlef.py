"""Collect RLEF multi-turn trajectories and save as HuggingFace dataset.

RLEF (Reinforcement Learning with Execution Feedback): the model generates
code, receives structured feedback from public test cases, and refines its
solution over multiple turns. Final solutions are scored against all tests.

Usage:
    python collect_trajectories_rlef.py \
    --dataset bicycleman15/intellect_3_code_very_hard \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --backend vllm \
    --start-problem 0 \
    --num-problems 10 \
    --num-samples 8 \
    --max-turns 3 \
    --num-public-tests 3 \
    --gpu-memory-utilization 0.75 \
    --eval-timeout-s 10.0 \
    --push-to-hub bicycleman15/temp

Multi-GPU (launches one vLLM server per GPU, shards prompts across them):
    python collect_trajectories_rlef.py \
    --dataset bicycleman15/intellect_3_code_very_hard \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --backend vllm \
    --vllm-multi-gpu \
    --vllm-gpu-ids 0,1 \
    --num-problems 10 \
    --num-samples 8 \
    --max-turns 3 \
    --num-public-tests 3 \
    --eval-timeout-s 10.0 \
    --push-to-hub bicycleman15/temp

For GPT-OSS models (uses Harmony format):
    python collect_trajectories_rlef.py \
    --dataset bicycleman15/intellect_3_code_very_hard \
    --model openai/gpt-oss-120b \
    --backend vllm \
    --num-problems 10 \
    --num-samples 8 \
    --max-turns 3 \
    --reasoning-effort medium \
    --push-to-hub bicycleman15/temp

Resume from checkpoint (if previous run failed):
    python collect_trajectories_rlef.py \
        --resume-from checkpoints/20260117_143052 \
        --dataset bicycleman15/intellect_3_code_very_hard \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        ... (same args as original run)

Checkpoints are automatically saved after each generation round to:
    checkpoints/<YYYYMMDD_HHMMSS>/checkpoint.pkl
    checkpoints/<YYYYMMDD_HHMMSS>/checkpoint_info.json
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import asyncio
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Optional

import requests
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from utils.harmony_utils import (
    is_gpt_oss_model,
    parse_harmony_response,
    load_harmony_encoding,
    HarmonyEncodingName,
    HarmonyRole,
    HarmonyMessage,
    Conversation,
    DeveloperContent,
    SystemContent,
    ReasoningEffort,
)

from checkpoint import CheckpointManager, get_checkpoint_dir
from rlef_env import RLEFCodeEnv, run_public_tests
from utils.fast_eval import EvalTask, evaluate_task, evaluate_tasks
from utils.gpu_keepalive import GPUKeepAlive
from utils.pass_at_k import compute_pass_at_k
from utils.vllm_multi_gpu import (
    resolve_vllm_gpu_ids,
    build_vllm_server_urls,
    launch_vllm_servers,
    wait_for_vllm_servers,
    register_vllm_shutdown,
)

try:
    import tinker
    from tinker import types as tinker_types
    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False

try:
    from chota_tinker import (
        SamplingClient,
        ServerSamplingClient,
        MultiServerSamplingClient,
        SamplingParams,
        ModelInput,
    )
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are an expert competitive programmer.

You will be given a programming problem along with public test cases.
Solve it in Python by submitting your solution inside a ```python``` code block.

Your code will be automatically tested against the public test cases. You will
receive structured feedback showing which tests passed or failed, including
the expected vs actual output for failures.

If any tests fail, carefully analyze the feedback, identify the bug or
misunderstanding, and submit an improved solution. You have multiple attempts.

RULES:
- Your code MUST read input from stdin and write output to stdout.
- Do NOT hardcode test inputs.
- Each response should contain exactly ONE ```python``` code block with your
  complete solution.
- Before the code block you may include a brief analysis of the problem or
  the feedback from the previous attempt.
"""

DEVELOPER_INSTRUCTIONS = SYSTEM_PROMPT

FINAL_PROMPT = """This is your LAST attempt. Submit your best solution now.

You MUST output your final solution code wrapped in a ```python``` code block."""


# =============================================================================
# HARMONY FORMAT HELPERS (for GPT-OSS models)
# =============================================================================

def build_harmony_conversation(history: list, obs: str, encoding) -> Conversation:
    """Build a Harmony Conversation from the history list and current observation."""
    messages = []

    for i, entry in enumerate(history):
        role = entry["role"]
        content = entry["content"]

        if role == "system":
            messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.SYSTEM, content))
        elif role == "developer":
            messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.DEVELOPER, content))
        elif role == "user":
            messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.USER, content))
        elif role == "assistant":
            channel = entry.get("channel", "final")

            should_drop_analysis = False
            if channel == "analysis":
                for j in range(i + 1, len(history)):
                    if history[j]["role"] == "assistant":
                        if history[j].get("channel") == "final":
                            should_drop_analysis = True
                        break

            if not should_drop_analysis:
                msg = HarmonyMessage.from_role_and_content(HarmonyRole.ASSISTANT, content)
                msg = msg.with_channel(channel)
                messages.append(msg)

    messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.USER, obs))
    return Conversation.from_messages(messages)


def render_trajectory(messages: list[dict], question: str, reward: float, num_turns: int, terminated: bool, truncated: bool) -> str:
    """Render a trajectory as a formatted string."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"Question: {question[:50]}..." if len(question) > 50 else f"Question: {question}")
    lines.append(f"Reward: {reward:.4f} | Turns: {num_turns} | Terminated: {terminated} | Truncated: {truncated}")
    lines.append("=" * 80)

    for msg in messages:
        role = msg['role'].upper()
        content = msg['content']
        lines.append(f"\n[{role}]\n{content}")

    return "\n".join(lines)


# =============================================================================
# ROLLOUT STATE
# =============================================================================

@dataclass
class RolloutState:
    """Track state of a single RLEF rollout for batched processing."""
    problem_index: int
    sample_index: int
    env: RLEFCodeEnv
    history: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    total_reward: float = 0.0
    obs: str = ""
    done: bool = False
    terminated: bool = False
    truncated: bool = False
    turn_wise_finish_reasons: list = field(default_factory=list)


def serialize_rollout_state(state: RolloutState) -> dict:
    """Serialize a RolloutState to a dictionary for checkpointing."""
    return {
        "problem_index": state.problem_index,
        "sample_index": state.sample_index,
        "history": [msg.copy() for msg in state.history],
        "messages": [msg.copy() for msg in state.messages],
        "total_reward": state.total_reward,
        "obs": state.obs,
        "done": state.done,
        "terminated": state.terminated,
        "truncated": state.truncated,
        "turn_wise_finish_reasons": list(state.turn_wise_finish_reasons),
        "question": state.env.question,
        "tests": state.env.tests,
        "current_turn": state.env.current_turn,
        "_last_valid_code": state.env._last_valid_code,
    }


def deserialize_rollout_state(data: dict, shared_dataset, args) -> RolloutState:
    """Deserialize a dictionary back to a RolloutState."""
    env = RLEFCodeEnv(
        system_prompt="",
        dataset_name=args.dataset,
        problem_index=data["problem_index"],
        max_turns=args.max_turns,
        num_public_tests=args.num_public_tests,
        eval_timeout_s=args.eval_timeout_s,
        dataset=shared_dataset,
    )
    env.reset()
    env.current_turn = data["current_turn"]
    env._last_valid_code = data.get("_last_valid_code")

    state = RolloutState(
        problem_index=data["problem_index"],
        sample_index=data["sample_index"],
        env=env,
        history=data["history"],
        messages=data["messages"],
        total_reward=data["total_reward"],
        obs=data["obs"],
        done=data["done"],
        terminated=data["terminated"],
        truncated=data["truncated"],
        turn_wise_finish_reasons=data.get("turn_wise_finish_reasons", []),
    )
    return state


# =============================================================================
# SAMPLING CLIENT + PARAMS
# =============================================================================

def create_sampling_client(args):
    """Create sampling client based on backend choice."""
    if args.backend == "tinker":
        if not TINKER_AVAILABLE:
            raise ImportError("tinker not installed. Install it or use --backend vllm")
        if args.vllm_multi_gpu:
            raise ValueError("--vllm-multi-gpu requires --backend vllm")
        service_client = tinker.ServiceClient()
        return service_client.create_sampling_client(base_model=args.model)
    else:
        if not VLLM_AVAILABLE:
            raise ImportError("chota_tinker not installed. Install it or use --backend tinker")
        if args.vllm_multi_gpu:
            if args.vllm_server_url:
                raise ValueError("--vllm-server-url cannot be used with --vllm-multi-gpu")
            gpu_ids = resolve_vllm_gpu_ids(args)
            urls = build_vllm_server_urls(args, gpu_ids)
            print(f"Launching vLLM servers for GPUs: {', '.join(gpu_ids)}")
            processes = launch_vllm_servers(args, gpu_ids)
            register_vllm_shutdown(processes)
            wait_for_vllm_servers(urls, args.vllm_server_startup_timeout_s)
            return MultiServerSamplingClient(urls)
        if args.vllm_server_url:
            return ServerSamplingClient(args.vllm_server_url)
        else:
            kwargs = {"gpu_memory_utilization": args.gpu_memory_utilization}
            if getattr(args, "max_model_len", None) is not None:
                kwargs["max_model_len"] = args.max_model_len
            if getattr(args, "tensor_parallel_size", 1) > 1:
                kwargs["tensor_parallel_size"] = args.tensor_parallel_size
            return SamplingClient(args.model, **kwargs)


def create_sampling_params(args, backend: str, harmony_encoding=None):
    """Create sampling params for the chosen backend (no </interact> stop)."""
    stop_token_ids = None

    if harmony_encoding is not None:
        stop_token_ids = harmony_encoding.stop_tokens_for_assistant_actions()

    if backend == "tinker":
        return tinker_types.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stop_token_ids=stop_token_ids,
        )
    else:
        return SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stop_token_ids=stop_token_ids,
        )


# =============================================================================
# PROMPT BUILDING
# =============================================================================

def build_prompt(state: RolloutState, tokenizer, max_turns: int) -> list[int]:
    """Build tokenized prompt from rollout state (standard HF tokenizer path)."""
    is_last_turn = state.env.current_turn == max_turns - 1
    if is_last_turn:
        obs_for_prompt = f"{state.obs}\n\n{FINAL_PROMPT}" if state.obs else FINAL_PROMPT
    else:
        obs_for_prompt = state.obs

    messages = state.history + [{"role": "user", "content": obs_for_prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(prompt_text)["input_ids"]


def build_prompt_harmony(state: RolloutState, encoding, max_turns: int) -> list[int]:
    """Build tokenized prompt from rollout state using Harmony encoding (GPT-OSS path)."""
    is_last_turn = state.env.current_turn == max_turns - 1
    if is_last_turn:
        obs_for_prompt = f"{state.obs}\n\n{FINAL_PROMPT}" if state.obs else FINAL_PROMPT
    else:
        obs_for_prompt = state.obs

    conversation = build_harmony_conversation(state.history, obs_for_prompt, encoding)
    return encoding.render_conversation_for_completion(conversation, HarmonyRole.ASSISTANT)


# =============================================================================
# BATCH SAMPLING
# =============================================================================

async def sample_batch_tinker(client, prompts: list[list[int]], sampling_params) -> list[tuple[list[int], Optional[str]]]:
    """Batch sample using tinker (via async gather). Returns list of (tokens, finish_reason)."""
    max_tokens = getattr(sampling_params, "max_tokens", None)

    async def sample_one(input_ids):
        result = await client.sample_async(
            prompt=tinker_types.ModelInput.from_ints(input_ids),
            sampling_params=sampling_params,
            num_samples=1,
        )
        seq = result.sequences[0]
        tokens = seq.tokens
        fr = getattr(seq, "finish_reason", None)
        if fr is None and max_tokens is not None and len(tokens) >= max_tokens:
            fr = "length"
        return (tokens, fr)

    results = await asyncio.gather(*[sample_one(p) for p in prompts])
    return results


async def sample_batch_tinker_harmony(client, prompts: list[list[int]], sampling_params, encoding) -> list[tuple[str, str, Optional[str], Optional[str]]]:
    """Batch sample using tinker with Harmony parsing.

    Returns list of (response_content, channel, analysis_content, finish_reason) tuples.
    """
    token_results = await sample_batch_tinker(client, prompts, sampling_params)

    parsed_results = []
    for tokens, finish_reason in token_results:
        content, channel, analysis = parse_harmony_response(tokens, encoding)
        parsed_results.append((content, channel, analysis, finish_reason))

    return parsed_results


def sample_batch_vllm(client, prompts: list[list[int]], sampling_params, show_progress: bool = False) -> list[tuple[str, Optional[str]]]:
    """Batch sample using vLLM. Returns list of (text, finish_reason)."""
    model_inputs = [ModelInput.from_ints(p) for p in prompts]
    max_tokens = getattr(sampling_params, "max_tokens", None)
    try:
        results = client.sample_batch(model_inputs, sampling_params, num_samples=1, show_progress=show_progress)
    except TypeError:
        results = client.sample_batch(model_inputs, sampling_params, num_samples=1)
    out = []
    for r in results:
        seq = r.sequences[0]
        text = seq.text
        fr = getattr(seq, "finish_reason", None)
        if fr is None and max_tokens is not None and len(seq.tokens) >= max_tokens:
            fr = "length"
        out.append((text, fr))
    return out


def sample_batch_vllm_harmony(client, prompts: list[list[int]], sampling_params, encoding, show_progress: bool = False) -> list[tuple[str, str, Optional[str], Optional[str]]]:
    """Batch sample using vLLM with Harmony parsing.

    Returns list of (response_content, channel, analysis_content, finish_reason) tuples.
    """
    model_inputs = [ModelInput.from_ints(p) for p in prompts]
    max_tokens = getattr(sampling_params, "max_tokens", None)
    try:
        results = client.sample_batch(model_inputs, sampling_params, num_samples=1, show_progress=show_progress)
    except TypeError:
        results = client.sample_batch(model_inputs, sampling_params, num_samples=1)

    parsed_results = []
    for result in results:
        seq = result.sequences[0]
        tokens = seq.tokens
        fr = getattr(seq, "finish_reason", None)
        if fr is None and max_tokens is not None and len(tokens) >= max_tokens:
            fr = "length"
        content, channel, analysis = parse_harmony_response(tokens, encoding)
        parsed_results.append((content, channel, analysis, fr))

    return parsed_results


# =============================================================================
# BATCHED RLEF STEPPING (public tests + final eval via process pools)
# =============================================================================

def step_rlef_batch(
    envs: list[RLEFCodeEnv],
    actions: list[str],
    eval_workers: int = 16,
    eval_batch_size: int = 8,
    eval_timeout_s: float = 10.0,
    show_progress: bool = False,
) -> list[tuple[str, float, bool, bool, dict[str, Any]]]:
    """Batch-step RLEF environments with pooled public test + final evaluation.

    Phase 1: Extract code from all actions (cheap, no subprocess).
    Phase 2: Batch all public test evaluations via ProcessPoolExecutor
             (one subprocess per rollout, all tests in one harness).
    Phase 3: Process results — format feedback or mark for final eval.
    Phase 4: Batch all final evaluations via evaluate_tasks (persistent pool).
    """
    n = len(envs)
    results: list[Optional[tuple[str, float, bool, bool, dict[str, Any]]]] = [None] * n

    # Phase 1 — extract code, handle no-code cases
    public_test_tasks: list[tuple[int, str]] = []
    for i, (env, action) in enumerate(zip(envs, actions)):
        env.current_turn += 1
        code = env._extract_answer_code(action)
        if not code:
            results[i] = env._handle_no_code()
            continue
        env._last_valid_code = code
        public_test_tasks.append((i, code))

    # Phase 2 — batch public test evaluation
    if public_test_tasks:
        pool_args = [
            (code, envs[i].public_tests, envs[i].eval_timeout_s or eval_timeout_s)
            for i, code in public_test_tasks
        ]
        with ProcessPoolExecutor(max_workers=min(eval_workers, len(pool_args))) as pool:
            public_results_list = list(
                tqdm(
                    pool.map(
                        _run_public_tests_star, pool_args,
                        chunksize=max(1, len(pool_args) // (eval_workers * 2)),
                    ),
                    total=len(pool_args),
                    desc="Public tests",
                    disable=not show_progress,
                )
            )

        # Phase 3 — process public test results
        for (idx, code), test_results in zip(public_test_tasks, public_results_list):
            env = envs[idx]
            all_passed = all(r["passed"] for r in test_results)

            if all_passed:
                results[idx] = ("", 0.0, True, False, {
                    "final": True, "public_all_passed": True,
                    "needs_eval": True, "code": code,
                })
            elif env.current_turn >= env.max_turns:
                results[idx] = ("", 0.0, True, False, {
                    "final": True, "public_all_passed": False,
                    "needs_eval": True, "code": code,
                })
            else:
                feedback = env._format_feedback(test_results)
                results[idx] = (feedback, 0.0, False, False, {"public_all_passed": False})

    # Phase 4 — batch final evaluations
    eval_tasks: list[EvalTask] = []
    eval_indices: list[int] = []
    for i, result in enumerate(results):
        if result is None:
            raise RuntimeError(f"Missing step result for index {i}")
        _, _, _, _, info = result
        if info.get("needs_eval") and info.get("code"):
            eval_tasks.append(EvalTask(
                response=f"```python\n{info['code']}\n```",
                tests=envs[i].private_tests,
                max_tests=envs[i].max_tests,
                timeout_s=eval_timeout_s,
                require_solution_class=True,
            ))
            eval_indices.append(i)

    if eval_tasks:
        if len(eval_tasks) == 1:
            eval_results_list = [evaluate_task(eval_tasks[0])]
        else:
            eval_results_list = evaluate_tasks(
                eval_tasks,
                max_workers=eval_workers,
                batch_size=eval_batch_size,
                show_progress=show_progress or len(eval_tasks) > 4,
            )
        for idx, eval_result in zip(eval_indices, eval_results_list):
            obs, _, terminated, truncated, info = results[idx]
            results[idx] = (obs, eval_result.reward, terminated, truncated, info)

    return results


def _run_public_tests_star(args: tuple) -> list[dict[str, Any]]:
    """Unpack tuple for ProcessPoolExecutor.map()."""
    return run_public_tests(*args)


# =============================================================================
# UTILS
# =============================================================================

def _truncated_by_token_limit(finish_reason: str | None) -> bool:
    """True if generation stopped due to max token limit."""
    return (finish_reason or "").lower() in ("length", "length_capped", "max_tokens")


# =============================================================================
# MAIN ROLLOUT LOOP
# =============================================================================

def run_batched_rollouts(
    args,
    client,
    tokenizer_or_encoding,
    sampling_params,
    checkpoint_manager: Optional[CheckpointManager] = None,
    use_harmony: bool = False,
) -> list[list[dict[str, Any]]]:
    """Run batched RLEF rollouts across all problems and samples."""
    print(f"Loading dataset {args.dataset}...")
    if args.dataset.startswith("bicycleman15/") or args.dataset.startswith("anirudhb11/intellect_"):
        from datasets import load_dataset
        full_dataset = load_dataset(args.dataset, split="train")
    elif args.dataset.__contains__('lcb'):
        from datasets import load_dataset
        full_dataset = load_dataset(args.dataset, split="test")
    else:
        from datasets import load_dataset
        full_dataset = load_dataset(args.dataset, "code", split="train")

    end_problem = min(args.start_problem + args.num_problems, len(full_dataset))
    shared_dataset = full_dataset.select(range(args.start_problem, end_problem))
    actual_num_problems = len(shared_dataset)
    print(f"Dataset loaded with {len(full_dataset)} total problems.")
    print(f"Selected slice: problems {args.start_problem} to {end_problem - 1} ({actual_num_problems} problems)")

    if actual_num_problems < args.num_problems:
        print(f"Warning: Requested {args.num_problems} problems but only {actual_num_problems} available in slice.")
        args.num_problems = actual_num_problems

    if use_harmony:
        harmony_system_content = (
            SystemContent.new()
            .with_reasoning_effort(ReasoningEffort[args.reasoning_effort.upper()])
            .with_conversation_start_date(date.today().isoformat())
        )
        harmony_developer_content = (
            DeveloperContent.new()
            .with_instructions(DEVELOPER_INSTRUCTIONS)
        )

    generation_round = 0
    active_states: list[RolloutState] = []
    completed_states: list[RolloutState] = []
    skip_generation = False

    if checkpoint_manager and checkpoint_manager.has_checkpoint():
        print(f"\nResuming from checkpoint: {checkpoint_manager.checkpoint_dir}")
        checkpoint_data = checkpoint_manager.load()

        for warning in checkpoint_manager.verify_args({
            "start_problem": args.start_problem,
            "num_problems": args.num_problems,
            "num_samples": args.num_samples,
            "dataset": args.dataset,
            "model": args.model,
        }):
            print(warning)

        generation_round = checkpoint_data.current_round
        print(f"  Resuming from generation round {generation_round}")
        print(f"  Restoring {len(checkpoint_data.active_states_data)} active states...")
        print(f"  Restoring {len(checkpoint_data.completed_states_data)} completed states...")

        for state_data in checkpoint_data.active_states_data:
            state = deserialize_rollout_state(state_data, shared_dataset, args)
            active_states.append(state)

        for state_data in checkpoint_data.completed_states_data:
            state = deserialize_rollout_state(state_data, shared_dataset, args)
            completed_states.append(state)

        print(f"  Successfully restored {len(active_states)} active and {len(completed_states)} completed states.")
        skip_generation = True
    else:
        for problem_idx in range(args.num_problems):
            for sample_idx in range(args.num_samples):
                env = RLEFCodeEnv(
                    system_prompt="",
                    dataset_name=args.dataset,
                    problem_index=problem_idx,
                    max_turns=args.max_turns,
                    num_public_tests=args.num_public_tests,
                    eval_timeout_s=args.eval_timeout_s,
                    dataset=shared_dataset,
                )
                obs, info = env.reset()

                if use_harmony:
                    history = [
                        {"role": "system", "content": harmony_system_content},
                        {"role": "developer", "content": harmony_developer_content},
                    ]
                    messages = [
                        {"role": "system", "content": f"[Harmony format] Reasoning effort: {args.reasoning_effort}"},
                        {"role": "developer", "content": DEVELOPER_INSTRUCTIONS},
                        {"role": "user", "content": obs},
                    ]
                else:
                    history = [{"role": "system", "content": SYSTEM_PROMPT}]
                    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": obs}]

                state = RolloutState(
                    problem_index=problem_idx,
                    sample_index=sample_idx,
                    env=env,
                    history=history,
                    messages=messages,
                    obs=obs,
                )
                active_states.append(state)

    problems_completed = set()
    for state in completed_states:
        problem_samples_done = sum(1 for s in completed_states if s.problem_index == state.problem_index)
        if problem_samples_done == args.num_samples:
            problems_completed.add(state.problem_index)

    pbar = tqdm(total=args.num_problems, desc="Problems completed", initial=len(problems_completed))

    while active_states:
        if skip_generation:
            print(f"\n[Resuming round {generation_round}] Skipping generation, going directly to env step...")
            processed_responses = []
            for state in active_states:
                last_assistant_msg = None
                for msg in reversed(state.history):
                    if msg["role"] == "assistant":
                        last_assistant_msg = msg["content"]
                        break
                if last_assistant_msg is None:
                    raise ValueError(f"No assistant response found in history for state {state.problem_index}:{state.sample_index}")
                processed_responses.append(last_assistant_msg)
            skip_generation = False
        else:
            generation_round += 1
            print(f"\n[Generation round {generation_round}] Processing {len(active_states)} active states...")

            if use_harmony:
                prompts = [build_prompt_harmony(s, tokenizer_or_encoding, args.max_turns) for s in active_states]
            else:
                prompts = [build_prompt(s, tokenizer_or_encoding, args.max_turns) for s in active_states]

            if use_harmony:
                if args.backend == "tinker":
                    harmony_results = asyncio.run(sample_batch_tinker_harmony(
                        client, prompts, sampling_params, tokenizer_or_encoding
                    ))
                else:
                    harmony_results = sample_batch_vllm_harmony(
                        client, prompts, sampling_params, tokenizer_or_encoding, show_progress=args.vllm_multi_gpu
                    )
                responses = harmony_results
            elif args.backend == "tinker":
                token_results = asyncio.run(sample_batch_tinker(client, prompts, sampling_params))
                responses = [(tokenizer_or_encoding.decode(tokens, skip_special_tokens=True), fr) for tokens, fr in token_results]
            else:
                responses = sample_batch_vllm(client, prompts, sampling_params, show_progress=args.vllm_multi_gpu)

            processed_responses = []
            for i, state in enumerate(active_states):
                if use_harmony:
                    response, channel, analysis, finish_reason = responses[i]
                else:
                    response, finish_reason = responses[i]
                    channel = None
                    analysis = None

                state.turn_wise_finish_reasons.append(finish_reason)

                is_last_turn = state.env.current_turn == args.max_turns - 1
                if is_last_turn:
                    obs_for_history = f"{state.obs}\n\n{FINAL_PROMPT}" if state.obs else FINAL_PROMPT
                    if state.messages and state.messages[-1]["role"] == "user":
                        state.messages[-1]["content"] = obs_for_history
                else:
                    obs_for_history = state.obs

                state.history.append({"role": "user", "content": obs_for_history})

                if use_harmony:
                    if analysis and channel == "final":
                        state.history.append({"role": "assistant", "content": analysis, "channel": "analysis"})
                    state.history.append({"role": "assistant", "content": response, "channel": channel})

                    if analysis:
                        state.messages.append({
                            "role": "assistant",
                            "content": f"[Analysis (internal CoT)]\n{analysis}\n\n[Response (channel: {channel})]\n{response}"
                        })
                    else:
                        state.messages.append({"role": "assistant", "content": response})
                else:
                    state.history.append({"role": "assistant", "content": response})
                    state.messages.append({"role": "assistant", "content": response})

                processed_responses.append(response)

            if checkpoint_manager:
                checkpoint_manager.save(
                    active_states_data=[serialize_rollout_state(s) for s in active_states],
                    completed_states_data=[serialize_rollout_state(s) for s in completed_states],
                    current_round=generation_round,
                    total_rounds=args.max_turns * args.num_problems * args.num_samples,
                )

        still_active = []

        with GPUKeepAlive():
            step_results = step_rlef_batch(
                [s.env for s in active_states],
                processed_responses,
                eval_workers=args.eval_workers,
                eval_batch_size=args.eval_batch_size,
                eval_timeout_s=args.eval_timeout_s,
                show_progress=True,
            )

        for state, _response, (obs, reward, terminated, truncated, info) in zip(
            active_states, processed_responses, step_results
        ):
            state.total_reward += reward
            state.terminated = terminated
            state.truncated = truncated

            if obs:
                state.messages.append({"role": "user", "content": obs})

            if terminated or truncated:
                state.done = True
                completed_states.append(state)

                problem_samples_done = sum(
                    1 for s in completed_states if s.problem_index == state.problem_index
                )
                if problem_samples_done == args.num_samples and state.problem_index not in problems_completed:
                    problems_completed.add(state.problem_index)
                    pbar.update(1)
            else:
                state.obs = obs
                still_active.append(state)

        active_states = still_active

    pbar.close()

    all_trajectories: list[list[dict]] = [[] for _ in range(args.num_problems)]
    for state in completed_states:
        turn_wise_finish_reason = list(state.turn_wise_finish_reasons)
        turn_wise_truncated_by_token_limit = [_truncated_by_token_limit(fr) for fr in state.turn_wise_finish_reasons]
        num_assistant_messages_truncated = sum(turn_wise_truncated_by_token_limit)
        num_assistant_messages_not_truncated = len(turn_wise_truncated_by_token_limit) - num_assistant_messages_truncated
        traj = {
            "question": state.env.question,
            "messages": state.messages,
            "num_turns": state.env.current_turn,
            "final_reward": state.total_reward,
            "terminated": state.terminated,
            "truncated": state.truncated,
            "tests": state.env.tests,
            "num_public_tests": len(state.env.public_tests.get("inputs", [])),
            "turn_wise_finish_reason": turn_wise_finish_reason,
            "turn_wise_truncated_by_token_limit": turn_wise_truncated_by_token_limit,
            "num_assistant_messages_truncated": num_assistant_messages_truncated,
            "num_assistant_messages_not_truncated": num_assistant_messages_not_truncated,
        }
        all_trajectories[state.problem_index].append(traj)

    return all_trajectories


def main(args):
    print(f"=" * 60)
    print(f"Collecting RLEF trajectories")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model}")
    print(f"  Backend: {args.backend}")
    if is_gpt_oss_model(args.model):
        print(f"  Format: Harmony (GPT-OSS)")
        print(f"  Reasoning effort: {args.reasoning_effort}")
    print(f"  Problem range: {args.start_problem} to {args.start_problem + args.num_problems - 1} ({args.num_problems} problems)")
    print(f"  Samples per problem: {args.num_samples}")
    print(f"  Max turns: {args.max_turns}")
    print(f"  Public tests: {args.num_public_tests}")
    print(f"  Eval timeout: {args.eval_timeout_s}s")
    print(f"  Output: {args.output_dir}")
    if args.resume_from:
        print(f"  Resuming from: {args.resume_from}")
    print(f"=" * 60)

    if args.resume_from:
        checkpoint_dir = args.resume_from
        print(f"\nResuming from checkpoint directory: {checkpoint_dir}")
    else:
        checkpoint_dir = get_checkpoint_dir()
        print(f"\nCheckpoint directory: {checkpoint_dir}")

    checkpoint_manager = CheckpointManager(
        checkpoint_dir,
        args_dict={
            "dataset": args.dataset,
            "model": args.model,
            "start_problem": args.start_problem,
            "num_problems": args.num_problems,
            "num_samples": args.num_samples,
            "max_turns": args.max_turns,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_public_tests": args.num_public_tests,
            "eval_timeout_s": args.eval_timeout_s,
        }
    )

    use_harmony = is_gpt_oss_model(args.model)

    if use_harmony:
        print(f"\nDetected GPT-OSS model: {args.model}")
        print(f"Using Harmony format with reasoning effort: {args.reasoning_effort}")
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        tokenizer_or_encoding = encoding
        sampling_params = create_sampling_params(args, args.backend, harmony_encoding=encoding)
    else:
        print(f"\nUsing standard model: {args.model}")
        tokenizer_or_encoding = AutoTokenizer.from_pretrained(args.model)
        sampling_params = create_sampling_params(args, args.backend)

    print(f"\nInitializing {args.backend} client...")
    sampling_client = create_sampling_client(args)

    print(f"\nCollecting RLEF trajectories for {args.num_problems} problems (batched)...")

    all_trajectories = run_batched_rollouts(
        args=args,
        client=sampling_client,
        tokenizer_or_encoding=tokenizer_or_encoding,
        sampling_params=sampling_params,
        checkpoint_manager=checkpoint_manager,
        use_harmony=use_harmony,
    )

    rows = []
    all_results = []

    for problem_idx, problem_trajectories in enumerate(all_trajectories):
        problem_results = []

        for traj_idx, traj in enumerate(problem_trajectories):
            is_successful = traj["final_reward"] == 1.0
            problem_results.append(is_successful)

            rows.append({
                "problem_id": problem_idx,
                "trajectory_id": traj_idx,
                "question": traj["question"],
                "messages": json.dumps(traj["messages"]),
                "num_turns": traj["num_turns"],
                "final_reward": traj["final_reward"],
                "terminated": traj["terminated"],
                "truncated": traj["truncated"],
                "tests": json.dumps(traj["tests"]),
                "num_public_tests": traj["num_public_tests"],
                "is_successful": is_successful,
                "turn_wise_finish_reason": traj["turn_wise_finish_reason"],
                "turn_wise_truncated_by_token_limit": traj["turn_wise_truncated_by_token_limit"],
                "num_assistant_messages_truncated": traj["num_assistant_messages_truncated"],
                "num_assistant_messages_not_truncated": traj["num_assistant_messages_not_truncated"],
                "rendered": render_trajectory(
                    traj["messages"], traj["question"],
                    traj["final_reward"], traj["num_turns"], traj["terminated"], traj["truncated"]
                ),
            })

        all_results.append(problem_results)

    pass_at_1 = compute_pass_at_k(all_results, k=1)
    pass_at_2 = compute_pass_at_k(all_results, k=2)
    pass_at_4 = compute_pass_at_k(all_results, k=4)
    pass_at_8 = compute_pass_at_k(all_results, k=min(8, args.num_samples))

    avg_reward = sum(r["final_reward"] for r in rows) / len(rows) if rows else 0.0

    print(f"\n{'=' * 60}")
    print(f"Results:")
    print(f"  Total trajectories: {len(rows)}")
    print(f"  Avg reward: {avg_reward:.4f}")
    print(f"  pass@1: {pass_at_1:.4f}")
    print(f"  pass@2: {pass_at_2:.4f}")
    print(f"  pass@4: {pass_at_4:.4f}")
    print(f"  pass@8: {pass_at_8:.4f}")
    print(f"{'=' * 60}")

    dataset = Dataset.from_list(rows)
    metadata = {
        "dataset": args.dataset,
        "model": args.model,
        "start_problem": args.start_problem,
        "num_problems": args.num_problems,
        "num_samples": args.num_samples,
        "max_turns": args.max_turns,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_public_tests": args.num_public_tests,
        "eval_timeout_s": args.eval_timeout_s,
        "timestamp": datetime.now().isoformat(),
        "avg_reward": avg_reward,
        "pass_at_1": pass_at_1,
        "pass_at_2": pass_at_2,
        "pass_at_4": pass_at_4,
        "pass_at_8": pass_at_8,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    dataset.save_to_disk(args.output_dir)

    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved dataset to: {args.output_dir}")
    print(f"Saved metadata to: {metadata_path}")

    summary_path = os.path.join(args.output_dir, "summary.json")
    summary = {
        **metadata,
        "num_successful_trajectories": sum(1 for r in rows if r["is_successful"]),
        "problems_solved": sum(1 for pr in all_results if any(pr)),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary to: {summary_path}")

    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.push_to_hub}")
        dataset.push_to_hub(args.push_to_hub, private=False)
        print(f"Successfully pushed to: https://huggingface.co/datasets/{args.push_to_hub}")

    return dataset, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect RLEF multi-turn trajectories for code problems")
    parser.add_argument("--dataset", type=str, default="bicycleman15/intellect_3_code_easy_medium",
                        choices=["bicycleman15/intellect_3_code_easy_medium", "bicycleman15/intellect_3_code_hard",
                                 "bicycleman15/intellect_3_code_very_hard", "PrimeIntellect/INTELLECT-3-RL",
                                 "anirudhb11/lcb_v6_feb_may_2025_formatted", "anirudhb11/lcb_v6_feb_may_2025_formatted_hardest_to_easiest",
                                 "anirudhb11/intellect_3_code_very_hard_top_400_hardest",
                                 "anirudhb11/qwen3_4b_instruct_top_400_hardest_interations_10_turns",
                                 "anirudhb11/lcbv6-st-160x8k-solves-mt-32x4kx10-not-solve"])
    parser.add_argument("--start-problem", type=int, default=0,
                        help="Starting problem index for dataset slicing (default: 0)")
    parser.add_argument("--num-problems", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0, dest="top_p")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output-dir", type=str, default="artifacts/trajectories_rlef")
    parser.add_argument("--push-to-hub", type=str, default=None, help="HF repo to push to (e.g. username/repo-name)")

    # RLEF-specific
    parser.add_argument("--num-public-tests", type=int, default=3,
                        help="Number of public test cases for execution feedback (default: 3)")
    parser.add_argument("--eval-timeout-s", type=float, default=10.0,
                        help="Per-test timeout in seconds for code execution (default: 10.0)")
    parser.add_argument("--eval-batch-size", type=int, default=8,
                        help="Number of evaluations per worker batch for final eval (default: 8)")

    # Checkpointing
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint directory to resume from (e.g. checkpoints/20260117_143052)")

    # Backend options
    parser.add_argument("--backend", type=str, default="vllm", choices=["tinker", "vllm"],
                        help="Inference backend: 'tinker' or 'vllm' (default: vllm)")
    parser.add_argument("--vllm-server-url", type=str, default=None,
                        help="URL for vLLM server (e.g. http://localhost:8000). If not set, uses local vLLM.")
    parser.add_argument("--vllm-multi-gpu", action="store_true",
                        help="Launch one local vLLM server per GPU and shard prompts across them.")
    parser.add_argument("--vllm-gpu-ids", type=str, default=None,
                        help="Comma-separated GPU IDs for vLLM servers (default: all visible GPUs).")
    parser.add_argument("--vllm-server-base-port", type=int, default=8000,
                        help="Base port for vLLM servers; ports increment per GPU.")
    parser.add_argument("--vllm-server-host", type=str, default="127.0.0.1",
                        help="Host to bind vLLM servers (default: 127.0.0.1).")
    parser.add_argument("--vllm-server-startup-timeout-s", type=float, default=300.0,
                        help="Seconds to wait for vLLM servers to be ready.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="GPU memory utilization for local vLLM or vLLM servers (default: 0.9)")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Max sequence length for vLLM; lower than model default to reduce KV cache")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism (default: 1)")
    parser.add_argument("--eval-workers", type=int, default=16,
                        help="Number of parallel workers for RLEF env stepping (default: 16)")

    # Harmony/GPT-OSS options
    parser.add_argument("--reasoning-effort", type=str, default="medium",
                        choices=["low", "medium", "high"],
                        help="Reasoning effort for GPT-OSS models using Harmony format (default: medium)")

    args = parser.parse_args()
    main(args)
