"""Collect single-turn trajectories and save as HuggingFace dataset.

Usage:

    python collect_trajectories_single_turn.py \
    --dataset anirudhb11/lcb_v6_feb_may_2025_formatted \
    --model openai/gpt-oss-120b \
    --backend vllm \
    --start-problem 0 \
    --num-problems 2 \
    --num-samples 32 \
    --max-tokens 8192 \
    --gpu-memory-utilization 0.8 \
    \
    --fast-eval \
    --eval-workers 8 \
    --eval-batch-size 8 \
    --eval-timeout-s 5.0 \
    --push-to-hub bicycleman15/temp

Multi-GPU (launches one vLLM server per GPU, shards prompts across them):
    python collect_trajectories_single_turn.py \
    --dataset bicycleman15/intellect_3_code_very_hard \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --backend vllm \
    --vllm-multi-gpu \
    --vllm-gpu-ids 0,1,2,3 \
    --start-problem 100 \
    --num-problems 50 \
    --num-samples 32 \
    \
    --fast-eval \
    --eval-workers 16 \
    --eval-batch-size 8 \
    --eval-timeout-s 1.0 \
    --push-to-hub bicycleman15/qwen3_4b_instruct_easy_medium_single_turn

For GPT-OSS models (uses Harmony format):
    python collect_trajectories_single_turn.py \
    --dataset bicycleman15/intellect_3_code_very_hard \
    --model openai/gpt-oss-120b \
    --backend vllm \
    --tensor-parallel-size 4 \
    --num-problems 10 \
    --num-samples 32 \
    --reasoning-effort medium \
    --fast-eval \
    --push-to-hub anirudhb11/oss_120b_single_turn

Resume from checkpoint (if previous run failed during evaluation):
    python collect_trajectories_single_turn.py \
        --resume-from checkpoints/20260117_143052 \
        --dataset bicycleman15/intellect_3_code_very_hard \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        ... (same args as original run)

Checkpoints are saved after generation (before evaluation) to:
    checkpoints/<YYYYMMDD_HHMMSS>/checkpoint.pkl
    checkpoints/<YYYYMMDD_HHMMSS>/checkpoint_info.json
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, date
from typing import Any, Optional

from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Harmony utilities for GPT-OSS models
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
from intellect_env import IntellectCodeEnv
from utils.fast_eval import EvalTask, evaluate_tasks
from utils.pass_at_k import compute_pass_at_k
from utils.vllm_multi_gpu import (
    resolve_vllm_gpu_ids,
    build_vllm_server_urls,
    launch_vllm_servers,
    wait_for_vllm_servers,
    register_vllm_shutdown,
)

# Backend imports (conditional)
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
        SamplingResult,
    )
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


def render_trajectory(messages: list[dict], question: str, reward: float, terminated: bool) -> str:
    """Render a trajectory as a formatted string."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"Question: {question[:50]}..." if len(question) > 50 else f"Question: {question}")
    lines.append(f"Reward: {reward:.2f} | Terminated: {terminated}")
    lines.append("=" * 80)
    
    for msg in messages:
        role = msg['role'].upper()
        content = msg['content']
        lines.append(f"\n[{role}]\n{content}")
    
    return "\n".join(lines)

# prompt_v1 -- original prompt
SYSTEM_PROMPT = """You are a helpful coding assistant.
Solve the given programming problem and provide your solution.

First, think about the problem step by step.
Then, provide your final solution wrapped in ```python``` code blocks.
"""

# prompt_v2
# SYSTEM_PROMPT = """You are a helpful coding assistant.

# IMPORTANT CONTEXT:
# - This is a single-turn conversation.

# ────────────────────────
# HARD RULES (NON-NEGOTIABLE)
# ────────────────────────

# - You must first reason about the problem step-by-step, and only then output the final answer. You are FORBIDDEN from outputting any ```python``` code block (even partial solutions) without any reasoning.
# - The final code should execute without any exceptions.
# - Use your reasoning to confirm, revise, or reject a stated hypothesis.

# ────────────────────────
# MANDATORY GUIDELINES
# ────────────────────────

# While formulating hypothesis, you MUST clearly state:
# - The specific assumption, or uncertainty being tested
# - What do you expect if the hypothesis is correct vs incorrect

# After testing the hypothesis using reasoning, you MUST then clearly state:
# - What the reasoning resulted in (summarize or quote key lines)
# - Whether the hypothesis was confirmed, weakened, or falsified
# - What (if anything) changed in your approach

# ────────────────────────
# SOLUTION STRESS TEST (CRITICAL)
# ────────────────────────
# - For algorithmic correctness problems, you could compare whether your implementation gives the same output compared to a bruteforce correct reference implementation
# - You can build brute force / exhaustive checking for small inputs (e.g., n ≤ 6–8) and check against those
# - If a counterexample is found, you MUST revise your approach and repeat the above tests.

# Testing only the examples provided in the prompt does NOT count as validation or falsification.

# ────────────────────────
# ITERATIVE WORKFLOW
# ────────────────────────
# 1. State your approach and any assumptions or uncertainties.
# 2. Use reasoning to address those uncertainties.
# 4. Repeat steps 1–2 if meaningful uncertainty remains.
# 5. ONLY when no critical uncertainty remains, produce the final solution.

# ────────────────────────
# FINAL CODE REQUIREMENTS
# ────────────────────────
# - The final code MUST be inside a ```python``` code block.
# - The final code MUST read inputs from stdin and MUST NOT hardcode inputs.
# - The final answer MUST clearly be supported by your reasoning evidence.
# """

# prompt_v3
# SYSTEM_PROMPT = """You are an expert competitive programming assistant.

# ----------------------------
# PROBLEM-SOLVING APPROACH
# ----------------------------
# 1. UNDERSTAND: Carefully read and restate the problem in your own words.
# 2. ANALYZE: Identify key constraints, edge cases, and the core algorithmic challenge.
# 3. VERIFY: Mentally trace through the provided examples step-by-step.
# 4. IMPLEMENT: Write clean, correct, and efficient code.

# ----------------------------
# REASONING REQUIREMENTS
# ----------------------------
# Before writing any code, you MUST:
# - Identify the input/output format precisely
# - State the time and space complexity constraints
# - Walk through at least one example by hand to verify your understanding

# ----------------------------
# CODE REQUIREMENTS
# ----------------------------
# - The solution MUST be inside a ```python``` code block
# """

# prompt_v4
# SYSTEM_PROMPT = """You are an expert competitive programming assistant.

# ----------------------------
# PROBLEM-SOLVING APPROACH
# ----------------------------
# 1. UNDERSTAND: Carefully read and restate the problem in your own words.
# 2. ANALYZE: Identify key constraints, edge cases, and the core algorithmic challenge.
# 3. DESIGN: Choose an appropriate algorithm/data structure and justify your choice.
# 4. VERIFY: Mentally trace through the provided examples step-by-step.
# 5. IMPLEMENT: Write clean, correct, and efficient code.

# ----------------------------
# REASONING REQUIREMENTS
# ----------------------------
# Before writing any code, you MUST:
# - Identify the input/output format precisely
# - State the time and space complexity constraints
# - Consider edge cases (empty input, single element, maximum values, etc.)
# - Walk through at least one example by hand to verify your understanding

# ----------------------------
# CODE REQUIREMENTS
# ----------------------------
# - The solution MUST be inside a ```python``` code block
# - The code MUST handle all edge cases mentioned in the problem
# - Use appropriate data structures for the problem's constraints

# ----------------------------
# COMMON PITFALLS TO AVOID
# ----------------------------
# - Off-by-one errors in loops and array indexing
# - Integer overflow (use appropriate types if needed)
# - Not handling edge cases (n=0, n=1, empty strings, etc.)
# - Inefficient algorithms that exceed time limits
# - Incorrect input parsing (watch for multiple test cases, line formats)
# - Forgetting to flush output when required
# """

# For Harmony GPT-OSS models, use SYSTEM_PROMPT as DEVELOPER_INSTRUCTIONS
DEVELOPER_INSTRUCTIONS = SYSTEM_PROMPT


# =============================================================================
# HARMONY FORMAT HELPERS (for GPT-OSS models)
# =============================================================================

def build_harmony_conversation(messages: list, encoding) -> Conversation:
    """
    Build a Harmony Conversation from messages list for single-turn.
    
    For single-turn:
    - messages[0] is the system content (becomes Harmony SystemContent)
    - messages[1] is the developer content (becomes Harmony DeveloperContent)
    - messages[2] is the user content (the problem)
    """
    harmony_messages = []
    
    for entry in messages:
        role = entry["role"]
        content = entry["content"]
        
        if role == "system":
            # This is the Harmony SystemContent
            harmony_messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.SYSTEM, content))
        elif role == "developer":
            # This is the Harmony DeveloperContent with our instructions
            harmony_messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.DEVELOPER, content))
        elif role == "user":
            harmony_messages.append(HarmonyMessage.from_role_and_content(HarmonyRole.USER, content))
        elif role == "assistant":
            channel = entry.get("channel", "final")
            harmony_msg = HarmonyMessage.from_role_and_content(HarmonyRole.ASSISTANT, content)
            harmony_msg = harmony_msg.with_channel(channel)
            harmony_messages.append(harmony_msg)
    
    return Conversation.from_messages(harmony_messages)


def build_prompt_harmony(messages: list, encoding) -> list[int]:
    """Build tokenized prompt from messages using Harmony encoding (GPT-OSS path)."""
    conversation = build_harmony_conversation(messages, encoding)
    return encoding.render_conversation_for_completion(conversation, HarmonyRole.ASSISTANT)


def create_sampling_client(args):
    """Create sampling client based on backend choice."""
    if args.backend == "tinker":
        if not TINKER_AVAILABLE:
            raise ImportError("tinker not installed. Install it or use --backend vllm")
        if args.vllm_multi_gpu:
            raise ValueError("--vllm-multi-gpu requires --backend vllm")
        service_client = tinker.ServiceClient()
        return service_client.create_sampling_client(base_model=args.model)
    else:  # vllm
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
            # Support tensor parallelism for large models like GPT-OSS-120B
            tensor_parallel_size = getattr(args, 'tensor_parallel_size', 1)
            if tensor_parallel_size > 1:
                # Import LLM directly to pass tensor_parallel_size
                from vllm import LLM
                llm = LLM(
                    model=args.model,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    tensor_parallel_size=tensor_parallel_size,
                )
                return SamplingClient(args.model, llm=llm)
            return SamplingClient(args.model, gpu_memory_utilization=args.gpu_memory_utilization)


def create_sampling_params(args, backend: str, harmony_encoding=None):
    """Create sampling params for the chosen backend.
    
    Args:
        args: Command line arguments
        backend: 'tinker' or 'vllm'
        harmony_encoding: If using Harmony (GPT-OSS), pass the encoding to get stop tokens
    """
    stop_token_ids = None
    
    # For Harmony models, add Harmony stop tokens
    if harmony_encoding is not None:
        # Use stop_tokens_for_assistant_actions() which returns only <|return|> and <|call|>
        # Do NOT use stop_tokens() which includes <|end|> - that marks the end of ONE message,
        # but the model outputs multiple messages (analysis channel -> final channel)
        stop_token_ids = harmony_encoding.stop_tokens_for_assistant_actions()
    
    if backend == "tinker":
        params = tinker_types.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=0.95,
        )
        if stop_token_ids:
            params.stop_token_ids = stop_token_ids
        return params
    else:
        return SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=0.95,
            stop_token_ids=stop_token_ids,
        )


def build_prompt(messages: list[dict], tokenizer) -> list[int]:
    """Build tokenized prompt from messages."""
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(prompt_text)["input_ids"]


async def sample_batch_tinker(client, prompts: list[list[int]], sampling_params, tokenizer) -> list[SamplingResult]:
    """Batch sample using tinker (via async gather). Returns list of SamplingResult."""
    async def sample_one(input_ids):
        return await client.sample_async(
            prompt=tinker_types.ModelInput.from_ints(input_ids),
            sampling_params=sampling_params,
            num_samples=1,
        )

    return list(await asyncio.gather(*[sample_one(p) for p in prompts]))


def sample_batch_vllm(client, prompts: list[list[int]], sampling_params, show_progress: bool = False) -> list[SamplingResult]:
    """Batch sample using vLLM. Returns list of SamplingResult."""
    model_inputs = [ModelInput.from_ints(p) for p in prompts]
    # Pass show_progress for MultiServerSamplingClient (ignored by other clients)
    try:
        results = client.sample_batch(model_inputs, sampling_params, num_samples=1, show_progress=show_progress)
    except TypeError:
        # Fallback for clients that don't support show_progress
        results = client.sample_batch(model_inputs, sampling_params, num_samples=1)
    return results


async def sample_batch_tinker_harmony(client, prompts: list[list[int]], sampling_params, encoding) -> list[tuple[str, str, Optional[str]]]:
    """Batch sample using tinker with Harmony parsing.
    
    Returns list of (response_content, channel, analysis_content) tuples.
    """
    token_results = await sample_batch_tinker(client, prompts, sampling_params, None)
    parsed_results = []
    for result in token_results:
        tokens = result.sequences[0].tokens
        content, channel, analysis = parse_harmony_response(tokens, encoding)
        parsed_results.append((content, channel, analysis))
    return parsed_results


def sample_batch_vllm_harmony(client, prompts: list[list[int]], sampling_params, encoding, show_progress: bool = False) -> list[tuple[str, str, Optional[str]]]:
    """Batch sample using vLLM with Harmony parsing.
    
    Returns list of (response_content, channel, analysis_content) tuples.
    """
    model_inputs = [ModelInput.from_ints(p) for p in prompts]
    try:
        results = client.sample_batch(model_inputs, sampling_params, num_samples=1, show_progress=show_progress)
    except TypeError:
        results = client.sample_batch(model_inputs, sampling_params, num_samples=1)
    
    parsed_results = []
    for result in results:
        # Get tokens from the result for Harmony parsing
        tokens = result.sequences[0].tokens
        content, channel, analysis = parse_harmony_response(tokens, encoding)
        parsed_results.append((content, channel, analysis))
    return parsed_results


def _truncated_by_token_limit(finish_reason: str | None) -> bool:
    """True if generation stopped due to max token limit."""
    return (finish_reason or "").lower() in ("length", "length_capped")


@dataclass
class SingleTurnState:
    """State for a single-turn trajectory."""
    problem_index: int
    sample_index: int
    question: str
    tests: list
    max_tests: int
    messages: list[dict]  # For prompt building (may contain Harmony objects)
    messages_for_record: list[dict] | None = None  # For JSON serialization (simplified for Harmony)
    response: str = ""
    finish_reason: str | None = None


def serialize_single_turn_state(state: SingleTurnState) -> dict:
    """Serialize a SingleTurnState to a dictionary for checkpointing.
    
    Note: For Harmony states, we serialize messages_for_record since the
    actual messages contain non-serializable Harmony objects.
    """
    # Use messages_for_record for serialization if available (Harmony case)
    serializable_messages = state.messages_for_record or state.messages
    return {
        "problem_index": state.problem_index,
        "sample_index": state.sample_index,
        "question": state.question,
        "tests": state.tests,
        "max_tests": state.max_tests,
        "messages": [msg.copy() for msg in serializable_messages],
        "messages_for_record": [msg.copy() for msg in serializable_messages],
        "response": state.response,
        "finish_reason": state.finish_reason,
    }


def deserialize_single_turn_state(data: dict) -> SingleTurnState:
    """Deserialize a dictionary back to a SingleTurnState.
    
    Note: For resumed Harmony states, both messages and messages_for_record
    will be the serializable version. This is fine since we've already
    generated the response before checkpointing.
    """
    return SingleTurnState(
        problem_index=data["problem_index"],
        sample_index=data["sample_index"],
        question=data["question"],
        tests=data["tests"],
        max_tests=data["max_tests"],
        messages=data["messages"],
        messages_for_record=data.get("messages_for_record"),
        response=data["response"],
        finish_reason=data.get("finish_reason"),
    )


def run_batched_rollouts(
    args,
    client,
    tokenizer_or_encoding,
    sampling_params,
    checkpoint_manager: Optional[CheckpointManager] = None,
    use_harmony: bool = False,
) -> list[list[dict[str, Any]]]:
    """Run batched single-turn rollouts across all problems and samples.
    
    Args:
        args: Command line arguments
        client: Sampling client (tinker or vLLM)
        tokenizer_or_encoding: HF tokenizer (standard) or HarmonyEncoding (GPT-OSS)
        sampling_params: Sampling parameters
        checkpoint_manager: Optional checkpoint manager for resuming
        use_harmony: If True, use Harmony format for GPT-OSS models
    """
    # Load dataset ONCE before creating environments
    print(f"Loading dataset {args.dataset}...")
    if args.dataset.startswith("bicycleman15/") or args.dataset.startswith("anirudhb11/intellect_3"):
        from datasets import load_dataset
        full_dataset = load_dataset(args.dataset, split="train")
    elif args.dataset.__contains__('lcb'):
        from datasets import load_dataset
        full_dataset = load_dataset(args.dataset, split="test")
    else:
        from datasets import load_dataset
        full_dataset = load_dataset(args.dataset, "code", split="train")
    
    # Select slice based on start_problem and num_problems
    end_problem = min(args.start_problem + args.num_problems, len(full_dataset))
    shared_dataset = full_dataset.select(range(args.start_problem, end_problem))
    actual_num_problems = len(shared_dataset)
    print(f"Dataset loaded with {len(full_dataset)} total problems.")
    print(f"Selected slice: problems {args.start_problem} to {end_problem - 1} ({actual_num_problems} problems)")
    
    # Update args.num_problems to reflect actual slice size
    if actual_num_problems < args.num_problems:
        print(f"Warning: Only {actual_num_problems} problems available from index {args.start_problem}")
        args.num_problems = actual_num_problems
    
    # For Harmony, build the system and developer content once
    harmony_system_content = None
    harmony_developer_content = None
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
    
    states: list[SingleTurnState] = []
    skip_generation = False
    
    # Resume from checkpoint if available
    if checkpoint_manager and checkpoint_manager.has_checkpoint():
        print(f"\nResuming from checkpoint: {checkpoint_manager.checkpoint_dir}")
        checkpoint_data = checkpoint_manager.load()
        
        # Verify args match
        for warning in checkpoint_manager.verify_args({
            "start_problem": args.start_problem,
            "num_problems": args.num_problems,
            "num_samples": args.num_samples,
            "dataset": args.dataset,
            "model": args.model,
        }):
            print(warning)
        
        # Restore states
        print(f"  Restoring {len(checkpoint_data.active_states_data)} states...")
        for state_data in checkpoint_data.active_states_data:
            state = deserialize_single_turn_state(state_data)
            states.append(state)
        
        print(f"  Successfully restored {len(states)} states with generated responses.")
        skip_generation = True
    else:
        # Initialize all environments and build prompts
        print(f"Initializing {args.num_problems * args.num_samples} rollouts...")
        for problem_idx in range(args.num_problems):
            for sample_idx in range(args.num_samples):
                env = IntellectCodeEnv(
                    system_prompt="",
                    dataset_name=args.dataset,
                    problem_index=problem_idx,
                    max_turns=1,
                    dataset=shared_dataset,
                    interaction_mode=False,
                )
                obs, info = env.reset()
                env.has_interacted = True  # Single-turn mode
                
                if use_harmony:
                    # Harmony format: system content + developer content (instructions) + user
                    # We store two versions:
                    # - messages: contains actual Harmony objects for prompt building
                    # - messages_for_record: simplified version for JSON serialization
                    messages = [
                        {"role": "system", "content": harmony_system_content},
                        {"role": "developer", "content": harmony_developer_content},
                        {"role": "user", "content": obs},
                    ]
                    # Store serializable version in state for output
                    messages_for_record = [
                        {"role": "system", "content": f"[Harmony format] Reasoning effort: {args.reasoning_effort}"},
                        {"role": "developer", "content": DEVELOPER_INSTRUCTIONS},
                        {"role": "user", "content": obs},
                    ]
                else:
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": obs}
                    ]
                    messages_for_record = messages
                
                state = SingleTurnState(
                    problem_index=problem_idx,
                    sample_index=sample_idx,
                    question=env.question,
                    tests=env.tests,
                    max_tests=env.max_tests,
                    messages=messages,
                    messages_for_record=messages_for_record,
                )
                states.append(state)
    
    # Generation phase (skip if resuming from checkpoint)
    if not skip_generation:
        # Build prompts for all states
        if use_harmony:
            prompts = [build_prompt_harmony(state.messages, tokenizer_or_encoding) for state in states]
        else:
            prompts = [build_prompt(state.messages, tokenizer_or_encoding) for state in states]
        
        # Batch sample all responses at once
        print(f"Sampling {len(prompts)} responses...")
        if use_harmony:
            # Harmony path: parse tokens with Harmony encoding
            if args.backend == "tinker":
                harmony_results = asyncio.run(sample_batch_tinker_harmony(
                    client, prompts, sampling_params, tokenizer_or_encoding
                ))
            else:
                harmony_results = sample_batch_vllm_harmony(
                    client, prompts, sampling_params, tokenizer_or_encoding, show_progress=args.vllm_multi_gpu
                )
            # Store response in states (harmony_results is list of (content, channel, analysis))
            for state, (content, channel, analysis) in zip(states, harmony_results):
                state.response = content
                state.finish_reason = None  # Harmony doesn't expose finish_reason
        else:
            if args.backend == "tinker":
                sampling_results = asyncio.run(sample_batch_tinker(client, prompts, sampling_params, tokenizer_or_encoding))
            else:
                sampling_results = sample_batch_vllm(client, prompts, sampling_params, show_progress=args.vllm_multi_gpu)
            # Store response and finish_reason in states
            for state, result in zip(states, sampling_results):
                seq = result.sequences[0]
                state.response = getattr(seq, "text", None) or tokenizer_or_encoding.decode(getattr(seq, "tokens", []), skip_special_tokens=True)
                state.finish_reason = getattr(seq, "finish_reason", None)
        
        # Save checkpoint BEFORE evaluation (so we can retry if evaluation fails)
        if checkpoint_manager:
            print(f"Saving checkpoint after generation...")
            checkpoint_manager.save(
                active_states_data=[serialize_single_turn_state(s) for s in states],
                completed_states_data=[],
                current_round=1,
                total_rounds=1,
            )
    else:
        print(f"\n[Resuming] Skipping generation, going directly to evaluation...")
    
    # Evaluation phase
    print(f"Evaluating {len(states)} responses...")
    all_trajectories: list[list[dict]] = [[] for _ in range(args.num_problems)]

    if args.fast_eval:
        eval_tasks = [
            EvalTask(
                response=state.response,
                tests=state.tests,
                max_tests=state.max_tests,
                timeout_s=args.eval_timeout_s,
                require_solution_class=True,
            )
            for state in states
        ]
        eval_results = evaluate_tasks(
            eval_tasks,
            max_workers=args.eval_workers,
            batch_size=args.eval_batch_size,
            show_progress=True,
        )

        for state, eval_result in zip(states, eval_results):
            # Use messages_for_record (serializable) if available, otherwise messages
            output_messages = (state.messages_for_record or state.messages).copy()
            output_messages.append({"role": "assistant", "content": state.response})

            traj = {
                "question": state.question,
                "messages": output_messages,
                "final_reward": eval_result.reward,
                "terminated": eval_result.terminated,
                "truncated": eval_result.truncated,
                "tests": state.tests,
                "finish_reason": state.finish_reason,
                "truncated_by_token_limit": _truncated_by_token_limit(state.finish_reason),
            }
            all_trajectories[state.problem_index].append(traj)
    else:
        # Sequential evaluation using IntellectCodeEnv
        for state in tqdm(states, desc="Evaluating"):
            env = IntellectCodeEnv(
                system_prompt="",
                dataset_name=args.dataset,
                problem_index=state.problem_index,
                max_turns=1,
                dataset=shared_dataset,
                interaction_mode=False,
            )
            env.reset()
            env.has_interacted = True  # Single-turn mode
            
            obs, reward, terminated, truncated, info = env.step(state.response)
            
            # Use messages_for_record (serializable) if available, otherwise messages
            output_messages = (state.messages_for_record or state.messages).copy()
            output_messages.append({"role": "assistant", "content": state.response})

            traj = {
                "question": state.question,
                "messages": output_messages,
                "final_reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "tests": state.tests,
                "finish_reason": state.finish_reason,
                "truncated_by_token_limit": _truncated_by_token_limit(state.finish_reason),
            }
            all_trajectories[state.problem_index].append(traj)

    return all_trajectories


def main(args):
    print(f"=" * 60)
    print(f"Collecting single-turn trajectories")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model}")
    print(f"  Backend: {args.backend}")
    if is_gpt_oss_model(args.model):
        print(f"  Format: Harmony (GPT-OSS)")
        print(f"  Reasoning effort: {args.reasoning_effort}")
    print(f"  Problem range: {args.start_problem} to {args.start_problem + args.num_problems - 1} ({args.num_problems} problems)")
    print(f"  Samples per problem: {args.num_samples}")
    print(f"  Output: {args.output_dir}")
    if args.resume_from:
        print(f"  Resuming from: {args.resume_from}")
    print(f"=" * 60)
    
    # Setup checkpoint manager
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
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
    )
    
    # Detect if we're using a GPT-OSS model that requires Harmony format
    use_harmony = is_gpt_oss_model(args.model)
    
    if use_harmony:
        print(f"\nDetected GPT-OSS model: {args.model}")
        print(f"Using Harmony format with reasoning effort: {args.reasoning_effort}")
        
        # Load Harmony encoding instead of HuggingFace tokenizer
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        tokenizer_or_encoding = encoding
        
        # Create sampling params with Harmony stop tokens
        sampling_params = create_sampling_params(args, args.backend, harmony_encoding=encoding)
    else:
        # Standard path: load HuggingFace tokenizer
        tokenizer_or_encoding = AutoTokenizer.from_pretrained(args.model)
        sampling_params = create_sampling_params(args, args.backend)
    
    # Initialize client based on backend
    print(f"\nInitializing {args.backend} client...")
    sampling_client = create_sampling_client(args)
    
    print(f"\nCollecting trajectories for {args.num_problems} problems (batched)...")
    
    # Run batched rollouts
    all_trajectories = run_batched_rollouts(
        args=args,
        client=sampling_client,
        tokenizer_or_encoding=tokenizer_or_encoding,
        sampling_params=sampling_params,
        checkpoint_manager=checkpoint_manager,
        use_harmony=use_harmony,
    )
    
    # Flatten into rows (one per trajectory)
    rows = []
    all_results = []
    
    for problem_idx, problem_trajectories in enumerate(all_trajectories):
        problem_results = []
        
        for traj_idx, traj in enumerate(problem_trajectories):
            is_successful = traj["final_reward"] > 0
            problem_results.append(is_successful)
            
            rows.append({
                "problem_id": problem_idx,
                "trajectory_id": traj_idx,
                "question": traj["question"],
                "messages": json.dumps(traj["messages"]),
                "final_reward": traj["final_reward"],
                "terminated": traj["terminated"],
                "truncated": traj["truncated"],
                "tests": json.dumps(traj["tests"]),
                "finish_reason": traj.get("finish_reason"),
                "truncated_by_token_limit": traj.get("truncated_by_token_limit", False),
                "is_successful": is_successful,
                "rendered": render_trajectory(
                    traj["messages"], traj["question"],
                    traj["final_reward"], traj["terminated"]
                ),
            })
        
        all_results.append(problem_results)
    
    pass_at_1 = compute_pass_at_k(all_results, k=1)
    pass_at_2 = compute_pass_at_k(all_results, k=2)
    pass_at_4 = compute_pass_at_k(all_results, k=4)
    pass_at_8 = compute_pass_at_k(all_results, k=min(8, args.num_samples))
    
    print(f"\n{'=' * 60}")
    print(f"Results:")
    print(f"  Total trajectories: {len(rows)}")
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
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "timestamp": datetime.now().isoformat(),
        "pass_at_1": pass_at_1,
        "pass_at_2": pass_at_2,
        "pass_at_4": pass_at_4,
        "pass_at_8": pass_at_8,
        "mode": "single_turn",
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
    parser = argparse.ArgumentParser(description="Collect single-turn trajectories for code problems")
    parser.add_argument("--dataset", type=str, default="bicycleman15/intellect_3_code_easy_medium",
                        choices=["bicycleman15/intellect_3_code_easy_medium", "bicycleman15/intellect_3_code_hard",
                                 "bicycleman15/intellect_3_code_very_hard", "anirudhb11/intellect_3_code_very_hard_top_400_hardest",
                                 "PrimeIntellect/INTELLECT-3-RL", "anirudhb11/lcb_v6_feb_may_2025_formatted"])
    parser.add_argument("--start-problem", type=int, default=0,
                        help="Starting problem index for dataset slicing (default: 0)")
    parser.add_argument("--num-problems", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output-dir", type=str, default="artifacts/trajectories_single_turn")
    parser.add_argument("--push-to-hub", type=str, default=None, help="HF repo to push to (e.g. username/repo-name)")
    
    # Checkpointing options
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
    parser.add_argument("--fast-eval", action="store_true",
                        help="Use parallel fast eval for final answers")
    parser.add_argument("--eval-workers", type=int, default=16,
                        help="Number of parallel evaluator workers (default: min(32, cpu_count))")
    parser.add_argument("--eval-batch-size", type=int, default=8,
                        help="Number of responses per evaluator task (default: 8)")
    parser.add_argument("--eval-timeout-s", type=float, default=5.0,
                        help="Per-test timeout in seconds for fast evaluation (default: 5.0)")
    
    # Harmony/GPT-OSS options
    parser.add_argument("--reasoning-effort", type=str, default="medium",
                        choices=["low", "medium", "high"],
                        help="Reasoning effort for GPT-OSS models using Harmony format (default: medium)")
    
    # Tensor parallelism for large models
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism (for large models like GPT-OSS-120B). Default: 1")
    
    args = parser.parse_args()
    main(args)
