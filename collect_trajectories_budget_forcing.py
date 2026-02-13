"""Collect trajectories with s1 budget forcing and save as HuggingFace dataset.

Implements the "budget forcing" technique from:
    s1: Simple test-time scaling (https://arxiv.org/abs/2501.19393)

When the model tries to end generation (hits EOS), we forcefully append "Wait"
to make it continue reasoning. This often leads the model to double-check
and fix incorrect reasoning steps.

Usage:
    python collect_trajectories_budget_forcing.py \
        --dataset bicycleman15/intellect_3_code_very_hard \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --backend vllm \
        --start-problem 0 \
        --num-problems 25 \
        --num-samples 35 \
        --num-attempts 10 \
        \
        --fast-eval \
        --eval-workers 8 \
        --eval-batch-size 8 \
        --eval-timeout-s 1.0 \
        --push-to-hub bicycleman15/temp2

Multi-GPU (launches one vLLM server per GPU, shards prompts across them):
    python collect_trajectories_budget_forcing.py \
        --dataset bicycleman15/intellect_3_code_very_hard \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --backend vllm \
        --vllm-multi-gpu \
        --vllm-gpu-ids 0,1,2,3 \
        --num-problems 2 \
        --num-samples 32 \
        --num-attempts 5 \
        \
        --fast-eval \
        --eval-workers 16 \
        --eval-batch-size 8 \
        --eval-timeout-s 1.0 \
        --push-to-hub bicycleman15/qwen3_4b_very_hard_s1_x5

Resume from checkpoint (if previous run failed):
    python collect_trajectories_budget_forcing.py \
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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from checkpoint import CheckpointManager, get_checkpoint_dir
from intellect_env import IntellectCodeEnv, step_batch
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
    )
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


# The magic word that extends thinking (from s1 paper)
WAIT_TOKEN = "Wait"


def render_trajectory(messages: list[dict], question: str, reward: float, 
                      terminated: bool, num_extensions: int) -> str:
    """Render a trajectory as a formatted string."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"Question: {question[:50]}..." if len(question) > 50 else f"Question: {question}")
    lines.append(f"Reward: {reward:.2f} | Terminated: {terminated} | Extensions: {num_extensions}")
    lines.append("=" * 80)
    
    for msg in messages:
        role = msg['role'].upper()
        content = msg['content']
        lines.append(f"\n[{role}]\n{content}")
    
    return "\n".join(lines)


# SYSTEM_PROMPT = """You are a helpful coding assistant.
# Solve the given programming problem and provide your solution.

# First, think about the problem step by step.
# Then, provide your final solution wrapped in ```python``` code blocks.
# """

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

SYSTEM_PROMPT = """You are a helpful coding assistant.

IMPORTANT CONTEXT:
- This is a single-turn conversation.

────────────────────────
HARD RULES (NON-NEGOTIABLE)
────────────────────────

- You must first reason about the problem step-by-step, and only then output the final answer. You are FORBIDDEN from outputting any ```python``` code block (even partial solutions) without any reasoning.
- The final code should execute without any exceptions.
- Use your reasoning to confirm, revise, or reject a stated hypothesis.

────────────────────────
MANDATORY GUIDELINES
────────────────────────

While formulating hypothesis, you MUST clearly state:
- The specific assumption, or uncertainty being tested
- What do you expect if the hypothesis is correct vs incorrect

After testing the hypothesis using reasoning, you MUST then clearly state:
- What the reasoning resulted in (summarize or quote key lines)
- Whether the hypothesis was confirmed, weakened, or falsified
- What (if anything) changed in your approach

────────────────────────
SOLUTION STRESS TEST (CRITICAL)
────────────────────────
- For algorithmic correctness problems, you could compare whether your implementation gives the same output compared to a bruteforce correct reference implementation
- You can build brute force / exhaustive checking for small inputs (e.g., n ≤ 6–8) and check against those
- If a counterexample is found, you MUST revise your approach and repeat the above tests.

Testing only the examples provided in the prompt does NOT count as validation or falsification.

────────────────────────
ITERATIVE WORKFLOW
────────────────────────
1. State your approach and any assumptions or uncertainties.
2. Use reasoning to address those uncertainties.
4. Repeat steps 1–2 if meaningful uncertainty remains.
5. ONLY when no critical uncertainty remains, produce the final solution.

────────────────────────
FINAL CODE REQUIREMENTS
────────────────────────
- The final code MUST be inside a ```python``` code block.
- The final code MUST read inputs from stdin and MUST NOT hardcode inputs.
- The final answer MUST clearly be supported by your reasoning evidence.
"""


@dataclass
class BudgetForcingState:
    """Track state of a single rollout with budget forcing."""
    problem_index: int
    sample_index: int
    env: IntellectCodeEnv
    prompt_text: str  # Current accumulated prompt
    full_response: str = ""  # Accumulated response
    total_tokens: int = 0
    current_extension: int = 0  # Which extension we're on (0 = first gen)
    done: bool = False
    # Final results
    messages: list[dict] = field(default_factory=list)
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False


def serialize_budget_forcing_state(state: BudgetForcingState) -> dict:
    """Serialize a BudgetForcingState to a dictionary for checkpointing."""
    return {
        "problem_index": state.problem_index,
        "sample_index": state.sample_index,
        "prompt_text": state.prompt_text,
        "full_response": state.full_response,
        "total_tokens": state.total_tokens,
        "current_extension": state.current_extension,
        "done": state.done,
        "messages": state.messages.copy(),
        "reward": state.reward,
        "terminated": state.terminated,
        "truncated": state.truncated,
        # Env-related data needed for reconstruction
        "question": state.env.question,
        "tests": state.env.tests,
    }


def deserialize_budget_forcing_state(data: dict, shared_dataset, args) -> BudgetForcingState:
    """Deserialize a dictionary back to a BudgetForcingState."""
    env = IntellectCodeEnv(
        system_prompt="",
        dataset_name=args.dataset,
        problem_index=data["problem_index"],
        max_turns=1,
        dataset=shared_dataset,
        interaction_mode=False,
    )
    env.reset()
    env.has_interacted = True
    
    state = BudgetForcingState(
        problem_index=data["problem_index"],
        sample_index=data["sample_index"],
        env=env,
        prompt_text=data["prompt_text"],
        full_response=data["full_response"],
        total_tokens=data["total_tokens"],
        current_extension=data["current_extension"],
        done=data["done"],
        messages=data["messages"],
        reward=data["reward"],
        terminated=data["terminated"],
        truncated=data["truncated"],
    )
    return state


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
            return SamplingClient(
                args.model,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
            )


def create_sampling_params(args, backend: str, max_tokens: int):
    """Create sampling params for the chosen backend."""
    if backend == "tinker":
        return tinker_types.SamplingParams(
            max_tokens=max_tokens,
            temperature=args.temperature,
            top_p=0.95,
        )
    else:
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=args.temperature,
            top_p=0.95,
        )


async def sample_batch_tinker(client, prompts: list[list[int]], sampling_params, tokenizer) -> list[tuple[str, int]]:
    """Batch sample using tinker (via async gather). Returns (text, num_tokens) pairs."""
    async def sample_one(input_ids):
        result = await client.sample_async(
            prompt=tinker_types.ModelInput.from_ints(input_ids),
            sampling_params=sampling_params,
            num_samples=1,
        )
        tokens = result.sequences[0].tokens
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        return (text, len(tokens))
    
    results = await asyncio.gather(*[sample_one(p) for p in prompts])
    return results


def sample_batch_vllm(client, prompts: list[list[int]], sampling_params, show_progress: bool = False) -> list[tuple[str, int]]:
    """Batch sample using vLLM. Returns (text, num_tokens) pairs."""
    model_inputs = [ModelInput.from_ints(p) for p in prompts]
    # Pass show_progress for MultiServerSamplingClient (ignored by other clients)
    try:
        results = client.sample_batch(model_inputs, sampling_params, num_samples=1, show_progress=show_progress)
    except TypeError:
        # Fallback for clients that don't support show_progress
        results = client.sample_batch(model_inputs, sampling_params, num_samples=1)
    return [(r.sequences[0].text, len(r.sequences[0].tokens)) for r in results]


def run_batched_rollouts_with_budget_forcing(
    args,
    client,
    tokenizer,
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> list[list[dict[str, Any]]]:
    """Run batched single-turn rollouts with budget forcing."""
    # Load dataset ONCE before creating environments
    print(f"Loading dataset {args.dataset}...")
    if args.dataset.startswith("bicycleman15/"):
        from datasets import load_dataset
        full_dataset = load_dataset(args.dataset, split="train")
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
        print(f"Warning: Requested {args.num_problems} problems but only {actual_num_problems} available in slice.")
        args.num_problems = actual_num_problems
    
    num_attempts = args.num_attempts
    start_round = 0
    active_states: list[BudgetForcingState] = []
    completed_states: list[BudgetForcingState] = []
    
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
        start_round = checkpoint_data.current_round
        print(f"  Resuming from round {start_round + 1}/{num_attempts}")
        print(f"  Restoring {len(checkpoint_data.active_states_data)} active states...")
        print(f"  Restoring {len(checkpoint_data.completed_states_data)} completed states...")
        
        for state_data in checkpoint_data.active_states_data:
            state = deserialize_budget_forcing_state(state_data, shared_dataset, args)
            active_states.append(state)
        
        for state_data in checkpoint_data.completed_states_data:
            state = deserialize_budget_forcing_state(state_data, shared_dataset, args)
            completed_states.append(state)
        
        print(f"  Successfully restored {len(active_states)} active and {len(completed_states)} completed states.")
    else:
        # Initialize all states from scratch
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
                
                # Build initial prompt
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": obs}
                ]
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                state = BudgetForcingState(
                    problem_index=problem_idx,
                    sample_index=sample_idx,
                    env=env,
                    prompt_text=prompt_text,
                    messages=messages.copy(),
                )
                active_states.append(state)
    
    max_tokens = args.max_tokens
    
    # Process in rounds: each round handles one generation step for all active states
    for extension_round in range(start_round, num_attempts):
        if not active_states:
            break
            
        is_final_round = (extension_round == num_attempts - 1)
        print(f"\n[Round {extension_round + 1}/{num_attempts}] Generating for {len(active_states)} active states...")
        
        # Build prompts for all active states
        prompts = [tokenizer(state.prompt_text)["input_ids"] for state in active_states]
        
        # Each extension gets max_tokens (same semantics as collect_trajectories.py)
        sampling_params = create_sampling_params(args, args.backend, max_tokens)
        
        # Batch sample
        if args.backend == "tinker":
            results = asyncio.run(sample_batch_tinker(client, prompts, sampling_params, tokenizer))
        else:
            results = sample_batch_vllm(client, prompts, sampling_params, show_progress=args.vllm_multi_gpu)
        
        # Process results
        still_active = []
        for state, (generated_text, num_tokens) in zip(active_states, results):
            state.total_tokens += num_tokens
            state.current_extension = extension_round
            
            if is_final_round:
                # Final generation - append and finish
                state.full_response += generated_text
                state.done = True
                completed_states.append(state)
            else:
                # Force continuation by appending "Wait"
                state.full_response += generated_text + " " + WAIT_TOKEN
                state.prompt_text += generated_text + " " + WAIT_TOKEN
                still_active.append(state)
        
        active_states = still_active
        print(f"  Generated. {len(completed_states)} completed, {len(active_states)} continuing.")
        
        # Save checkpoint after each round
        if checkpoint_manager:
            checkpoint_manager.save(
                active_states_data=[serialize_budget_forcing_state(s) for s in active_states],
                completed_states_data=[serialize_budget_forcing_state(s) for s in completed_states],
                current_round=extension_round + 1,
                total_rounds=num_attempts,
            )
    
    # Handle any remaining active states (shouldn't happen but just in case)
    for state in active_states:
        state.done = True
        completed_states.append(state)
    
    # Evaluate all completed states
    print(f"\nEvaluating {len(completed_states)} trajectories...")
    if args.fast_eval:
        results = step_batch(
            [state.env for state in completed_states],
            [state.full_response for state in completed_states],
            eval_workers=args.eval_workers,
            eval_batch_size=args.eval_batch_size,
            eval_timeout_s=args.eval_timeout_s,
            show_progress=True,
        )
        for state, (_obs, reward, terminated, truncated, info) in zip(completed_states, results):
            # Add assistant response to messages
            state.messages.append({"role": "assistant", "content": state.full_response})
            state.reward = reward
            state.terminated = terminated
            state.truncated = truncated
    else:
        for state in tqdm(completed_states, desc="Evaluating"):
            # Add assistant response to messages
            state.messages.append({"role": "assistant", "content": state.full_response})
            
            # Step environment to get reward
            _, reward, terminated, truncated, info = state.env.step(state.full_response)
            state.reward = reward
            state.terminated = terminated
            state.truncated = truncated
    
    # Organize results by problem
    all_trajectories: list[list[dict]] = [[] for _ in range(args.num_problems)]
    for state in completed_states:
        traj = {
            "question": state.env.question,
            "messages": state.messages,
            "final_reward": state.reward,
            "terminated": state.terminated,
            "truncated": state.truncated,
            "tests": state.env.tests,
            "num_extensions": state.current_extension,  # How many "Wait" tokens were appended
            "total_tokens": state.total_tokens,
        }
        all_trajectories[state.problem_index].append(traj)
    
    return all_trajectories


def main(args):
    print(f"=" * 60)
    print(f"Collecting trajectories with s1 Budget Forcing")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model}")
    print(f"  Backend: {args.backend}")
    print(f"  Problem range: {args.start_problem} to {args.start_problem + args.num_problems - 1} ({args.num_problems} problems)")
    print(f"  Samples per problem: {args.num_samples}")
    print(f"  Num attempts: {args.num_attempts}")
    print(f"  Max tokens: {args.max_tokens}")
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
            "num_attempts": args.num_attempts,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
    )
    
    # Initialize client based on backend
    print(f"\nInitializing {args.backend} client...")
    sampling_client = create_sampling_client(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    print(f"\nCollecting trajectories for {args.num_problems} problems with budget forcing...")
    
    # Run batched rollouts with budget forcing
    all_trajectories = run_batched_rollouts_with_budget_forcing(
        args=args,
        client=sampling_client,
        tokenizer=tokenizer,
        checkpoint_manager=checkpoint_manager,
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
                "is_successful": is_successful,
                "num_extensions": traj["num_extensions"],
                "total_tokens": traj["total_tokens"],
                "rendered": render_trajectory(
                    traj["messages"], traj["question"],
                    traj["final_reward"], traj["terminated"],
                    traj["num_extensions"]
                ),
            })
        
        all_results.append(problem_results)
    
    pass_at_1 = compute_pass_at_k(all_results, k=1)
    pass_at_2 = compute_pass_at_k(all_results, k=2)
    pass_at_4 = compute_pass_at_k(all_results, k=4)
    pass_at_8 = compute_pass_at_k(all_results, k=min(8, args.num_samples))
    
    # Compute average tokens used
    avg_tokens = sum(r["total_tokens"] for r in rows) / len(rows) if rows else 0
    
    print(f"\n{'=' * 60}")
    print(f"Results:")
    print(f"  Total trajectories: {len(rows)}")
    print(f"  Avg tokens per trajectory: {avg_tokens:.1f}")
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
        "num_attempts": args.num_attempts,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "timestamp": datetime.now().isoformat(),
        "pass_at_1": pass_at_1,
        "pass_at_2": pass_at_2,
        "pass_at_4": pass_at_4,
        "pass_at_8": pass_at_8,
        "avg_tokens": avg_tokens,
        "mode": "budget_forcing",
        "method": "s1",
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
    parser = argparse.ArgumentParser(description="Collect trajectories with s1 budget forcing")
    parser.add_argument("--dataset", type=str, default="bicycleman15/intellect_3_code_easy_medium",
                        choices=["bicycleman15/intellect_3_code_easy_medium", "bicycleman15/intellect_3_code_hard",
                                 "bicycleman15/intellect_3_code_very_hard", "PrimeIntellect/INTELLECT-3-RL", "anirudhb11/lcb_v6_formatted", "anirudhb11/lcb_v6_feb_may_2025_formatted"])
    parser.add_argument("--start-problem", type=int, default=0,
                        help="Starting problem index for dataset slicing (default: 0)")
    parser.add_argument("--num-problems", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output-dir", type=str, default="artifacts/trajectories_budget_forcing")
    parser.add_argument("--push-to-hub", type=str, default=None, help="HF repo to push to (e.g. username/repo-name)")
    
    # Budget forcing specific
    parser.add_argument("--num-attempts", type=int, default=5,
                        help="Total number of generation rounds (comparable to --max-turns in collect_trajectories.py)")
    
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
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Maximum model context length for vLLM (default: None, uses model default)")
    parser.add_argument("--fast-eval", action="store_true",
                        help="Use parallel fast eval for final answers")
    parser.add_argument("--eval-workers", type=int, default=max(1, min(32, os.cpu_count() or 1)),
                        help="Number of parallel evaluator workers (default: min(32, cpu_count))")
    parser.add_argument("--eval-batch-size", type=int, default=8,
                        help="Number of responses per evaluator task (default: 8)")
    parser.add_argument("--eval-timeout-s", type=float, default=1,
                        help="Per-test timeout in seconds for fast evaluation (default: 5.0)")
    
    args = parser.parse_args()
    main(args)
