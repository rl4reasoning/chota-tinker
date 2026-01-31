"""Collect single-turn trajectories and save as HuggingFace dataset.

Usage:

    python collect_trajectories_single_turn.py \
    --dataset bicycleman15/intellect_3_code_very_hard \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --backend vllm \
    --start-problem 0 \
    --num-problems 25 \
    --num-samples 320 \
    \
    --fast-eval \
    --eval-workers 8 \
    --eval-batch-size 8 \
    --eval-timeout-s 1.0 \
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
from datetime import datetime
from typing import Any, Optional

from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

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


SYSTEM_PROMPT = """You are a helpful coding assistant.
Solve the given programming problem and provide your solution.

First, think about the problem step by step.
Then, provide your final solution wrapped in ```python``` code blocks.
"""


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
            return SamplingClient(args.model, gpu_memory_utilization=args.gpu_memory_utilization)


def create_sampling_params(args, backend: str):
    """Create sampling params for the chosen backend."""
    if backend == "tinker":
        return tinker_types.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=0.95,
        )
    else:
        return SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=0.95,
        )


def build_prompt(messages: list[dict], tokenizer) -> list[int]:
    """Build tokenized prompt from messages."""
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(prompt_text)["input_ids"]


async def sample_batch_tinker(client, prompts: list[list[int]], sampling_params, tokenizer) -> list[str]:
    """Batch sample using tinker (via async gather)."""
    async def sample_one(input_ids):
        result = await client.sample_async(
            prompt=tinker_types.ModelInput.from_ints(input_ids),
            sampling_params=sampling_params,
            num_samples=1,
        )
        return tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
    
    results = await asyncio.gather(*[sample_one(p) for p in prompts])
    return results


def sample_batch_vllm(client, prompts: list[list[int]], sampling_params, show_progress: bool = False) -> list[str]:
    """Batch sample using vLLM."""
    model_inputs = [ModelInput.from_ints(p) for p in prompts]
    # Pass show_progress for MultiServerSamplingClient (ignored by other clients)
    try:
        results = client.sample_batch(model_inputs, sampling_params, num_samples=1, show_progress=show_progress)
    except TypeError:
        # Fallback for clients that don't support show_progress
        results = client.sample_batch(model_inputs, sampling_params, num_samples=1)
    return [r.sequences[0].text for r in results]


@dataclass
class SingleTurnState:
    """State for a single-turn trajectory."""
    problem_index: int
    sample_index: int
    question: str
    tests: list
    max_tests: int
    messages: list[dict]
    response: str = ""


def serialize_single_turn_state(state: SingleTurnState) -> dict:
    """Serialize a SingleTurnState to a dictionary for checkpointing."""
    return {
        "problem_index": state.problem_index,
        "sample_index": state.sample_index,
        "question": state.question,
        "tests": state.tests,
        "max_tests": state.max_tests,
        "messages": [msg.copy() for msg in state.messages],
        "response": state.response,
    }


def deserialize_single_turn_state(data: dict) -> SingleTurnState:
    """Deserialize a dictionary back to a SingleTurnState."""
    return SingleTurnState(
        problem_index=data["problem_index"],
        sample_index=data["sample_index"],
        question=data["question"],
        tests=data["tests"],
        max_tests=data["max_tests"],
        messages=data["messages"],
        response=data["response"],
    )


def run_batched_rollouts(
    args,
    client,
    tokenizer,
    sampling_params,
    checkpoint_manager: Optional[CheckpointManager] = None,
) -> list[list[dict[str, Any]]]:
    """Run batched single-turn rollouts across all problems and samples."""
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
        print(f"Warning: Only {actual_num_problems} problems available from index {args.start_problem}")
        args.num_problems = actual_num_problems
    
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
                
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": obs}
                ]
                
                state = SingleTurnState(
                    problem_index=problem_idx,
                    sample_index=sample_idx,
                    question=env.question,
                    tests=env.tests,
                    max_tests=env.max_tests,
                    messages=messages,
                )
                states.append(state)
    
    # Generation phase (skip if resuming from checkpoint)
    if not skip_generation:
        prompts = [build_prompt(state.messages, tokenizer) for state in states]
        
        # Batch sample all responses at once
        print(f"Sampling {len(prompts)} responses...")
        if args.backend == "tinker":
            responses = asyncio.run(sample_batch_tinker(client, prompts, sampling_params, tokenizer))
        else:
            responses = sample_batch_vllm(client, prompts, sampling_params, show_progress=args.vllm_multi_gpu)
        
        # Store responses in states
        for state, response in zip(states, responses):
            state.response = response
        
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
            messages = state.messages.copy()
            messages.append({"role": "assistant", "content": state.response})

            traj = {
                "question": state.question,
                "messages": messages,
                "final_reward": eval_result.reward,
                "terminated": eval_result.terminated,
                "truncated": eval_result.truncated,
                "tests": state.tests,
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
            
            messages = state.messages.copy()
            messages.append({"role": "assistant", "content": state.response})

            traj = {
                "question": state.question,
                "messages": messages,
                "final_reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "tests": state.tests,
            }
            all_trajectories[state.problem_index].append(traj)
    
    return all_trajectories


def main(args):
    print(f"=" * 60)
    print(f"Collecting single-turn trajectories")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model}")
    print(f"  Backend: {args.backend}")
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
    
    # Initialize client based on backend
    print(f"\nInitializing {args.backend} client...")
    sampling_client = create_sampling_client(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sampling_params = create_sampling_params(args, args.backend)
    
    print(f"\nCollecting trajectories for {args.num_problems} problems (batched)...")
    
    # Run batched rollouts
    all_trajectories = run_batched_rollouts(
        args=args,
        client=sampling_client,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
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
                                 "bicycleman15/intellect_3_code_very_hard", "PrimeIntellect/INTELLECT-3-RL"])
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
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                        help="GPU memory utilization for local vLLM or vLLM servers (default: 0.8)")
    parser.add_argument("--fast-eval", action="store_true",
                        help="Use parallel fast eval for final answers")
    parser.add_argument("--eval-workers", type=int, default=16,
                        help="Number of parallel evaluator workers (default: min(32, cpu_count))")
    parser.add_argument("--eval-batch-size", type=int, default=8,
                        help="Number of responses per evaluator task (default: 8)")
    parser.add_argument("--eval-timeout-s", type=float, default=1.0,
                        help="Per-test timeout in seconds for fast evaluation (default: 1.0)")
    
    args = parser.parse_args()
    main(args)
