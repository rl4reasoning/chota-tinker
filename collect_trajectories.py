"""Collect multi-turn trajectories and save as HuggingFace dataset.

Usage:
    python collect_trajectories.py \
    --dataset bicycleman15/intellect_3_code_very_hard \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --backend vllm \
    --num-problems 30 \
    --num-samples 32 \
    --push-to-hub bicycleman15/qwen3_4b_instruct_easy_medium

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

from intellect_env import IntellectCodeEnv, step_batch
from utils.pass_at_k import compute_pass_at_k

# Backend imports (conditional)
try:
    import tinker
    from tinker import types as tinker_types
    TINKER_AVAILABLE = True
except ImportError:
    TINKER_AVAILABLE = False

try:
    from chota_tinker import SamplingClient, ServerSamplingClient, SamplingParams, ModelInput
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


def render_trajectory(messages: list[dict], interactions: list[dict], question: str, reward: float, num_turns: int, terminated: bool, truncated: bool) -> str:
    """Render a trajectory as a formatted string."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"Question: {question[:50]}..." if len(question) > 50 else f"Question: {question}")
    lines.append(f"Reward: {reward:.2f} | Turns: {num_turns} | Terminated: {terminated} | Truncated: {truncated}")
    lines.append("=" * 80)
    
    for msg in messages:
        role = msg['role'].upper()
        content = msg['content']
        lines.append(f"\n[{role}]\n{content}")
    
    if interactions:
        lines.append(f"\n{'─' * 80}")
        lines.append(f"Code Interactions ({len(interactions)}):")
        for i, inter in enumerate(interactions):
            lines.append(f"  [{i+1}] Code:\n{inter['code']}")
            lines.append(f"  Output:\n{inter['output']}")
    
    return "\n".join(lines)


SYSTEM_PROMPT = """You are a helpful coding assistant.
You are allowed to interact with the Python interpreter.
You can wrap your code in <interact></interact>, and I will run it for you and give you the output.
Make sure that you define the inputs (or hardcode inputs) yourself when you give me <interact></interact> block.
You can use the output to refine your code.

Once you are done, wrap the final code in ```python``` code blocks. 
When returning the final code, there is no need to hardcode inputs, you will take inputs from stdin.

Please first think about the problem before you output <interact></interact> or ```python``` code blocks.

NOTE: You must interact atleast once successfully before you submit the final code!
"""


@dataclass
class RolloutState:
    """Track state of a single rollout for batched processing."""
    problem_index: int
    sample_index: int
    env: IntellectCodeEnv
    history: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    interactions: list[dict] = field(default_factory=list)
    total_reward: float = 0.0
    obs: str = ""
    done: bool = False
    terminated: bool = False
    truncated: bool = False


def create_sampling_client(args):
    """Create sampling client based on backend choice."""
    if args.backend == "tinker":
        if not TINKER_AVAILABLE:
            raise ImportError("tinker not installed. Install it or use --backend vllm")
        service_client = tinker.ServiceClient()
        return service_client.create_sampling_client(base_model=args.model)
    else:  # vllm
        if not VLLM_AVAILABLE:
            raise ImportError("chota_tinker not installed. Install it or use --backend tinker")
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
            stop=["</interact>"],
        )
    else:
        return SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=0.95,
            stop=["</interact>"],
        )


def build_prompt(state: RolloutState, tokenizer) -> list[int]:
    """Build tokenized prompt from rollout state."""
    messages = state.history + [{"role": "user", "content": state.obs}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(prompt_text)["input_ids"]


def postprocess_response(response: str) -> str:
    """Postprocess LLM response."""
    if "<interact>" in response and "</interact>" not in response:
        response += "</interact>"
    return response


async def sample_batch_tinker(client, prompts: list[list[int]], sampling_params) -> list[str]:
    """Batch sample using tinker (via async gather)."""
    async def sample_one(input_ids):
        result = await client.sample_async(
            prompt=tinker_types.ModelInput.from_ints(input_ids),
            sampling_params=sampling_params,
            num_samples=1,
        )
        return result.sequences[0].tokens
    
    results = await asyncio.gather(*[sample_one(p) for p in prompts])
    return results


def sample_batch_vllm(client, prompts: list[list[int]], sampling_params) -> list[str]:
    """Batch sample using vLLM."""
    model_inputs = [ModelInput.from_ints(p) for p in prompts]
    results = client.sample_batch(model_inputs, sampling_params, num_samples=1)
    return [r.sequences[0].text for r in results]


def run_batched_rollouts(
    args,
    client,
    tokenizer,
    sampling_params,
) -> list[list[dict[str, Any]]]:
    """Run batched rollouts across all problems and samples."""
    # Load dataset ONCE before creating environments
    print(f"Loading dataset {args.dataset}...")
    if args.dataset.startswith("bicycleman15/"):
        from datasets import load_dataset
        shared_dataset = load_dataset(args.dataset, split="train")
    else:
        from datasets import load_dataset
        shared_dataset = load_dataset(args.dataset, "code", split="train")
    print(f"Dataset loaded with {len(shared_dataset)} problems.")
    
    # Initialize all rollout states
    active_states: list[RolloutState] = []
    for problem_idx in range(args.num_problems):
        for sample_idx in range(args.num_samples):
            env = IntellectCodeEnv(
                system_prompt="",
                dataset_name=args.dataset,
                problem_index=problem_idx,
                max_turns=args.max_turns,
                dataset=shared_dataset,  # Pass pre-loaded dataset
            )
            obs, info = env.reset()
            state = RolloutState(
                problem_index=problem_idx,
                sample_index=sample_idx,
                env=env,
                history=[{"role": "system", "content": SYSTEM_PROMPT}],
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": obs}],
                obs=obs,
            )
            active_states.append(state)
    
    completed_states: list[RolloutState] = []
    problems_completed = set()
    
    pbar = tqdm(total=args.num_problems, desc="Problems completed")
    
    while active_states:
        # Build prompts for all active states
        prompts = [build_prompt(s, tokenizer) for s in active_states]
        
        # Batch sample
        if args.backend == "tinker":
            token_results = asyncio.run(sample_batch_tinker(client, prompts, sampling_params))
            responses = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in token_results]
        else:
            responses = sample_batch_vllm(client, prompts, sampling_params)
        
        # Process responses and step environments
        still_active = []
        processed_responses = []
        for state, response in zip(active_states, responses):
            response = postprocess_response(response)
            
            # Update history and messages
            state.history.append({"role": "user", "content": state.obs})
            state.history.append({"role": "assistant", "content": response})
            state.messages.append({"role": "assistant", "content": response})
            processed_responses.append(response)

        if args.fast_eval:
            step_results = step_batch(
                [s.env for s in active_states],
                processed_responses,
                eval_workers=args.eval_workers,
                eval_batch_size=args.eval_batch_size,
                eval_timeout_s=args.eval_timeout_s,
                show_progress=False,
            )
        else:
            step_results = [s.env.step(r) for s, r in zip(active_states, processed_responses)]

        for state, _response, (obs, reward, terminated, truncated, info) in zip(
            active_states, processed_responses, step_results
        ):
            
            # Step environment
            state.total_reward += reward
            state.terminated = terminated
            state.truncated = truncated
            
            # Update interactions
            if state.env.history and len(state.env.history) > len(state.interactions):
                state.interactions = state.env.history.copy()
            
            if obs:
                state.messages.append({"role": "user", "content": obs})
            
            if terminated or truncated:
                state.done = True
                completed_states.append(state)
                
                # Check if all samples for this problem are done
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
    
    # Organize results by problem
    all_trajectories: list[list[dict]] = [[] for _ in range(args.num_problems)]
    for state in completed_states:
        traj = {
            "question": state.env.question,
            "messages": state.messages,
            "num_turns": state.env.current_turn,
            "final_reward": state.total_reward,
            "terminated": state.terminated,
            "truncated": state.truncated,
            "interactions": state.interactions,
            "tests": state.env.tests,
        }
        all_trajectories[state.problem_index].append(traj)
    
    return all_trajectories

def main(args):
    print(f"=" * 60)
    print(f"Collecting trajectories")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model}")
    print(f"  Backend: {args.backend}")
    print(f"  Problems: {args.num_problems}")
    print(f"  Samples per problem: {args.num_samples}")
    print(f"  Max turns: {args.max_turns}")
    print(f"  Output: {args.output_dir}")
    print(f"=" * 60)
    
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
                "num_turns": traj["num_turns"],
                "final_reward": traj["final_reward"],
                "terminated": traj["terminated"],
                "truncated": traj["truncated"],
                "interactions": json.dumps(traj["interactions"]),
                "tests": json.dumps(traj["tests"]),
                "is_successful": is_successful,
                "rendered": render_trajectory(
                    traj["messages"], traj["interactions"], traj["question"],
                    traj["final_reward"], traj["num_turns"], traj["terminated"], traj["truncated"]
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
        "num_problems": args.num_problems,
        "num_samples": args.num_samples,
        "max_turns": args.max_turns,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "timestamp": datetime.now().isoformat(),
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
    parser = argparse.ArgumentParser(description="Collect multi-turn trajectories for code problems")
    parser.add_argument("--dataset", type=str, default="bicycleman15/intellect_3_code_easy_medium",
                        choices=["bicycleman15/intellect_3_code_easy_medium", "bicycleman15/intellect_3_code_hard",
                                 "bicycleman15/intellect_3_code_very_hard", "PrimeIntellect/INTELLECT-3-RL"])
    parser.add_argument("--num-problems", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output-dir", type=str, default="artifacts/trajectories")
    parser.add_argument("--push-to-hub", type=str, default=None, help="HF repo to push to (e.g. username/repo-name)")
    # Backend options
    parser.add_argument("--backend", type=str, default="vllm", choices=["tinker", "vllm"],
                        help="Inference backend: 'tinker' or 'vllm' (default: vllm)")
    parser.add_argument("--vllm-server-url", type=str, default=None,
                        help="URL for vLLM server (e.g. http://localhost:8000). If not set, uses local vLLM.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="GPU memory utilization for local vLLM (default: 0.9)")
    parser.add_argument("--fast-eval", action="store_true",
                        help="Use parallel fast eval for final answers")
    parser.add_argument("--eval-workers", type=int, default=max(1, min(32, os.cpu_count() or 1)),
                        help="Number of parallel evaluator workers (default: min(32, cpu_count))")
    parser.add_argument("--eval-batch-size", type=int, default=8,
                        help="Number of responses per evaluator task (default: 8)")
    parser.add_argument("--eval-timeout-s", type=float, default=5.0,
                        help="Per-test timeout in seconds for fast evaluation (default: 5.0)")
    
    args = parser.parse_args()
    main(args)
