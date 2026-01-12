"""Collect single-turn trajectories and save as HuggingFace dataset.

Usage:
    python collect_trajectories_single_turn.py \
    --dataset bicycleman15/intellect_3_code_easy_medium \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --num-problems 20 \
    --num-samples 8 \
    --push-to-hub bicycleman15/qwen3_4b_instruct_easy_medium_single_turn
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import asyncio
import json
from datetime import datetime
from typing import Any

import tinker
from tinker import types
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from intellect_env import IntellectCodeEnv
from utils.pass_at_k import compute_pass_at_k


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


async def get_llm_action(
    obs: str,
    history: list[dict],
    tokenizer: AutoTokenizer,
    client: tinker.SamplingClient,
    sampling_params: types.SamplingParams,
) -> str:
    """Get LLM action given observation and history."""
    messages = history + [{"role": "user", "content": obs}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    input_ids = tokenizer(prompt_text)["input_ids"]
    result = await client.sample_async(
        prompt=types.ModelInput.from_ints(input_ids),
        sampling_params=sampling_params,
        num_samples=1,
    )
    response = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
    
    return response


async def do_single_rollout(
    problem_index: int,
    dataset_name: str,
    tokenizer: AutoTokenizer,
    client: tinker.SamplingClient,
    sampling_params: types.SamplingParams,
) -> dict[str, Any]:
    """Run a single-turn rollout for a problem."""
    env = IntellectCodeEnv(
        system_prompt="",  # We add system prompt to history instead
        dataset_name=dataset_name,
        problem_index=problem_index,
        max_turns=1,  # Single turn only
    )
    
    obs, info = env.reset()
    # Single-turn mode: allow immediate final answer without requiring <interact>
    env.has_interacted = True
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": obs}]
    
    # Single turn: get response and evaluate
    action = await get_llm_action(obs, history, tokenizer, client, sampling_params)
    messages.append({"role": "assistant", "content": action})
    
    # Step to get reward (will evaluate the code)
    _, reward, terminated, truncated, info = env.step(action)
    
    return {
        "question": env.question,
        "messages": messages,
        "final_reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "tests": env.tests,
    }


async def collect_trajectories_for_problem(
    problem_index: int,
    dataset_name: str,
    tokenizer: AutoTokenizer,
    client: tinker.SamplingClient,
    sampling_params: types.SamplingParams,
    num_samples: int = 8,
) -> list[dict[str, Any]]:
    """Collect multiple trajectories for a single problem using parallel rollouts."""
    trajectories = await asyncio.gather(*[
        do_single_rollout(
            problem_index=problem_index,
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            client=client,
            sampling_params=sampling_params,
        )
        for _ in range(num_samples)
    ])
    
    return list(trajectories)


async def main(args):
    print(f"=" * 60)
    print(f"Collecting single-turn trajectories")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model}")
    print(f"  Problems: {args.num_problems}")
    print(f"  Samples per problem: {args.num_samples}")
    print(f"  Output: {args.output_dir}")
    print(f"=" * 60)
    
    # Initialize tinker client
    print("\nInitializing tinker client...")
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    sampling_params = types.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=0.95,
    )
    
    print(f"\nCollecting trajectories for {args.num_problems} problems...")
    
    # Process problems sequentially (parallel rollouts per problem)
    all_trajectories = []
    for i in tqdm(range(args.num_problems), desc="Problems"):
        problem_trajectories = await collect_trajectories_for_problem(
            problem_index=i,
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            client=sampling_client,
            sampling_params=sampling_params,
            num_samples=args.num_samples,
        )
        all_trajectories.append(problem_trajectories)
    
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
    parser.add_argument("--num-problems", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output-dir", type=str, default="artifacts/trajectories_single_turn")
    parser.add_argument("--push-to-hub", type=str, default=None, help="HF repo to push to (e.g. username/repo-name)")
    
    args = parser.parse_args()
    asyncio.run(main(args))
