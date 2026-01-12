"""Collect multi-turn trajectories and save as HuggingFace dataset.

Usage:
    python collect_trajectories.py --dataset bicycleman15/intellect_3_code_easy_medium --num-problems 20
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
    
    # If stopped by </interact>, append it back so regex can match
    if "<interact>" in response and "</interact>" not in response:
        response += "</interact>"
    
    return response


async def do_single_rollout(
    problem_index: int,
    dataset_name: str,
    tokenizer: AutoTokenizer,
    client: tinker.SamplingClient,
    sampling_params: types.SamplingParams,
    max_turns: int = 5,
) -> dict[str, Any]:
    """Run a single multi-turn rollout for a problem."""
    env = IntellectCodeEnv(
        system_prompt="",  # We add system prompt to history instead
        dataset_name=dataset_name,
        problem_index=problem_index,
        max_turns=max_turns,
    )
    
    obs, info = env.reset()
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": obs}]
    total_reward = 0.0
    interactions = []
    
    for step in range(max_turns):
        action = await get_llm_action(obs, history, tokenizer, client, sampling_params)
        
        history.append({"role": "user", "content": obs})
        history.append({"role": "assistant", "content": action})
        messages.append({"role": "assistant", "content": action})
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if env.history and len(env.history) > len(interactions):
            interactions = env.history.copy()
        
        if obs:
            messages.append({"role": "user", "content": obs})
        
        if terminated or truncated:
            break
    
    return {
        "question": env.question,
        "messages": messages,
        "num_turns": env.current_turn,
        "final_reward": total_reward,
        "terminated": terminated,
        "truncated": truncated,
        "interactions": interactions,
        "tests": env.tests,
    }


async def collect_trajectories_for_problem(
    problem_index: int,
    dataset_name: str,
    tokenizer: AutoTokenizer,
    client: tinker.SamplingClient,
    sampling_params: types.SamplingParams,
    num_samples: int = 8,
    max_turns: int = 5,
) -> list[dict[str, Any]]:
    """Collect multiple trajectories for a single problem using parallel rollouts."""
    trajectories = await asyncio.gather(*[
        do_single_rollout(
            problem_index=problem_index,
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            client=client,
            sampling_params=sampling_params,
            max_turns=max_turns,
        )
        for _ in range(num_samples)
    ])
    
    return list(trajectories)


async def gather_with_progress(coroutines: list, desc: str) -> list:
    """Run coroutines concurrently with a progress bar."""
    pbar = tqdm(total=len(coroutines), desc=desc)
    
    async def track(coro):
        result = await coro
        pbar.update(1)
        return result
    
    try:
        results = await asyncio.gather(*[track(coro) for coro in coroutines])
    finally:
        pbar.close()
    
    return results


async def main(args):
    print(f"=" * 60)
    print(f"Collecting trajectories")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model}")
    print(f"  Problems: {args.num_problems}")
    print(f"  Samples per problem: {args.num_samples}")
    print(f"  Max turns: {args.max_turns}")
    print(f"  Output: {args.output_dir}")
    print(f"=" * 60)
    
    # Initialize tinker client (from eval.py and gem_math_demo.py)
    print("\nInitializing tinker client...")
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    sampling_params = types.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=0.95,
        stop=["</interact>"],
    )
    
    print(f"\nCollecting trajectories for {args.num_problems} problems...")
    
    all_trajectories = await gather_with_progress(
        [
            collect_trajectories_for_problem(
                problem_index=i,
                dataset_name=args.dataset,
                tokenizer=tokenizer,
                client=sampling_client,
                sampling_params=sampling_params,
                num_samples=args.num_samples,
                max_turns=args.max_turns,
            )
            for i in range(args.num_problems)
        ],
        desc="Problems",
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
    parser.add_argument("--output-dir", type=str, default="trajectories")
    parser.add_argument("--push-to-hub", type=str, default=None, help="HF repo to push to (e.g. username/repo-name)")
    
    args = parser.parse_args()
    asyncio.run(main(args))
