"""Evaluate single-turn checkpoints.

Usage:
    python eval_checkpoint_single_turn.py checkpoints/20260205_134734_bba79239

    # With custom eval settings
    python eval_checkpoint_single_turn.py checkpoints/20260205_134734_bba79239 \
        --eval-workers 16 \
        --eval-batch-size 8 \
        --eval-timeout-s 5.0 \
        --push-to-hub bicycleman15/prompt_v3_single_turn_0_50

    # Push results to HuggingFace Hub
    python eval_checkpoint_single_turn.py checkpoints/20260205_134734_bba79239 \
        --push-to-hub username/repo-name
"""

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from datasets import Dataset
from tqdm import tqdm

from utils.fast_eval import EvalTask, evaluate_tasks
from utils.pass_at_k import compute_pass_at_k


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


def load_checkpoint(checkpoint_dir: str) -> tuple[list[SingleTurnState], dict]:
    """Load checkpoint from directory.
    
    Returns:
        Tuple of (states, args_dict)
    """
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pkl")
    info_path = os.path.join(checkpoint_dir, "checkpoint_info.json")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_dir}")
    
    with open(checkpoint_path, "rb") as f:
        checkpoint_data = pickle.load(f)
    
    # Load args from checkpoint
    args_dict = checkpoint_data.args_dict
    
    # Deserialize states
    states = []
    for state_data in checkpoint_data.active_states_data:
        state = deserialize_single_turn_state(state_data)
        states.append(state)
    
    print(f"  Loaded {len(states)} states")
    print(f"  Args: {json.dumps(args_dict, indent=2)}")
    
    return states, args_dict


def evaluate_checkpoint(
    states: list[SingleTurnState],
    args_dict: dict,
    eval_workers: int = 16,
    eval_batch_size: int = 8,
    eval_timeout_s: float = 1.0,
    require_solution_class: bool = True,
) -> tuple[list[list[dict[str, Any]]], list[list[bool]]]:
    """Evaluate all states and return trajectories and results.
    
    Returns:
        Tuple of (all_trajectories, all_results)
    """
    num_problems = args_dict.get("num_problems", max(s.problem_index for s in states) + 1)
    
    print(f"\nEvaluating {len(states)} responses...")
    print(f"  Workers: {eval_workers}")
    print(f"  Batch size: {eval_batch_size}")
    print(f"  Timeout: {eval_timeout_s}s")
    
    all_trajectories: list[list[dict]] = [[] for _ in range(num_problems)]
    
    eval_tasks = [
        EvalTask(
            response=state.response,
            tests=state.tests,
            max_tests=state.max_tests,
            timeout_s=eval_timeout_s,
            require_solution_class=require_solution_class,
        )
        for state in states
    ]
    eval_results = evaluate_tasks(
        eval_tasks,
        max_workers=eval_workers,
        batch_size=eval_batch_size,
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
    
    # Compute results
    all_results = []
    for problem_trajectories in all_trajectories:
        problem_results = [traj["final_reward"] > 0 for traj in problem_trajectories]
        all_results.append(problem_results)
    
    return all_trajectories, all_results


def main(args):
    print(f"=" * 60)
    print(f"Evaluating checkpoint: {args.checkpoint_dir}")
    print(f"=" * 60)
    
    # Load checkpoint
    states, args_dict = load_checkpoint(args.checkpoint_dir)
    
    # Get num_samples for pass@k calculation
    num_samples = args_dict.get("num_samples", 8)
    
    # Evaluate
    all_trajectories, all_results = evaluate_checkpoint(
        states=states,
        args_dict=args_dict,
        eval_workers=args.eval_workers,
        eval_batch_size=args.eval_batch_size,
        eval_timeout_s=args.eval_timeout_s,
        require_solution_class=not args.no_require_solution_class,
    )
    
    # Compute pass@k metrics
    pass_at_1 = compute_pass_at_k(all_results, k=1)
    pass_at_2 = compute_pass_at_k(all_results, k=2)
    pass_at_4 = compute_pass_at_k(all_results, k=4)
    pass_at_8 = compute_pass_at_k(all_results, k=min(8, num_samples))
    pass_at_16 = compute_pass_at_k(all_results, k=min(16, num_samples))
    pass_at_32 = compute_pass_at_k(all_results, k=min(32, num_samples))
    pass_at_64 = compute_pass_at_k(all_results, k=min(64, num_samples))
    pass_at_128 = compute_pass_at_k(all_results, k=min(128, num_samples))
    pass_at_256 = compute_pass_at_k(all_results, k=min(256, num_samples))
    
    # Count total successful trajectories
    total_trajectories = sum(len(trajs) for trajs in all_trajectories)
    successful_trajectories = sum(
        1 for trajs in all_trajectories for traj in trajs if traj["final_reward"] > 0
    )
    problems_solved = sum(1 for results in all_results if any(results))
    
    print(f"\n{'=' * 60}")
    print(f"Results:")
    print(f"  Total trajectories: {total_trajectories}")
    print(f"  Successful trajectories: {successful_trajectories}")
    print(f"  Problems solved: {problems_solved}/{len(all_results)}")
    print(f"  pass@1: {pass_at_1:.4f}")
    print(f"  pass@2: {pass_at_2:.4f}")
    print(f"  pass@4: {pass_at_4:.4f}")
    print(f"  pass@8: {pass_at_8:.4f}")
    if num_samples >= 16:
        print(f"  pass@16: {pass_at_16:.4f}")
    if num_samples >= 32:
        print(f"  pass@32: {pass_at_32:.4f}")
    if num_samples >= 64:
        print(f"  pass@64: {pass_at_64:.4f}")
    if num_samples >= 128:
        print(f"  pass@128: {pass_at_128:.4f}")
    if num_samples >= 256:
        print(f"  pass@256: {pass_at_256:.4f}")
    print(f"{'=' * 60}")
    
    # Push to HuggingFace Hub if requested
    if args.push_to_hub:
        # Create dataset rows
        rows = []
        for problem_idx, problem_trajectories in enumerate(all_trajectories):
            for traj_idx, traj in enumerate(problem_trajectories):
                is_successful = traj["final_reward"] > 0
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
        
        dataset = Dataset.from_list(rows)
        
        print(f"\nPushing to HuggingFace Hub: {args.push_to_hub}")
        dataset.push_to_hub(args.push_to_hub, private=False)
        print(f"Successfully pushed to: https://huggingface.co/datasets/{args.push_to_hub}")
    
    return all_trajectories, all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate single-turn checkpoints")
    parser.add_argument("checkpoint_dir", type=str,
                        help="Path to checkpoint directory (e.g. checkpoints/20260205_134734_bba79239)")
    
    # Evaluation options
    parser.add_argument("--eval-workers", type=int, default=16,
                        help="Number of parallel evaluator workers (default: 16)")
    parser.add_argument("--eval-batch-size", type=int, default=8,
                        help="Number of responses per evaluator task (default: 8)")
    parser.add_argument("--eval-timeout-s", type=float, default=1.0,
                        help="Per-test timeout in seconds for evaluation (default: 1.0)")
    parser.add_argument("--no-require-solution-class", action="store_true",
                        help="Don't require Solution class for problems with fn_name")
    
    # Output options
    parser.add_argument("--push-to-hub", type=str, default=None,
                        help="HF repo to push results to (e.g. username/repo-name)")
    
    args = parser.parse_args()
    main(args)
