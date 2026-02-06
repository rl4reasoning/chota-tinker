"""Evaluate multi-turn checkpoints.

Usage:
    python eval_checkpoint_multi_turn.py checkpoints/20260205_134734_bba79239

    # With custom eval settings
    python eval_checkpoint_multi_turn.py checkpoints/20260205_134734_bba79239 \
        --eval-workers 16 \
        --eval-batch-size 8 \
        --eval-timeout-s 1.0

    # Push results to HuggingFace Hub
    python eval_checkpoint_multi_turn.py checkpoints/20260205_134734_bba79239 \
        --push-to-hub username/repo-name
"""

import argparse
import json
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from datasets import Dataset
from tqdm import tqdm

from utils.fast_eval import EvalTask, evaluate_tasks
from utils.pass_at_k import compute_pass_at_k


@dataclass
class RolloutState:
    """Track state of a single rollout for batched processing."""
    problem_index: int
    sample_index: int
    history: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    interactions: list[dict] = field(default_factory=list)
    total_reward: float = 0.0
    obs: str = ""
    done: bool = False
    terminated: bool = False
    truncated: bool = False
    interaction_timeout_count: int = 0
    eval_timeout_count: int = 0
    # Env-related data
    question: str = ""
    tests: dict = field(default_factory=dict)
    max_tests: int = 100
    current_turn: int = 0
    has_interacted: bool = False


def deserialize_rollout_state(data: dict) -> RolloutState:
    """Deserialize a dictionary back to a RolloutState."""
    return RolloutState(
        problem_index=data["problem_index"],
        sample_index=data["sample_index"],
        history=data.get("history", []),
        messages=data.get("messages", []),
        interactions=data.get("interactions", []),
        total_reward=data.get("total_reward", 0.0),
        obs=data.get("obs", ""),
        done=data.get("done", False),
        terminated=data.get("terminated", False),
        truncated=data.get("truncated", False),
        interaction_timeout_count=data.get("interaction_timeout_count", 0),
        eval_timeout_count=data.get("eval_timeout_count", 0),
        question=data.get("question", ""),
        tests=data.get("tests", {}),
        max_tests=data.get("max_tests", 100),
        current_turn=data.get("current_turn", 0),
        has_interacted=data.get("has_interacted", False),
    )


def render_trajectory(messages: list[dict], interactions: list[dict], question: str, 
                      reward: float, num_turns: int, terminated: bool, truncated: bool) -> str:
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
            lines.append(f"  [{i+1}] Code:\n{inter.get('code', 'N/A')}")
            lines.append(f"  Output:\n{inter.get('output', 'N/A')}")
    
    return "\n".join(lines)


def load_checkpoint(checkpoint_dir: str) -> tuple[list[RolloutState], list[RolloutState], dict]:
    """Load checkpoint from directory.
    
    Returns:
        Tuple of (completed_states, active_states, args_dict)
    """
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pkl")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_dir}")
    
    with open(checkpoint_path, "rb") as f:
        checkpoint_data = pickle.load(f)
    
    # Load args from checkpoint
    args_dict = checkpoint_data.args_dict
    
    # Deserialize completed states
    completed_states = []
    for state_data in checkpoint_data.completed_states_data:
        state = deserialize_rollout_state(state_data)
        completed_states.append(state)
    
    # Deserialize active states
    active_states = []
    for state_data in checkpoint_data.active_states_data:
        state = deserialize_rollout_state(state_data)
        active_states.append(state)
    
    print(f"  Loaded {len(completed_states)} completed states")
    print(f"  Loaded {len(active_states)} active (in-progress) states")
    print(f"  Args: {json.dumps(args_dict, indent=2)}")
    
    return completed_states, active_states, args_dict


def get_last_assistant_response(state: RolloutState) -> Optional[str]:
    """Extract the last assistant response from a state's history."""
    for msg in reversed(state.history):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return None


def evaluate_active_states(
    states: list[RolloutState],
    eval_workers: int = 16,
    eval_batch_size: int = 8,
    eval_timeout_s: float = 1.0,
    require_solution_class: bool = True,
) -> list[tuple[RolloutState, float]]:
    """Evaluate active states by treating their last response as final.
    
    Returns:
        List of (state, reward) tuples
    """
    if not states:
        return []
    
    print(f"\nEvaluating {len(states)} active states...")
    
    # Get last responses
    states_with_responses = []
    for state in states:
        response = get_last_assistant_response(state)
        if response:
            states_with_responses.append((state, response))
    
    if not states_with_responses:
        print("  No active states have assistant responses to evaluate")
        return []
    
    print(f"  Found {len(states_with_responses)} states with responses to evaluate")
    
    eval_tasks = [
        EvalTask(
            response=response,
            tests=state.tests,
            max_tests=state.max_tests,
            timeout_s=eval_timeout_s,
            require_solution_class=require_solution_class,
        )
        for state, response in states_with_responses
    ]
    
    eval_results = evaluate_tasks(
        eval_tasks,
        max_workers=eval_workers,
        batch_size=eval_batch_size,
        show_progress=True,
    )
    
    results = []
    for (state, _), eval_result in zip(states_with_responses, eval_results):
        results.append((state, eval_result.reward))
    
    return results


def main(args):
    print(f"=" * 60)
    print(f"Evaluating checkpoint: {args.checkpoint_dir}")
    print(f"=" * 60)
    
    # Load checkpoint
    completed_states, active_states, args_dict = load_checkpoint(args.checkpoint_dir)
    
    # Get num_samples and num_problems for pass@k calculation
    num_samples = args_dict.get("num_samples", 8)
    num_problems = args_dict.get("num_problems", 1)
    
    # Collect results from completed states (already have rewards)
    all_trajectories: list[list[dict]] = [[] for _ in range(num_problems)]
    
    for state in completed_states:
        traj = {
            "question": state.question,
            "messages": state.messages,
            "num_turns": state.current_turn,
            "final_reward": state.total_reward,
            "terminated": state.terminated,
            "truncated": state.truncated,
            "interactions": state.interactions,
            "tests": state.tests,
            "interaction_timeout_count": state.interaction_timeout_count,
            "eval_timeout_count": state.eval_timeout_count,
            "from_active": False,
        }
        all_trajectories[state.problem_index].append(traj)
    
    # Evaluate active states (their last response hasn't been evaluated yet)
    if active_states:
        active_results = evaluate_active_states(
            states=active_states,
            eval_workers=args.eval_workers,
            eval_batch_size=args.eval_batch_size,
            eval_timeout_s=args.eval_timeout_s,
            require_solution_class=not args.no_require_solution_class,
        )
        
        for state, reward in active_results:
            traj = {
                "question": state.question,
                "messages": state.messages,
                "num_turns": state.current_turn,
                "final_reward": reward,
                "terminated": False,
                "truncated": True,  # Mark as truncated since it didn't finish naturally
                "interactions": state.interactions,
                "tests": state.tests,
                "interaction_timeout_count": state.interaction_timeout_count,
                "eval_timeout_count": state.eval_timeout_count,
                "from_active": True,
            }
            all_trajectories[state.problem_index].append(traj)
    
    # Compute results
    all_results = []
    for problem_trajectories in all_trajectories:
        problem_results = [traj["final_reward"] > 0 for traj in problem_trajectories]
        all_results.append(problem_results)
    
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
    
    # Count statistics
    total_trajectories = sum(len(trajs) for trajs in all_trajectories)
    successful_trajectories = sum(
        1 for trajs in all_trajectories for traj in trajs if traj["final_reward"] > 0
    )
    problems_solved = sum(1 for results in all_results if any(results))
    from_active_count = sum(
        1 for trajs in all_trajectories for traj in trajs if traj.get("from_active", False)
    )
    
    print(f"\n{'=' * 60}")
    print(f"Results:")
    print(f"  Total trajectories: {total_trajectories}")
    print(f"    - From completed: {total_trajectories - from_active_count}")
    print(f"    - From active (evaluated): {from_active_count}")
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
                    "num_turns": traj["num_turns"],
                    "final_reward": traj["final_reward"],
                    "terminated": traj["terminated"],
                    "truncated": traj["truncated"],
                    "interactions": json.dumps(traj["interactions"]),
                    "tests": json.dumps(traj["tests"]),
                    "is_successful": is_successful,
                    "interaction_timeout_count": traj["interaction_timeout_count"],
                    "eval_timeout_count": traj["eval_timeout_count"],
                    "from_active": traj.get("from_active", False),
                    "rendered": render_trajectory(
                        traj["messages"], traj["interactions"], traj["question"],
                        traj["final_reward"], traj["num_turns"], traj["terminated"], traj["truncated"]
                    ),
                })
        
        dataset = Dataset.from_list(rows)
        
        print(f"\nPushing to HuggingFace Hub: {args.push_to_hub}")
        dataset.push_to_hub(args.push_to_hub, private=False)
        print(f"Successfully pushed to: https://huggingface.co/datasets/{args.push_to_hub}")
    
    return all_trajectories, all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multi-turn checkpoints")
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
