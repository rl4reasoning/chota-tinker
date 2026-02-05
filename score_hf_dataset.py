"""Score responses from a HuggingFace dataset.

Usage:
    python score_hf_dataset.py --dataset bicycleman15/prompt_v2_single_turn_0_25

    # With more workers for faster evaluation:
    python score_hf_dataset.py \
        --dataset bicycleman15/prompt_v2_single_turn_0_75 \
        --eval-workers 8 \
        --eval-batch-size 8 \
        --push-to-hub username/prompt_v2_single_turn_0_75_rescored

    # To score only a subset:
    python score_hf_dataset.py \
        --dataset bicycleman15/prompt_v2_single_turn_0_25 \
        --max-samples 100

    # To push re-scored dataset to hub:
    python score_hf_dataset.py \
        --dataset bicycleman15/prompt_v2_single_turn_0_25 \
        --push-to-hub username/rescored_dataset
"""

import argparse
import json
from datasets import load_dataset, Dataset
from tqdm import tqdm

from utils.fast_eval import EvalTask, EvalResult, evaluate_tasks
from utils.pass_at_k import compute_pass_at_k


def extract_assistant_response(messages: list[dict]) -> str:
    """Extract the last assistant response from messages."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def score_dataset(
    dataset_name: str,
    max_samples: int | None = None,
    eval_workers: int = 8,
    eval_batch_size: int = 8,
    eval_timeout_s: float = 1.0,
    require_solution_class: bool = True,
    max_tests: int = 15,
) -> tuple[Dataset, dict]:
    """Score all responses in a HuggingFace dataset.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., bicycleman15/prompt_v2_single_turn_0_25)
        max_samples: Maximum number of samples to score (None for all)
        eval_workers: Number of parallel evaluation workers
        eval_batch_size: Batch size for evaluation
        eval_timeout_s: Timeout per test case in seconds
        require_solution_class: Whether to require Solution class for fn_name problems
        max_tests: Maximum number of test cases to run per problem
        
    Returns:
        Tuple of (scored_dataset, metrics_dict)
    """
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    
    if max_samples is not None and max_samples < len(dataset):
        print(f"Limiting to {max_samples} samples (out of {len(dataset)})")
        dataset = dataset.select(range(max_samples))
    
    print(f"Scoring {len(dataset)} samples...")
    
    # Build evaluation tasks
    eval_tasks = []
    for row in dataset:
        # Parse messages from JSON string
        if isinstance(row["messages"], str):
            messages = json.loads(row["messages"])
        else:
            messages = row["messages"]
        
        # Parse tests from JSON string
        if isinstance(row["tests"], str):
            tests = json.loads(row["tests"])
        else:
            tests = row["tests"]
        
        # Extract assistant response
        response = extract_assistant_response(messages)
        
        eval_tasks.append(EvalTask(
            response=response,
            tests=tests,
            max_tests=max_tests,
            timeout_s=eval_timeout_s,
            require_solution_class=require_solution_class,
        ))
    
    # Run evaluation
    print(f"Evaluating with {eval_workers} workers, batch_size={eval_batch_size}...")
    eval_results = evaluate_tasks(
        eval_tasks,
        max_workers=eval_workers,
        batch_size=eval_batch_size,
        show_progress=True,
    )
    
    # Build scored rows
    scored_rows = []
    problem_results = {}  # problem_id -> list of results
    
    original_rewards = []
    new_rewards = []
    
    for row, eval_result in zip(dataset, eval_results):
        original_reward = row.get("final_reward", 0.0)
        new_reward = eval_result.reward
        
        original_rewards.append(original_reward)
        new_rewards.append(new_reward)
        
        problem_id = row.get("problem_id", 0)
        if problem_id not in problem_results:
            problem_results[problem_id] = []
        problem_results[problem_id].append(new_reward > 0)
        
        scored_row = dict(row)
        # Rename original final_reward to old_final_reward
        scored_row["old_final_reward"] = original_reward
        # Replace final_reward with the new rescored value
        scored_row["final_reward"] = new_reward
        scored_row["rescored_terminated"] = eval_result.terminated
        scored_row["rescored_truncated"] = eval_result.truncated
        scored_row["rescored_is_successful"] = new_reward > 0
        scored_row["reward_diff"] = new_reward - original_reward
        scored_rows.append(scored_row)
    
    # Compute metrics
    all_results = [problem_results[pid] for pid in sorted(problem_results.keys())]
    
    num_samples_per_problem = len(all_results[0]) if all_results else 0
    
    metrics = {
        "total_samples": len(scored_rows),
        "num_problems": len(all_results),
        "samples_per_problem": num_samples_per_problem,
        "original_success_rate": sum(1 for r in original_rewards if r > 0) / len(original_rewards) if original_rewards else 0,
        "rescored_success_rate": sum(1 for r in new_rewards if r > 0) / len(new_rewards) if new_rewards else 0,
        "pass_at_1": compute_pass_at_k(all_results, k=1),
        "pass_at_2": compute_pass_at_k(all_results, k=2) if num_samples_per_problem >= 2 else None,
        "pass_at_4": compute_pass_at_k(all_results, k=4) if num_samples_per_problem >= 4 else None,
        "pass_at_8": compute_pass_at_k(all_results, k=8) if num_samples_per_problem >= 8 else None,
        "reward_match_rate": sum(1 for o, n in zip(original_rewards, new_rewards) if abs(o - n) < 0.01) / len(original_rewards) if original_rewards else 0,
    }
    
    scored_dataset = Dataset.from_list(scored_rows)
    return scored_dataset, metrics


def main():
    parser = argparse.ArgumentParser(description="Score responses from a HuggingFace dataset")
    parser.add_argument("--dataset", type=str, required=True,
                        help="HuggingFace dataset name (e.g., bicycleman15/prompt_v2_single_turn_0_25)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to score (default: all)")
    parser.add_argument("--eval-workers", type=int, default=8,
                        help="Number of parallel evaluation workers (default: 8)")
    parser.add_argument("--eval-batch-size", type=int, default=8,
                        help="Batch size for evaluation (default: 8)")
    parser.add_argument("--eval-timeout-s", type=float, default=1.0,
                        help="Timeout per test case in seconds (default: 1.0)")
    parser.add_argument("--max-tests", type=int, default=15,
                        help="Maximum number of test cases per problem (default: 15)")
    parser.add_argument("--no-require-solution-class", action="store_true",
                        help="Don't require Solution class for fn_name problems")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save scored dataset (optional)")
    parser.add_argument("--push-to-hub", type=str, default=None,
                        help="HuggingFace repo to push scored dataset (e.g., username/repo)")
    
    args = parser.parse_args()
    
    scored_dataset, metrics = score_dataset(
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        eval_workers=args.eval_workers,
        eval_batch_size=args.eval_batch_size,
        eval_timeout_s=args.eval_timeout_s,
        require_solution_class=not args.no_require_solution_class,
        max_tests=args.max_tests,
    )
    
    # Print metrics
    print("\n" + "=" * 60)
    print("Scoring Results:")
    print("=" * 60)
    print(f"  Total samples: {metrics['total_samples']}")
    print(f"  Number of problems: {metrics['num_problems']}")
    print(f"  Samples per problem: {metrics['samples_per_problem']}")
    print(f"  Original success rate: {metrics['original_success_rate']:.4f}")
    print(f"  Rescored success rate: {metrics['rescored_success_rate']:.4f}")
    print(f"  Reward match rate: {metrics['reward_match_rate']:.4f}")
    print(f"\nPass@k metrics (rescored):")
    print(f"  pass@1: {metrics['pass_at_1']:.4f}")
    if metrics['pass_at_2'] is not None:
        print(f"  pass@2: {metrics['pass_at_2']:.4f}")
    if metrics['pass_at_4'] is not None:
        print(f"  pass@4: {metrics['pass_at_4']:.4f}")
    if metrics['pass_at_8'] is not None:
        print(f"  pass@8: {metrics['pass_at_8']:.4f}")
    print("=" * 60)
    
    # Save if requested
    if args.output_dir:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        scored_dataset.save_to_disk(args.output_dir)
        
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved scored dataset to: {args.output_dir}")
        print(f"Saved metrics to: {metrics_path}")
    
    # Push to hub if requested
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.push_to_hub}")
        scored_dataset.push_to_hub(args.push_to_hub, private=False)
        print(f"Successfully pushed to: https://huggingface.co/datasets/{args.push_to_hub}")
    
    return scored_dataset, metrics


if __name__ == "__main__":
    main()
