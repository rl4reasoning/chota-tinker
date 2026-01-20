"""Plot best-of-K curves comparing s1, interactions, and single-turn datasets."""

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from collections import defaultdict


def compute_best_of_k(problem_results: dict[int, list[float]], k: int) -> float:
    """
    Compute best-of-K metric using max of final rewards.
    
    Args:
        problem_results: Dict mapping problem_id to list of final rewards
        k: Number of samples to consider
    
    Returns:
        Average of max reward over first k samples across all problems
    """
    total_reward = 0.0
    count = 0
    
    for problem_id, rewards in problem_results.items():
        if len(rewards) >= k:
            best_reward = max(rewards[:k])
            total_reward += best_reward
            count += 1
    
    return total_reward / count if count > 0 else 0.0


def load_and_process_dataset(dataset_name: str, max_problem_id: int = None) -> dict[int, list[float]]:
    """Load dataset and organize final rewards by problem_id."""
    print(f"Loading {dataset_name}...")
    ds = load_dataset(dataset_name, split="train")
    
    # Group by problem_id
    problem_results = defaultdict(list)
    
    # Sort by problem_id and trajectory_id to ensure consistent ordering
    df = ds.to_pandas()
    df = df.sort_values(['problem_id', 'trajectory_id'])
    
    for _, row in df.iterrows():
        problem_id = row['problem_id']
        # Filter to first N problems if specified
        if max_problem_id is not None and problem_id >= max_problem_id:
            continue
        final_reward = row['final_reward']
        problem_results[problem_id].append(final_reward)
    
    return dict(problem_results)


def main():
    # Use all 1000 problems (only S1 and Multi-turn have all 1000)
    max_problems = 1000
    
    # Load S1 and Multi-turn datasets
    s1_results = load_and_process_dataset("bicycleman15/1k_32_s1", max_problem_id=max_problems)
    interactions_results = load_and_process_dataset("bicycleman15/1k_32_interactions", max_problem_id=max_problems)
    
    print(f"\nUsing first {max_problems} problems")
    print(f"S1: {len(s1_results)} problems, {len(list(s1_results.values())[0])} trajectories each")
    print(f"Multi-turn: {len(interactions_results)} problems, {len(list(interactions_results.values())[0])} trajectories each")
    
    # Compute best-of-K for K from 1 to 32
    k_values = list(range(1, 33))
    
    s1_scores = []
    interactions_scores = []
    
    for k in k_values:
        s1_score = compute_best_of_k(s1_results, k)
        interactions_score = compute_best_of_k(interactions_results, k)
        
        s1_scores.append(s1_score)
        interactions_scores.append(interactions_score)
        
        if k in [1, 4, 8, 16, 32]:
            print(f"K={k:2d}: S1={s1_score:.4f}, Multi-turn={interactions_score:.4f}")
    
    # Plot with style matching reference
    plt.figure(figsize=(8, 6))
    
    # Colors: coral for multi-turn, teal for S1
    plt.plot(k_values, interactions_scores, '-o', color='#FF6B6B', label='Multi-turn', 
             markersize=3, linewidth=1.5)
    plt.plot(k_values, s1_scores, '-o', color='#4ECDC4', label='S1', 
             markersize=3, linewidth=1.5)
    
    plt.xlabel('K (samples)')
    plt.ylabel('Best-of-K Reward')
    plt.title('Best-of-K (Oracle Selection) - 1000 Problems', fontsize=12, style='italic')
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks
    plt.xticks([1, 5, 10, 15, 20, 25, 30])
    
    plt.tight_layout()
    plt.savefig('best_of_k_comparison.png', dpi=150)
    plt.savefig('best_of_k_comparison.pdf')
    print("\nPlots saved to best_of_k_comparison.png and best_of_k_comparison.pdf")
    
    # Also print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"\nS1 (32 trajectories):")
    print(f"  Best@1:  {s1_scores[0]:.4f}")
    print(f"  Best@8:  {s1_scores[7]:.4f}")
    print(f"  Best@32: {s1_scores[31]:.4f}")
    
    print(f"\nMulti-turn (32 trajectories):")
    print(f"  Best@1:  {interactions_scores[0]:.4f}")
    print(f"  Best@8:  {interactions_scores[7]:.4f}")
    print(f"  Best@32: {interactions_scores[31]:.4f}")


if __name__ == "__main__":
    main()
