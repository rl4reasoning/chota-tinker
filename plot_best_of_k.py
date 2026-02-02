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


def compute_pass_at_k(problem_results: dict[int, list[float]], k: int) -> float:
    """
    Compute pass@K metric - fraction of problems solved (reward == 1.0) in first k samples.
    
    Args:
        problem_results: Dict mapping problem_id to list of final rewards
        k: Number of samples to consider
    
    Returns:
        Fraction of problems that have at least one sample with reward == 1.0 in first k
    """
    num_passed = 0
    count = 0
    
    for problem_id, rewards in problem_results.items():
        if len(rewards) >= k:
            # Check if any of the first k samples has reward == 1.0
            if any(r == 1.0 for r in rewards[:k]):
                num_passed += 1
            count += 1
    
    return num_passed / count if count > 0 else 0.0


def compute_mean_at_k(problem_results: dict[int, list[float]], k: int) -> float:
    """
    Compute mean@K metric - average reward across first k samples.
    
    Args:
        problem_results: Dict mapping problem_id to list of final rewards
        k: Number of samples to consider
    
    Returns:
        Average of mean reward over first k samples across all problems
    """
    total_mean = 0.0
    count = 0
    
    for problem_id, rewards in problem_results.items():
        if len(rewards) >= k:
            mean_reward = sum(rewards[:k]) / k
            total_mean += mean_reward
            count += 1
    
    return total_mean / count if count > 0 else 0.0


def compute_success_rate_at_k(problem_results: dict[int, list[float]], k: int) -> float:
    """
    Compute success rate@K - fraction of samples with reward == 1.0 among first k samples.
    
    Args:
        problem_results: Dict mapping problem_id to list of final rewards
        k: Number of samples to consider
    
    Returns:
        Average success rate (count of reward==1.0 / k) across all problems
    """
    total_rate = 0.0
    count = 0
    
    for problem_id, rewards in problem_results.items():
        if len(rewards) >= k:
            success_count = sum(1 for r in rewards[:k] if r == 1.0)
            success_rate = success_count / k
            total_rate += success_rate
            count += 1
    
    return total_rate / count if count > 0 else 0.0


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


def load_and_merge_datasets(dataset_configs: list[tuple[str, int]]) -> dict[int, list[float]]:
    """Load multiple datasets and merge them by problem_id with offset.
    
    Args:
        dataset_configs: List of (dataset_name, problem_id_offset) tuples
    """
    merged_results = {}
    
    for dataset_name, offset in dataset_configs:
        print(f"Loading {dataset_name} (offset={offset})...")
        ds = load_dataset(dataset_name, split="train")
        df = ds.to_pandas()
        df = df.sort_values(['problem_id', 'trajectory_id'])
        
        for _, row in df.iterrows():
            # Apply offset to get global problem_id
            problem_id = row['problem_id'] + offset
            final_reward = row['final_reward']
            
            if problem_id not in merged_results:
                merged_results[problem_id] = []
            merged_results[problem_id].append(final_reward)
    
    return merged_results


def load_and_merge_datasets_with_filter(dataset_configs: list[tuple[str, int]], max_problem_id: int = None) -> dict[int, list[float]]:
    """Load multiple datasets and merge them by problem_id with offset, with optional filtering.
    
    Args:
        dataset_configs: List of (dataset_name, problem_id_offset) tuples
        max_problem_id: If specified, only include problems with global id < max_problem_id
    """
    merged_results = {}
    
    for dataset_name, offset in dataset_configs:
        print(f"Loading {dataset_name} (offset={offset})...")
        ds = load_dataset(dataset_name, split="train")
        df = ds.to_pandas()
        df = df.sort_values(['problem_id', 'trajectory_id'])
        
        for _, row in df.iterrows():
            # Apply offset to get global problem_id
            problem_id = row['problem_id'] + offset
            
            # Filter by max_problem_id if specified
            if max_problem_id is not None and problem_id >= max_problem_id:
                continue
                
            final_reward = row['final_reward']
            
            if problem_id not in merged_results:
                merged_results[problem_id] = []
            merged_results[problem_id].append(final_reward)
    
    return merged_results


def load_and_merge_rsa_datasets(dataset_configs: list[tuple[str, int]], split: str = "step_10") -> dict[int, list[float]]:
    """Load multiple RSA datasets from a specific split and merge them by problem_id with offset.
    
    Args:
        dataset_configs: List of (dataset_name, problem_id_offset) tuples
        split: The split to load (default: "step_10")
    """
    merged_results = {}
    
    for dataset_name, offset in dataset_configs:
        print(f"Loading {dataset_name} (split={split}, offset={offset})...")
        ds = load_dataset(dataset_name, split=split)
        df = ds.to_pandas()
        
        # RSA dataset may have different column structure, check for it
        if 'problem_id' in df.columns:
            df = df.sort_values(['problem_id'])
        elif 'problem_index' in df.columns:
            df['problem_id'] = df['problem_index']
            df = df.sort_values(['problem_id'])
        
        for _, row in df.iterrows():
            problem_id = (row['problem_id'] if 'problem_id' in row else row['problem_index']) + offset
            final_reward = row['final_reward']
            
            if problem_id not in merged_results:
                merged_results[problem_id] = []
            merged_results[problem_id].append(final_reward)
    
    return merged_results


def load_rsa_by_step(dataset_configs: list[tuple[str, int]], num_steps: int = 10) -> dict[int, dict[int, list[float]]]:
    """Load RSA datasets and organize by problem_id and step.
    
    Returns:
        Dict mapping problem_id -> {step_num -> list of rewards for all candidates at that step}
    """
    # problem_id -> step_num -> list of rewards
    results_by_step = {}
    
    for dataset_name, offset in dataset_configs:
        print(f"Loading {dataset_name} (all steps, offset={offset})...")
        
        for step in range(1, num_steps + 1):
            split_name = f"step_{step}"
            try:
                ds = load_dataset(dataset_name, split=split_name)
            except Exception as e:
                print(f"  Warning: Could not load {split_name} from {dataset_name}: {e}")
                continue
            
            df = ds.to_pandas()
            
            # Handle different column names
            if 'problem_index' in df.columns and 'problem_id' not in df.columns:
                df['problem_id'] = df['problem_index']
            
            for _, row in df.iterrows():
                problem_id = row['problem_id'] + offset
                final_reward = row['final_reward']
                
                if problem_id not in results_by_step:
                    results_by_step[problem_id] = {}
                if step not in results_by_step[problem_id]:
                    results_by_step[problem_id][step] = []
                results_by_step[problem_id][step].append(final_reward)
    
    return results_by_step


def compute_best_at_step(results_by_step: dict[int, dict[int, list[float]]], step: int) -> float:
    """
    Compute best@step_k: average of max reward across all candidates at step k for each problem.
    
    Args:
        results_by_step: Dict mapping problem_id -> {step_num -> list of rewards}
        step: The RSA step number (1-indexed)
    
    Returns:
        Average of max reward at this step across all problems
    """
    total_reward = 0.0
    count = 0
    
    for problem_id, step_rewards in results_by_step.items():
        if step in step_rewards and len(step_rewards[step]) > 0:
            best_reward = max(step_rewards[step])
            total_reward += best_reward
            count += 1
    
    return total_reward / count if count > 0 else 0.0


def compute_pass_at_step(results_by_step: dict[int, dict[int, list[float]]], step: int) -> float:
    """
    Compute pass@step_k: fraction of problems with at least one reward == 1.0 at step k.
    """
    num_passed = 0
    count = 0
    
    for problem_id, step_rewards in results_by_step.items():
        if step in step_rewards and len(step_rewards[step]) > 0:
            if any(r == 1.0 for r in step_rewards[step]):
                num_passed += 1
            count += 1
    
    return num_passed / count if count > 0 else 0.0


def compute_mean_at_step(results_by_step: dict[int, dict[int, list[float]]], step: int) -> float:
    """
    Compute mean@step_k: average reward across all candidates at step k for each problem.
    """
    total_mean = 0.0
    count = 0
    
    for problem_id, step_rewards in results_by_step.items():
        if step in step_rewards and len(step_rewards[step]) > 0:
            mean_reward = sum(step_rewards[step]) / len(step_rewards[step])
            total_mean += mean_reward
            count += 1
    
    return total_mean / count if count > 0 else 0.0


def compute_success_rate_at_step(results_by_step: dict[int, dict[int, list[float]]], step: int) -> float:
    """
    Compute success rate@step: fraction of candidates with reward == 1.0 at step k.
    """
    total_rate = 0.0
    count = 0
    
    for problem_id, step_rewards in results_by_step.items():
        if step in step_rewards and len(step_rewards[step]) > 0:
            rewards = step_rewards[step]
            success_count = sum(1 for r in rewards if r == 1.0)
            success_rate = success_count / len(rewards)
            total_rate += success_rate
            count += 1
    
    return total_rate / count if count > 0 else 0.0


def main():
    # Multi-turn interaction datasets (10 turns per sample)
    # Format: (dataset_name, problem_id_offset)
    # Only 0-75 available
    multiturn_datasets = [
        ("anirudhb11/0_25_interations_10_attempts_new_pr", 0),     # problem 0-24
        ("anirudhb11/25_50_interations_10_attempts_new_pr", 25),   # problem 25-49
        ("anirudhb11/50_75_interations_10_attempts_new_pr", 50),   # problem 50-74
    ]
    # S1 datasets
    # Format: (dataset_name, problem_id_offset)
    # Only 0-75 available
    s1_datasets = [
        ("anirudhb11/0_25_s1_10_attempts", 0),      # problem 0-24
        ("anirudhb11/25_50_s1_10_attempts", 25),    # problem 25-49
        ("anirudhb11/50_75_s1_10_attempts", 50),    # problem 50-74
    ]
    
    # Single-turn datasets (higher K, up to 320)
    # Note: single_50_150 covers 50-150 but we restrict to problem_id < 75
    single_turn_datasets = [
        ("bicycleman15/single_0_50", 0),     # problem 0-49
        ("bicycleman15/single_50_150", 50),  # problem 50-149, filter to < 75
    ]
    
    # RSA datasets - 0-75, use split="step_10"
    rsa_datasets = [
        ("anirudhb11/0_25_rsa_pop_32_k_4_steps_10_v2", 0),      # problem 0-24
        ("anirudhb11/25_50_rsa_pop_32_k_4_steps_10_v2", 25),    # problem 25-49
        ("anirudhb11/50_75_rsa_pop_32_k_4_steps_10_v2", 50),    # problem 50-74
    ]
    
    # Load and merge datasets
    print("Loading Multi-turn datasets...")
    interactions_results = load_and_merge_datasets(multiturn_datasets)
    
    print("\nLoading S1 datasets...")
    s1_results = load_and_merge_datasets(s1_datasets)
    
    print("\nLoading Single-turn datasets...")
    # Use filtered version to restrict to problem_id < 75
    single_turn_results = load_and_merge_datasets_with_filter(single_turn_datasets, max_problem_id=75)
    
    print("\nLoading RSA datasets (all steps)...")
    rsa_results_by_step = load_rsa_by_step(rsa_datasets, num_steps=10)
    
    # Print dataset info
    print(f"\nDataset Summary:")
    print(f"Multi-turn: {len(interactions_results)} problems")
    if interactions_results:
        sample_counts = [len(v) for v in interactions_results.values()]
        print(f"  Samples per problem: min={min(sample_counts)}, max={max(sample_counts)}")
    
    print(f"S1: {len(s1_results)} problems")
    if s1_results:
        sample_counts = [len(v) for v in s1_results.values()]
        print(f"  Samples per problem: min={min(sample_counts)}, max={max(sample_counts)}")
    
    print(f"Single-turn: {len(single_turn_results)} problems")
    if single_turn_results:
        sample_counts = [len(v) for v in single_turn_results.values()]
        print(f"  Samples per problem: min={min(sample_counts)}, max={max(sample_counts)}")
    
    print(f"RSA (by step): {len(rsa_results_by_step)} problems")
    if rsa_results_by_step:
        # Check how many steps and candidates per step
        sample_problem = next(iter(rsa_results_by_step.values()))
        num_steps = max(sample_problem.keys())
        sample_counts = [len(sample_problem.get(s, [])) for s in range(1, num_steps + 1)]
        print(f"  Steps: {num_steps}, Candidates per step: {sample_counts[0] if sample_counts else 0}")
    
    # Compute best-of-K for K from 1 to 32 for multi-turn and s1
    k_values_32 = list(range(1, 33))
    # Compute best-of-K for K from 1 to 320 for single-turn
    k_values_320 = list(range(1, 321))
    # RSA uses steps 1-10, but x-axis is step * 32 (total forward passes)
    rsa_step_values = list(range(1, 11))
    rsa_x_values = [step * 32 for step in rsa_step_values]  # 32, 64, ..., 320
    
    # Best-of-K scores (continuous rewards)
    s1_scores = []
    interactions_scores = []
    single_turn_scores = []
    
    # Mean@K scores
    s1_mean_scores = []
    interactions_mean_scores = []
    single_turn_mean_scores = []
    
    # Success rate@K scores (fraction of samples with reward == 1.0)
    s1_success_rate_scores = []
    interactions_success_rate_scores = []
    single_turn_success_rate_scores = []
    
    # RSA scores by step
    rsa_step_scores = []
    rsa_step_pass_scores = []
    rsa_step_mean_scores = []
    rsa_step_success_rate_scores = []
    
    # Pass@K scores (binary: reward == 1.0)
    s1_pass_scores = []
    interactions_pass_scores = []
    single_turn_pass_scores = []
    
    print("\n=== Best-of-K Results (Continuous Rewards) ===")
    for k in k_values_32:
        s1_score = compute_best_of_k(s1_results, k)
        interactions_score = compute_best_of_k(interactions_results, k)
        
        s1_scores.append(s1_score)
        interactions_scores.append(interactions_score)
        
        # Pass@K
        s1_pass_scores.append(compute_pass_at_k(s1_results, k))
        interactions_pass_scores.append(compute_pass_at_k(interactions_results, k))
        
        # Mean@K
        s1_mean_scores.append(compute_mean_at_k(s1_results, k))
        interactions_mean_scores.append(compute_mean_at_k(interactions_results, k))
        
        # Success rate@K
        s1_success_rate_scores.append(compute_success_rate_at_k(s1_results, k))
        interactions_success_rate_scores.append(compute_success_rate_at_k(interactions_results, k))
        
        if k in [1, 4, 8, 16, 32]:
            print(f"K={k:2d}: S1={s1_score:.4f}, Multi-turn={interactions_score:.4f}")
    
    # Compute for single-turn up to K=320
    for k in k_values_320:
        single_turn_score = compute_best_of_k(single_turn_results, k)
        single_turn_scores.append(single_turn_score)
        single_turn_pass_scores.append(compute_pass_at_k(single_turn_results, k))
        single_turn_mean_scores.append(compute_mean_at_k(single_turn_results, k))
        single_turn_success_rate_scores.append(compute_success_rate_at_k(single_turn_results, k))
    
    # Compute RSA by step (best across all candidates at each step)
    print(f"\nRSA (best@step across all parallel populations):")
    for step in rsa_step_values:
        rsa_score = compute_best_at_step(rsa_results_by_step, step)
        rsa_pass = compute_pass_at_step(rsa_results_by_step, step)
        rsa_mean = compute_mean_at_step(rsa_results_by_step, step)
        rsa_success_rate = compute_success_rate_at_step(rsa_results_by_step, step)
        rsa_step_scores.append(rsa_score)
        rsa_step_pass_scores.append(rsa_pass)
        rsa_step_mean_scores.append(rsa_mean)
        rsa_step_success_rate_scores.append(rsa_success_rate)
        print(f"  Step {step:2d}: Best={rsa_score:.4f}, Mean={rsa_mean:.4f}, SuccessRate={rsa_success_rate:.4f}, Pass={rsa_pass:.4f}")
    
    print(f"\nSingle-turn (up to 320 samples):")
    for k in [1, 8, 32, 64, 128, 256, 320]:
        if k <= len(single_turn_scores):
            print(f"  Best@{k}: {single_turn_scores[k-1]:.4f}")
    
    print("\n=== Pass@K Results (Binary: reward == 1.0) ===")
    for k in [1, 4, 8, 16, 32]:
        print(f"K={k:2d}: S1={s1_pass_scores[k-1]:.4f}, Multi-turn={interactions_pass_scores[k-1]:.4f}")
    
    print(f"\nSingle-turn (up to 320 samples):")
    for k in [1, 8, 32, 64, 128, 256, 320]:
        if k <= len(single_turn_pass_scores):
            print(f"  Pass@{k}: {single_turn_pass_scores[k-1]:.4f}")
    
    # Create figure with four subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Best-of-K (continuous rewards)
    ax1.plot(k_values_320, single_turn_scores, '-', color='#9B59B6', label='Single-turn (K)', 
             linewidth=1.5)
    ax1.plot(k_values_32, interactions_scores, '-o', color='#FF6B6B', label='Multi-turn (K, 10 turns each)', 
             markersize=3, linewidth=1.5)
    ax1.plot(k_values_32, s1_scores, '-o', color='#4ECDC4', label='S1 (K)', 
             markersize=3, linewidth=1.5)
    ax1.plot(rsa_x_values, rsa_step_scores, '-s', color='#F39C12', label='RSA (step*32)', 
             markersize=5, linewidth=2)
    
    ax1.set_xlabel('K (samples)')
    ax1.set_ylabel('Best-of-K Reward')
    ax1.set_title('Best-of-K (Continuous Rewards)', fontsize=12, style='italic')
    ax1.legend(loc='lower right', frameon=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([1, 10, 32, 64, 128, 192, 256, 320])
    
    # Plot 2: Mean@K
    ax2.plot(k_values_320, single_turn_mean_scores, '-', color='#9B59B6', label='Single-turn (K)', 
             linewidth=1.5)
    ax2.plot(k_values_32, interactions_mean_scores, '-o', color='#FF6B6B', label='Multi-turn (K, 10 turns each)', 
             markersize=3, linewidth=1.5)
    ax2.plot(k_values_32, s1_mean_scores, '-o', color='#4ECDC4', label='S1 (K)', 
             markersize=3, linewidth=1.5)
    ax2.plot(rsa_x_values, rsa_step_mean_scores, '-s', color='#F39C12', label='RSA (step*32)', 
             markersize=5, linewidth=2)
    
    ax2.set_xlabel('K (samples)')
    ax2.set_ylabel('Mean@K Reward')
    ax2.set_title('Mean@K (Average Reward)', fontsize=12, style='italic')
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([1, 10, 32, 64, 128, 192, 256, 320])
    
    # Plot 3: Pass@K (binary rewards)
    ax3.plot(k_values_320, single_turn_pass_scores, '-', color='#9B59B6', label='Single-turn (K)', 
             linewidth=1.5)
    ax3.plot(k_values_32, interactions_pass_scores, '-o', color='#FF6B6B', label='Multi-turn (K, 10 turns each)', 
             markersize=3, linewidth=1.5)
    ax3.plot(k_values_32, s1_pass_scores, '-o', color='#4ECDC4', label='S1 (K)', 
             markersize=3, linewidth=1.5)
    ax3.plot(rsa_x_values, rsa_step_pass_scores, '-s', color='#F39C12', label='RSA (step*32)', 
             markersize=5, linewidth=2)
    
    ax3.set_xlabel('K (samples)')
    ax3.set_ylabel('Pass@K')
    ax3.set_title('Pass@K (Any reward == 1.0)', fontsize=12, style='italic')
    ax3.legend(loc='lower right', frameon=True)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks([1, 10, 32, 64, 128, 192, 256, 320])
    
    # Plot 4: Success Rate@K (fraction of samples with reward == 1.0)
    ax4.plot(k_values_320, single_turn_success_rate_scores, '-', color='#9B59B6', label='Single-turn (K)', 
             linewidth=1.5)
    ax4.plot(k_values_32, interactions_success_rate_scores, '-o', color='#FF6B6B', label='Multi-turn (K, 10 turns each)', 
             markersize=3, linewidth=1.5)
    ax4.plot(k_values_32, s1_success_rate_scores, '-o', color='#4ECDC4', label='S1 (K)', 
             markersize=3, linewidth=1.5)
    ax4.plot(rsa_x_values, rsa_step_success_rate_scores, '-s', color='#F39C12', label='RSA (step*32)', 
             markersize=5, linewidth=2)
    
    ax4.set_xlabel('K (samples)')
    ax4.set_ylabel('Success Rate@K')
    ax4.set_title('Success Rate@K (Fraction with reward == 1.0)', fontsize=12, style='italic')
    ax4.legend(loc='upper right', frameon=True)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks([1, 10, 32, 64, 128, 192, 256, 320])
    
    plt.tight_layout()
    plt.savefig('best_of_k_comparison.png', dpi=150)
    plt.savefig('best_of_k_comparison.pdf')
    print("\nPlots saved to best_of_k_comparison.png and best_of_k_comparison.pdf")
    
    # Also print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"\nS1 (problems 0-74):")
    print(f"  Best@1:  {s1_scores[0]:.4f}  |  Pass@1:  {s1_pass_scores[0]:.4f}")
    print(f"  Best@8:  {s1_scores[7]:.4f}  |  Pass@8:  {s1_pass_scores[7]:.4f}")
    print(f"  Best@32: {s1_scores[31]:.4f}  |  Pass@32: {s1_pass_scores[31]:.4f}")
    
    print(f"\nMulti-turn (10 turns each, problems 0-74):")
    print(f"  Best@1:  {interactions_scores[0]:.4f}  |  Pass@1:  {interactions_pass_scores[0]:.4f}")
    print(f"  Best@8:  {interactions_scores[7]:.4f}  |  Pass@8:  {interactions_pass_scores[7]:.4f}")
    print(f"  Best@32: {interactions_scores[31]:.4f}  |  Pass@32: {interactions_pass_scores[31]:.4f}")
    
    print(f"\nSingle-turn (problems 0-74):")
    print(f"  Best@1:   {single_turn_scores[0]:.4f}  |  Pass@1:   {single_turn_pass_scores[0]:.4f}")
    print(f"  Best@32:  {single_turn_scores[31]:.4f}  |  Pass@32:  {single_turn_pass_scores[31]:.4f}")
    if len(single_turn_scores) >= 320:
        print(f"  Best@320: {single_turn_scores[319]:.4f}  |  Pass@320: {single_turn_pass_scores[319]:.4f}")
    
    print(f"\nRSA (best@step across pop=32, problems 0-74):")
    print(f"  Step 1:  {rsa_step_scores[0]:.4f}  |  Pass@Step1:  {rsa_step_pass_scores[0]:.4f}")
    print(f"  Step 5:  {rsa_step_scores[4]:.4f}  |  Pass@Step5:  {rsa_step_pass_scores[4]:.4f}")
    print(f"  Step 10: {rsa_step_scores[9]:.4f}  |  Pass@Step10: {rsa_step_pass_scores[9]:.4f}")


if __name__ == "__main__":
    main()
