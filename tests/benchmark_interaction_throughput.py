#!/usr/bin/env python3
"""
Benchmark interaction throughput for the bicycleman15/1k_32_interactions dataset.

Measures how fast we can execute all interactions for the first 100 problems × 32 trajectories.
"""

import argparse
import json
import time
import statistics
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.insert(0, '/Users/jp7467/Desktop/chota-tinker')

from intellect_env import _exec_interaction_code_subprocess


@dataclass
class TrajectoryResult:
    """Result from processing a single trajectory."""
    problem_id: int
    trajectory_id: int
    num_interactions: int
    total_duration_ms: float
    interaction_durations_ms: list[float] = field(default_factory=list)
    successes: int = 0
    failures: int = 0


@dataclass
class InteractionResult:
    """Result from a single interaction execution."""
    success: bool
    duration_ms: float


def execute_interaction(code: str, timeout_s: float = 1.0) -> InteractionResult:
    """Execute a single interaction and return the result."""
    start = time.perf_counter()
    try:
        success, stdout, stderr = _exec_interaction_code_subprocess(code, timeout_s)
        duration_ms = (time.perf_counter() - start) * 1000
        return InteractionResult(success=success, duration_ms=duration_ms)
    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        return InteractionResult(success=False, duration_ms=duration_ms)


def process_trajectory(
    row: dict,
    timeout_s: float = 1.0,
) -> TrajectoryResult:
    """Process all interactions in a single trajectory."""
    problem_id = row.get("problem_id", -1)
    trajectory_id = row.get("trajectory_id", -1)
    
    interactions_str = row.get("interactions", "[]")
    try:
        interactions = json.loads(interactions_str)
    except (json.JSONDecodeError, TypeError):
        interactions = []
    
    result = TrajectoryResult(
        problem_id=problem_id,
        trajectory_id=trajectory_id,
        num_interactions=len(interactions),
        total_duration_ms=0.0,
    )
    
    trajectory_start = time.perf_counter()
    
    for interaction in interactions:
        code = interaction.get("code", "")
        if code and len(code.strip()) > 0:
            exec_result = execute_interaction(code, timeout_s)
            result.interaction_durations_ms.append(exec_result.duration_ms)
            if exec_result.success:
                result.successes += 1
            else:
                result.failures += 1
    
    result.total_duration_ms = (time.perf_counter() - trajectory_start) * 1000
    return result


def load_subset(dataset, max_problems: int = 100, trajectories_per_problem: int = 32) -> list[dict]:
    """Load a subset of the dataset: first N problems with all trajectories."""
    rows = []
    seen_problems = set()
    
    for row in dataset:
        problem_id = row.get("problem_id", -1)
        if problem_id >= max_problems:
            continue
        seen_problems.add(problem_id)
        rows.append(row)
        
        # Stop if we have all desired problems
        if len(seen_problems) >= max_problems and len(rows) >= max_problems * trajectories_per_problem:
            break
    
    return rows


def run_sequential_benchmark(
    rows: list[dict],
    timeout_s: float = 1.0,
) -> list[TrajectoryResult]:
    """Run benchmark sequentially (one trajectory at a time)."""
    results = []
    
    for row in tqdm(rows, desc="Processing trajectories"):
        result = process_trajectory(row, timeout_s)
        results.append(result)
    
    return results


def _process_trajectory_worker(args: tuple) -> TrajectoryResult:
    """Worker function for parallel processing."""
    row, timeout_s = args
    return process_trajectory(row, timeout_s)


def run_parallel_benchmark(
    rows: list[dict],
    timeout_s: float = 1.0,
    num_workers: int = 8,
) -> list[TrajectoryResult]:
    """Run benchmark in parallel using ProcessPoolExecutor."""
    results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(_process_trajectory_worker, (row, timeout_s)): i
            for i, row in enumerate(rows)
        }
        
        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing (workers={num_workers})"):
            result = future.result()
            results.append(result)
    
    return results


def print_results(results: list[TrajectoryResult], total_time_s: float, mode: str):
    """Print benchmark results."""
    print(f"\n{'=' * 80}")
    print(f"BENCHMARK RESULTS ({mode})")
    print("=" * 80)
    
    total_trajectories = len(results)
    total_interactions = sum(r.num_interactions for r in results)
    total_successes = sum(r.successes for r in results)
    total_failures = sum(r.failures for r in results)
    
    # Aggregate all interaction durations
    all_interaction_durations = []
    for r in results:
        all_interaction_durations.extend(r.interaction_durations_ms)
    
    # Trajectory durations
    trajectory_durations = [r.total_duration_ms for r in results]
    
    print(f"\nOverall Statistics:")
    print(f"  Total trajectories:    {total_trajectories:,}")
    print(f"  Total interactions:    {total_interactions:,}")
    print(f"  Successful:            {total_successes:,} ({100*total_successes/max(1,total_interactions):.1f}%)")
    print(f"  Failed:                {total_failures:,} ({100*total_failures/max(1,total_interactions):.1f}%)")
    print(f"  Wall-clock time:       {total_time_s:.2f}s")
    print(f"  Throughput:            {total_interactions/total_time_s:.1f} interactions/sec")
    print(f"  Throughput:            {total_trajectories/total_time_s:.1f} trajectories/sec")
    
    if all_interaction_durations:
        print(f"\nInteraction Timing (per interaction):")
        print(f"  Mean:   {statistics.mean(all_interaction_durations):.2f} ms")
        print(f"  Median: {statistics.median(all_interaction_durations):.2f} ms")
        print(f"  Stdev:  {statistics.stdev(all_interaction_durations) if len(all_interaction_durations) > 1 else 0:.2f} ms")
        print(f"  Min:    {min(all_interaction_durations):.2f} ms")
        print(f"  Max:    {max(all_interaction_durations):.2f} ms")
        
        sorted_durations = sorted(all_interaction_durations)
        if len(sorted_durations) >= 10:
            print(f"  P10:    {sorted_durations[len(sorted_durations) // 10]:.2f} ms")
            print(f"  P90:    {sorted_durations[len(sorted_durations) * 9 // 10]:.2f} ms")
        if len(sorted_durations) >= 100:
            print(f"  P99:    {sorted_durations[len(sorted_durations) * 99 // 100]:.2f} ms")
    
    if trajectory_durations:
        print(f"\nTrajectory Timing (per trajectory):")
        print(f"  Mean:   {statistics.mean(trajectory_durations):.2f} ms")
        print(f"  Median: {statistics.median(trajectory_durations):.2f} ms")
        
        # Interactions per trajectory
        interactions_per_traj = [r.num_interactions for r in results]
        print(f"\nInteractions per trajectory:")
        print(f"  Mean:   {statistics.mean(interactions_per_traj):.1f}")
        print(f"  Min:    {min(interactions_per_traj)}")
        print(f"  Max:    {max(interactions_per_traj)}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark interaction throughput for the 1k_32_interactions dataset"
    )
    parser.add_argument("--max-problems", type=int, default=100,
                        help="Number of problems to process (default: 100)")
    parser.add_argument("--trajectories-per-problem", type=int, default=32,
                        help="Trajectories per problem (default: 32)")
    parser.add_argument("--timeout", type=float, default=1.0,
                        help="Timeout per interaction in seconds (default: 1.0)")
    parser.add_argument("--mode", choices=["sequential", "parallel", "both"], default="both",
                        help="Execution mode (default: both)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    parser.add_argument("--dataset", type=str, default="bicycleman15/1k_32_interactions",
                        help="HuggingFace dataset to use")
    args = parser.parse_args()
    
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train")
    print(f"Dataset loaded: {len(dataset)} rows")
    
    print(f"\nLoading first {args.max_problems} problems × {args.trajectories_per_problem} trajectories...")
    rows = load_subset(dataset, max_problems=args.max_problems, trajectories_per_problem=args.trajectories_per_problem)
    print(f"Loaded {len(rows)} trajectories")
    
    # Count total interactions
    total_interactions = 0
    for row in rows:
        try:
            interactions = json.loads(row.get("interactions", "[]"))
            total_interactions += len(interactions)
        except:
            pass
    print(f"Total interactions to process: {total_interactions:,}")
    
    # Warmup
    print("\nWarmup run...")
    warmup_code = "print('warmup')"
    for _ in range(3):
        _exec_interaction_code_subprocess(warmup_code, args.timeout)
    
    if args.mode in ["sequential", "both"]:
        print(f"\n{'=' * 80}")
        print("SEQUENTIAL BENCHMARK")
        print("=" * 80)
        
        start_time = time.perf_counter()
        seq_results = run_sequential_benchmark(rows, timeout_s=args.timeout)
        seq_time = time.perf_counter() - start_time
        
        print_results(seq_results, seq_time, "Sequential")
    
    if args.mode in ["parallel", "both"]:
        print(f"\n{'=' * 80}")
        print(f"PARALLEL BENCHMARK ({args.workers} workers)")
        print("=" * 80)
        
        start_time = time.perf_counter()
        par_results = run_parallel_benchmark(rows, timeout_s=args.timeout, num_workers=args.workers)
        par_time = time.perf_counter() - start_time
        
        print_results(par_results, par_time, f"Parallel ({args.workers} workers)")
        
        if args.mode == "both":
            speedup = seq_time / par_time if par_time > 0 else float('inf')
            print(f"\n{'=' * 80}")
            print("COMPARISON")
            print("=" * 80)
            print(f"Sequential time:   {seq_time:.2f}s")
            print(f"Parallel time:     {par_time:.2f}s")
            print(f"Speedup:           {speedup:.2f}x")


if __name__ == "__main__":
    main()
