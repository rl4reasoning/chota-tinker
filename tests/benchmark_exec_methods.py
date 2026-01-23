#!/usr/bin/env python3
"""
Benchmark comparing _exec_interaction_code (in-process) vs _exec_interaction_code_subprocess.

Uses the bicycleman15/1k_32_interactions dataset to get real interaction code samples.
"""

import argparse
import json
import time
import statistics
from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset
from tqdm import tqdm

import sys
sys.path.insert(0, '/Users/jp7467/Desktop/chota-tinker')

from intellect_env import _exec_interaction_code, _exec_interaction_code_subprocess


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    method: str
    code_snippet: str  # First 100 chars for identification
    success: bool
    duration_ms: float
    stdout_len: int
    stderr_len: int
    error: Optional[str] = None


def extract_interaction_codes(dataset, max_samples: int = 100) -> list[str]:
    """Extract interaction code samples from the dataset."""
    codes = []
    
    for row in dataset:
        if len(codes) >= max_samples:
            break
            
        interactions_str = row.get("interactions", "[]")
        try:
            interactions = json.loads(interactions_str)
        except (json.JSONDecodeError, TypeError):
            continue
            
        for interaction in interactions:
            if len(codes) >= max_samples:
                break
            code = interaction.get("code", "")
            if code and len(code.strip()) > 0:
                codes.append(code)
    
    return codes


def benchmark_single(code: str, method: str, timeout_s: float) -> BenchmarkResult:
    """Benchmark a single code execution with a specific method."""
    snippet = code[:100].replace('\n', '\\n')
    
    if method == "in_process":
        exec_func = _exec_interaction_code
    else:
        exec_func = _exec_interaction_code_subprocess
    
    start = time.perf_counter()
    try:
        success, stdout, stderr = exec_func(code, timeout_s)
        duration_ms = (time.perf_counter() - start) * 1000
        return BenchmarkResult(
            method=method,
            code_snippet=snippet,
            success=success,
            duration_ms=duration_ms,
            stdout_len=len(stdout),
            stderr_len=len(stderr),
        )
    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        return BenchmarkResult(
            method=method,
            code_snippet=snippet,
            success=False,
            duration_ms=duration_ms,
            stdout_len=0,
            stderr_len=0,
            error=str(e),
        )


def run_benchmark(
    codes: list[str],
    timeout_s: float = 5.0,
    warmup_runs: int = 3,
) -> tuple[list[BenchmarkResult], list[BenchmarkResult]]:
    """Run benchmark comparing both methods."""
    
    # Warmup runs to reduce cold-start effects
    if warmup_runs > 0 and len(codes) > 0:
        print(f"Running {warmup_runs} warmup iterations...")
        warmup_code = "print('warmup')"
        for _ in range(warmup_runs):
            _exec_interaction_code(warmup_code, timeout_s)
            _exec_interaction_code_subprocess(warmup_code, timeout_s)
    
    in_process_results: list[BenchmarkResult] = []
    subprocess_results: list[BenchmarkResult] = []
    
    print(f"\nBenchmarking {len(codes)} code samples...")
    
    for code in tqdm(codes, desc="Benchmarking"):
        # Run in-process first
        result_in = benchmark_single(code, "in_process", timeout_s)
        in_process_results.append(result_in)
        
        # Run subprocess
        result_sub = benchmark_single(code, "subprocess", timeout_s)
        subprocess_results.append(result_sub)
    
    return in_process_results, subprocess_results


def compute_stats(results: list[BenchmarkResult]) -> dict:
    """Compute statistics from benchmark results."""
    durations = [r.duration_ms for r in results]
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]
    timeouts = [r for r in failures if r.duration_ms >= 4900]  # Close to 5s timeout
    
    return {
        "count": len(results),
        "successes": len(successes),
        "failures": len(failures),
        "timeouts": len(timeouts),
        "mean_ms": statistics.mean(durations) if durations else 0,
        "median_ms": statistics.median(durations) if durations else 0,
        "stdev_ms": statistics.stdev(durations) if len(durations) > 1 else 0,
        "min_ms": min(durations) if durations else 0,
        "max_ms": max(durations) if durations else 0,
        "p10_ms": sorted(durations)[len(durations) // 10] if len(durations) >= 10 else 0,
        "p90_ms": sorted(durations)[len(durations) * 9 // 10] if len(durations) >= 10 else 0,
        "p99_ms": sorted(durations)[len(durations) * 99 // 100] if len(durations) >= 100 else 0,
    }


def print_comparison(in_process_stats: dict, subprocess_stats: dict):
    """Print a comparison table of the two methods."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS: _exec_interaction_code vs _exec_interaction_code_subprocess")
    print("=" * 80)
    
    print(f"\n{'Metric':<25} {'In-Process':<20} {'Subprocess':<20} {'Ratio':<15}")
    print("-" * 80)
    
    metrics = [
        ("Total samples", "count", False),
        ("Successes", "successes", False),
        ("Failures", "failures", False),
        ("Timeouts", "timeouts", False),
        ("Mean (ms)", "mean_ms", True),
        ("Median (ms)", "median_ms", True),
        ("Std Dev (ms)", "stdev_ms", True),
        ("Min (ms)", "min_ms", True),
        ("Max (ms)", "max_ms", True),
        ("P10 (ms)", "p10_ms", True),
        ("P90 (ms)", "p90_ms", True),
        ("P99 (ms)", "p99_ms", True),
    ]
    
    for label, key, is_timing in metrics:
        in_val = in_process_stats[key]
        sub_val = subprocess_stats[key]
        
        if is_timing:
            ratio = sub_val / in_val if in_val > 0 else float('inf')
            print(f"{label:<25} {in_val:>15.2f} ms   {sub_val:>15.2f} ms   {ratio:>10.2f}x")
        else:
            print(f"{label:<25} {in_val:>18}    {sub_val:>18}")
    
    print("-" * 80)
    
    # Summary
    speedup = subprocess_stats["mean_ms"] / in_process_stats["mean_ms"] if in_process_stats["mean_ms"] > 0 else float('inf')
    print(f"\nSummary:")
    print(f"  - In-process execution is {speedup:.2f}x faster on average")
    print(f"  - Subprocess overhead: ~{subprocess_stats['mean_ms'] - in_process_stats['mean_ms']:.2f}ms per execution")


def analyze_discrepancies(
    codes: list[str],
    in_process_results: list[BenchmarkResult],
    subprocess_results: list[BenchmarkResult],
    show_max: int = 5,
):
    """Analyze cases where the two methods gave different results."""
    print(f"\n{'=' * 80}")
    print("DISCREPANCY ANALYSIS")
    print("=" * 80)
    
    discrepancies = []
    for i, (code, in_res, sub_res) in enumerate(zip(codes, in_process_results, subprocess_results)):
        if in_res.success != sub_res.success:
            discrepancies.append({
                "index": i,
                "code": code,
                "in_process": in_res,
                "subprocess": sub_res,
            })
    
    print(f"\nFound {len(discrepancies)} discrepancies where success/failure differed")
    
    if discrepancies and show_max > 0:
        print(f"\nShowing first {min(show_max, len(discrepancies))} discrepancies:")
        for disc in discrepancies[:show_max]:
            print(f"\n--- Discrepancy #{disc['index']} ---")
            print(f"Code (first 200 chars):\n{disc['code'][:200]}...")
            print(f"\nIn-Process: success={disc['in_process'].success}, "
                  f"duration={disc['in_process'].duration_ms:.2f}ms")
            if disc['in_process'].error:
                print(f"  Error: {disc['in_process'].error}")
            print(f"Subprocess: success={disc['subprocess'].success}, "
                  f"duration={disc['subprocess'].duration_ms:.2f}ms")
            if disc['subprocess'].error:
                print(f"  Error: {disc['subprocess'].error}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark exec methods")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Maximum number of code samples to benchmark")
    parser.add_argument("--timeout", type=float, default=5.0,
                        help="Timeout per execution in seconds")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup runs")
    parser.add_argument("--show-discrepancies", type=int, default=5,
                        help="Number of discrepancies to show in detail")
    parser.add_argument("--dataset", type=str, default="bicycleman15/1k_32_interactions",
                        help="HuggingFace dataset to use")
    args = parser.parse_args()
    
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train")
    print(f"Dataset loaded: {len(dataset)} rows")
    
    print(f"\nExtracting interaction codes (max {args.max_samples})...")
    codes = extract_interaction_codes(dataset, max_samples=args.max_samples)
    print(f"Extracted {len(codes)} code samples")
    
    if not codes:
        print("No code samples found!")
        return
    
    # Show sample code lengths
    code_lengths = [len(c) for c in codes]
    print(f"Code lengths: min={min(code_lengths)}, max={max(code_lengths)}, "
          f"avg={statistics.mean(code_lengths):.0f}")
    
    # Run benchmark
    in_process_results, subprocess_results = run_benchmark(
        codes,
        timeout_s=args.timeout,
        warmup_runs=args.warmup,
    )
    
    # Compute and display statistics
    in_process_stats = compute_stats(in_process_results)
    subprocess_stats = compute_stats(subprocess_results)
    
    print_comparison(in_process_stats, subprocess_stats)
    
    # Analyze discrepancies
    analyze_discrepancies(
        codes,
        in_process_results,
        subprocess_results,
        show_max=args.show_discrepancies,
    )
    
    print("\n" + "=" * 80)
    print("NOTES:")
    print("=" * 80)
    print("""
- In-process (_exec_interaction_code):
  * Uses exec() directly in the current process
  * Timeout via SIGALRM (cannot interrupt C extensions that hold GIL)
  * Faster but less safe (can corrupt process state)

- Subprocess (_exec_interaction_code_subprocess):
  * Spawns a new Python process for each execution
  * Hard timeout via subprocess.run() (can kill hung processes)
  * Slower due to process spawn overhead (~30-50ms typical)
  * Safer and more reliable for untrusted code
""")


if __name__ == "__main__":
    main()
