"""Test evaluation parity between old and new evaluation code.

This script loads model responses from a HuggingFace dataset and re-evaluates
them using the new evaluation infrastructure to verify consistency.

Example:
  python tests/test_eval_parity.py \
    --dataset bicycleman15/1k_160_single_turn \
    --max-rows 100 \
    --eval-timeout-s 5

  # Compare with original rewards
  python tests/test_eval_parity.py \
    --dataset bicycleman15/1k_160_single_turn \
    --max-rows 100 \
    --compare-original
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset

from utils.fast_eval import (
    EvalTask, evaluate_task, evaluate_tasks, _evaluate_code,
    _extract_answer_code, _select_tests, _normalize_io,
)


def _parse_json_field(value: Any) -> Any:
    """Parse JSON string field, return as-is if already parsed or on error."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def _extract_response(row: dict[str, Any]) -> str:
    """Extract assistant response from row."""
    messages = _parse_json_field(row.get("messages"))
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "") or ""
    
    # Fallback to common field names
    for fallback_key in ("response", "completion", "output"):
        value = row.get(fallback_key)
        if isinstance(value, str):
            return value
    return ""


def _build_tasks(
    dataset: Iterable[dict[str, Any]],
    max_rows: int,
    max_tests: int,
    timeout_s: float,
    require_solution_class: bool,
) -> tuple[list[EvalTask], list[dict[str, Any]], dict[str, int]]:
    """Build evaluation tasks from dataset rows."""
    tasks: list[EvalTask] = []
    row_data: list[dict[str, Any]] = []
    counters = {
        "rows_seen": 0,
        "rows_used": 0,
        "missing_tests": 0,
        "missing_response": 0,
    }

    for row in dataset:
        counters["rows_seen"] += 1
        if max_rows > 0 and counters["rows_seen"] > max_rows:
            break

        tests = _parse_json_field(row.get("tests"))
        if not isinstance(tests, dict) or "inputs" not in tests or "outputs" not in tests:
            counters["missing_tests"] += 1
            continue

        response = _extract_response(row)
        if not response:
            counters["missing_response"] += 1
            continue

        tasks.append(
            EvalTask(
                response=response,
                tests=tests,
                max_tests=max_tests,
                timeout_s=timeout_s,
                max_timeout_records=0,
                require_solution_class=require_solution_class,
            )
        )
        row_data.append({
            "problem_id": row.get("problem_id"),
            "trajectory_id": row.get("trajectory_id"),
            "original_reward": row.get("final_reward"),
            "original_is_successful": row.get("is_successful"),
            "fn_name": tests.get("fn_name"),
            "num_tests": len(tests.get("inputs", [])),
        })
        counters["rows_used"] += 1

    return tasks, row_data, counters


def debug_single_case(task: EvalTask, data: dict[str, Any], result, max_tests: int = 15):
    """Debug a single evaluation case by showing code, tests, and execution details."""
    print(f"\n{'=' * 80}")
    print(f"DEBUG: problem_id={data.get('problem_id')}, trajectory_id={data.get('trajectory_id')}")
    print(f"fn_name={data.get('fn_name')}, original_reward={data.get('original_reward')}, new_reward={result.reward}")
    print(f"{'=' * 80}")
    
    # Extract the code
    code = _extract_answer_code(task.response)
    if code:
        print(f"\n--- Extracted Code ({len(code)} chars) ---")
        # Show first 1500 chars of code
        if len(code) > 1500:
            print(code[:1500])
            print(f"\n... (truncated, {len(code) - 1500} more chars)")
        else:
            print(code)
    else:
        print("\n--- No code extracted ---")
        print(f"Response preview: {task.response[:500]}...")
        return
    
    # Show test cases
    tests = task.tests
    inputs, outputs = _select_tests(tests, max_tests)
    
    print(f"\n--- Test Cases ({len(inputs)} selected, {len(tests.get('inputs', []))} total) ---")
    for i, (inp, out) in enumerate(zip(inputs[:5], outputs[:5])):  # Show first 5
        inp_str = str(inp)[:200] if inp else "None"
        out_str = str(out)[:200] if out else "None"
        print(f"  Test {i}: input={inp_str}")
        print(f"          expected={out_str}")
    if len(inputs) > 5:
        print(f"  ... and {len(inputs) - 5} more tests")
    
    # Run evaluation with verbose output
    print(f"\n--- Running evaluation ---")
    from code_env.code_env.utils.deepcoder_utils import process_input_output, BASE_IMPORTS
    
    # Process inputs/outputs like we do in _evaluate_code
    processed_pairs = [process_input_output(inp, out) for inp, out in zip(inputs, outputs)]
    proc_inputs = [p[0] for p in processed_pairs]
    proc_outputs = [p[1] for p in processed_pairs]
    
    # Normalize
    norm_inputs = [str(_normalize_io(inp)) if inp is not None else "" for inp in proc_inputs]
    norm_outputs = [str(_normalize_io(out)) if out is not None else "" for out in proc_outputs]
    
    print(f"After processing:")
    for i in range(min(3, len(norm_inputs))):
        print(f"  Test {i}: input={norm_inputs[i][:100]}")
        print(f"          expected={norm_outputs[i][:100]}")
    
    # Try to run the code manually for the first test case
    print(f"\n--- Manual execution of first test ---")
    try:
        import subprocess
        import tempfile
        import os
        
        full_code = BASE_IMPORTS + "\n" + code
        
        # For stdin-based problems
        if not tests.get("fn_name"):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_code)
                temp_path = f.name
            
            try:
                result_run = subprocess.run(
                    ["python", temp_path],
                    input=norm_inputs[0],
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
                print(f"Exit code: {result_run.returncode}")
                print(f"Stdout: {result_run.stdout[:500]}")
                if result_run.stderr:
                    print(f"Stderr: {result_run.stderr[:500]}")
                print(f"Expected: {norm_outputs[0][:200]}")
            finally:
                os.unlink(temp_path)
        else:
            # For function-call problems
            print(f"Function-call problem with fn_name={tests.get('fn_name')}")
            print(f"Input args would be parsed from: {norm_inputs[0][:200]}")
    except Exception as e:
        print(f"Error during manual execution: {e}")


def analyze_discrepancies(
    tasks: list[EvalTask],
    row_data: list[dict[str, Any]],
    eval_results: list,
    tolerance: float = 0.001,
) -> dict[str, Any]:
    """Analyze discrepancies between original and new rewards."""
    discrepancies = []
    improved = []
    degraded = []
    
    for i, (task, data, result) in enumerate(zip(tasks, row_data, eval_results)):
        original = data.get("original_reward")
        new_reward = result.reward
        
        if original is None:
            continue
            
        diff = new_reward - original
        
        if abs(diff) > tolerance:
            entry = {
                "index": i,
                "problem_id": data.get("problem_id"),
                "trajectory_id": data.get("trajectory_id"),
                "original_reward": original,
                "new_reward": new_reward,
                "diff": diff,
                "fn_name": data.get("fn_name"),
                "is_leetcode": data.get("fn_name") is not None,
                "truncated": result.truncated,
                "invalidated": result.invalidated if hasattr(result, 'invalidated') else False,
                "task": task,  # Store task for debugging
            }
            discrepancies.append(entry)
            
            if diff > 0:
                improved.append(entry)
            else:
                degraded.append(entry)
    
    return {
        "discrepancies": discrepancies,
        "improved": improved,
        "degraded": degraded,
        "total_compared": len([d for d in row_data if d.get("original_reward") is not None]),
    }


def print_discrepancy_details(analysis: dict[str, Any], max_show: int = 20):
    """Print detailed discrepancy information."""
    discrepancies = analysis["discrepancies"]
    improved = analysis["improved"]
    degraded = analysis["degraded"]
    total = analysis["total_compared"]
    
    print(f"\n{'=' * 60}")
    print("Discrepancy Analysis")
    print(f"{'=' * 60}")
    print(f"Total rows compared: {total}")
    print(f"Total discrepancies: {len(discrepancies)}")
    print(f"  Improved (new > original): {len(improved)}")
    print(f"  Degraded (new < original): {len(degraded)}")
    
    if improved:
        print(f"\n--- Improved Cases (showing up to {max_show}) ---")
        for entry in improved[:max_show]:
            leetcode_str = "LeetCode" if entry["is_leetcode"] else "stdin"
            print(f"  [{entry['index']}] problem={entry['problem_id']}, traj={entry['trajectory_id']}, "
                  f"type={leetcode_str}, fn={entry['fn_name']}")
            print(f"       original={entry['original_reward']:.4f} -> new={entry['new_reward']:.4f} "
                  f"(+{entry['diff']:.4f})")
    
    if degraded:
        print(f"\n--- Degraded Cases (showing up to {max_show}) ---")
        for entry in degraded[:max_show]:
            leetcode_str = "LeetCode" if entry["is_leetcode"] else "stdin"
            trunc_str = " [TRUNCATED]" if entry["truncated"] else ""
            invalid_str = " [INVALIDATED]" if entry.get("invalidated") else ""
            print(f"  [{entry['index']}] problem={entry['problem_id']}, traj={entry['trajectory_id']}, "
                  f"type={leetcode_str}, fn={entry['fn_name']}{trunc_str}{invalid_str}")
            print(f"       original={entry['original_reward']:.4f} -> new={entry['new_reward']:.4f} "
                  f"({entry['diff']:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test evaluation parity using saved trajectories."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bicycleman15/1k_160_single_turn",
        help="HuggingFace dataset to load",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-rows", type=int, default=100)
    parser.add_argument("--max-tests", type=int, default=15)
    parser.add_argument("--eval-workers", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-timeout-s", type=float, default=5.0)
    parser.add_argument(
        "--compare-original",
        action="store_true",
        help="Compare new rewards with original final_reward from dataset",
    )
    parser.add_argument(
        "--show-discrepancies",
        type=int,
        default=20,
        help="Max number of discrepancies to show per category",
    )
    parser.add_argument(
        "--require-solution-class",
        action="store_true",
        default=True,
        help="Require Solution class for LeetCode-style problems",
    )
    parser.add_argument(
        "--no-require-solution-class",
        dest="require_solution_class",
        action="store_false",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--debug-degraded",
        type=int,
        default=0,
        metavar="N",
        help="Debug N degraded cases (show code, tests, execution details)",
    )
    parser.add_argument(
        "--debug-improved",
        type=int,
        default=0,
        metavar="N",
        help="Debug N improved cases (show code, tests, execution details)",
    )
    parser.add_argument(
        "--debug-problem",
        type=int,
        default=None,
        metavar="ID",
        help="Debug all cases for a specific problem_id",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Evaluation Parity Test")
    print(f"Dataset: {args.dataset} ({args.split})")
    print(f"Max rows: {args.max_rows}")
    print(f"Max tests per task: {args.max_tests}")
    print(f"Workers: {args.eval_workers} | Batch size: {args.eval_batch_size}")
    print(f"Timeout per test: {args.eval_timeout_s}s")
    print(f"Require Solution class: {args.require_solution_class}")
    print("=" * 60)

    print("\nLoading dataset...")
    dataset = load_dataset(args.dataset, split=args.split)
    print(f"Dataset loaded: {len(dataset)} rows")

    tasks, row_data, counters = _build_tasks(
        dataset,
        max_rows=args.max_rows,
        max_tests=args.max_tests,
        timeout_s=args.eval_timeout_s,
        require_solution_class=args.require_solution_class,
    )

    if not tasks:
        print("No tasks built. Check dataset fields or filters.")
        return

    # Count problem types
    leetcode_count = sum(1 for d in row_data if d.get("fn_name") is not None)
    stdin_count = len(row_data) - leetcode_count

    print(f"\nRows seen: {counters['rows_seen']}")
    print(f"Rows used: {counters['rows_used']}")
    print(f"Missing tests: {counters['missing_tests']}")
    print(f"Missing response: {counters['missing_response']}")
    print(f"Problem types: {leetcode_count} LeetCode-style, {stdin_count} stdin-based")

    print(f"\nEvaluating {len(tasks)} tasks...")
    eval_results = evaluate_tasks(
        tasks,
        max_workers=args.eval_workers,
        batch_size=args.eval_batch_size,
        show_progress=True,
    )

    # Basic statistics
    rewards = [res.reward for res in eval_results]
    solved = sum(1 for r in rewards if r > 0)
    perfect = sum(1 for r in rewards if r == 1.0)
    truncated = sum(1 for res in eval_results if res.truncated)
    invalidated = sum(1 for res in eval_results if getattr(res, 'invalidated', False))

    print(f"\n{'=' * 60}")
    print("Evaluation Results")
    print(f"{'=' * 60}")
    print(f"Tasks evaluated: {len(eval_results)}")
    print(f"Truncated (no code found): {truncated}")
    print(f"Invalidated (missing Solution class): {invalidated}")
    print(f"Any pass (reward > 0): {solved} ({solved / len(eval_results):.2%})")
    print(f"Perfect (reward = 1.0): {perfect} ({perfect / len(eval_results):.2%})")
    print(f"Average reward: {sum(rewards) / len(rewards):.4f}")

    if args.compare_original:
        analysis = analyze_discrepancies(tasks, row_data, eval_results)
        print_discrepancy_details(analysis, max_show=args.show_discrepancies)
        
        # Summary
        total = analysis["total_compared"]
        if total > 0:
            match_rate = 1 - len(analysis["discrepancies"]) / total
            print(f"\nMatch rate: {match_rate:.2%}")
        
        # Debug specific cases
        if args.debug_degraded > 0:
            print(f"\n{'#' * 80}")
            print(f"DEBUGGING {min(args.debug_degraded, len(analysis['degraded']))} DEGRADED CASES")
            print(f"{'#' * 80}")
            for entry in analysis["degraded"][:args.debug_degraded]:
                debug_single_case(entry["task"], entry, eval_results[entry["index"]], args.max_tests)
        
        if args.debug_improved > 0:
            print(f"\n{'#' * 80}")
            print(f"DEBUGGING {min(args.debug_improved, len(analysis['improved']))} IMPROVED CASES")
            print(f"{'#' * 80}")
            for entry in analysis["improved"][:args.debug_improved]:
                debug_single_case(entry["task"], entry, eval_results[entry["index"]], args.max_tests)
    
    # Debug specific problem
    if args.debug_problem is not None:
        print(f"\n{'#' * 80}")
        print(f"DEBUGGING ALL CASES FOR PROBLEM_ID={args.debug_problem}")
        print(f"{'#' * 80}")
        for i, (task, data, result) in enumerate(zip(tasks, row_data, eval_results)):
            if data.get("problem_id") == args.debug_problem:
                debug_single_case(task, data, result, args.max_tests)

    # Show per-problem-type breakdown
    print(f"\n{'=' * 60}")
    print("Breakdown by Problem Type")
    print(f"{'=' * 60}")
    
    leetcode_results = [(d, r) for d, r in zip(row_data, eval_results) if d.get("fn_name")]
    stdin_results = [(d, r) for d, r in zip(row_data, eval_results) if not d.get("fn_name")]
    
    if leetcode_results:
        lc_rewards = [r.reward for _, r in leetcode_results]
        lc_solved = sum(1 for r in lc_rewards if r > 0)
        lc_perfect = sum(1 for r in lc_rewards if r == 1.0)
        print(f"LeetCode-style ({len(leetcode_results)} tasks):")
        print(f"  Any pass: {lc_solved} ({lc_solved / len(leetcode_results):.2%})")
        print(f"  Perfect: {lc_perfect} ({lc_perfect / len(leetcode_results):.2%})")
        print(f"  Avg reward: {sum(lc_rewards) / len(lc_rewards):.4f}")
    
    if stdin_results:
        stdin_rewards = [r.reward for _, r in stdin_results]
        stdin_solved = sum(1 for r in stdin_rewards if r > 0)
        stdin_perfect = sum(1 for r in stdin_rewards if r == 1.0)
        print(f"Stdin-based ({len(stdin_results)} tasks):")
        print(f"  Any pass: {stdin_solved} ({stdin_solved / len(stdin_results):.2%})")
        print(f"  Perfect: {stdin_perfect} ({stdin_perfect / len(stdin_results):.2%})")
        print(f"  Avg reward: {sum(stdin_rewards) / len(stdin_rewards):.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
