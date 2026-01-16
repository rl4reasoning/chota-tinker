"""Benchmark fast evaluator using saved trajectories from a HuggingFace dataset.

Example:
  python tests/benchmark_fast_eval_from_dataset.py \
    --dataset bicycleman15/qwen3_4b_instruct_very_hard_single_turn_pass32 \
    --split train \
    --max-rows 256 \
    --eval-workers 8 \
    --eval-batch-size 8 \
    --eval-timeout-s 5

  python tests/benchmark_fast_eval_from_dataset.py \
    --dataset bicycleman15/qwen3_4b_instruct_very_hard_single_turn_pass32 \
    --split train \
    --max-rows 256 \
    --mode sequential \
    --eval-timeout-s 5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset

from utils.fast_eval import EvalTask, evaluate_task, evaluate_tasks


def _parse_json_field(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def _extract_response(row: dict[str, Any]) -> str:
    messages = _parse_json_field(row.get("messages"))
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "") or ""

    for fallback_key in ("response", "completion", "output"):
        value = row.get(fallback_key)
        if isinstance(value, str):
            return value
    return ""


def _iter_rows(dataset: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for row in dataset:
        if isinstance(row, dict):
            yield row


def _build_tasks(
    dataset: Iterable[dict[str, Any]],
    max_rows: int,
    max_tests: int,
    timeout_s: float,
    max_timeout_records: int,
    require_solution_class: bool,
) -> tuple[list[EvalTask], dict[str, int], int, list[dict[str, Any]]]:
    tasks: list[EvalTask] = []
    counters = {
        "rows_seen": 0,
        "rows_used": 0,
        "missing_tests": 0,
        "missing_response": 0,
    }
    total_selected_tests = 0
    task_meta: list[dict[str, Any]] = []
    problem_types: dict[str, set[int]] = {"leetcode": set(), "stdin": set()}

    for row_index, row in enumerate(_iter_rows(dataset)):
        counters["rows_seen"] += 1
        if max_rows > 0 and counters["rows_seen"] > max_rows:
            break

        tests = _parse_json_field(row.get("tests"))
        if not isinstance(tests, dict) or "inputs" not in tests or "outputs" not in tests:
            counters["missing_tests"] += 1
            continue

        problem_id = row.get("problem_id")
        if problem_id is not None:
            fn_name = tests.get("fn_name")
            if fn_name:
                problem_types["leetcode"].add(problem_id)
            else:
                problem_types["stdin"].add(problem_id)

        response = _extract_response(row)
        if not response:
            counters["missing_response"] += 1
            continue

        inputs = tests.get("inputs", [])
        if isinstance(inputs, list):
            total_selected_tests += min(len(inputs), max_tests)

        tasks.append(
            EvalTask(
                response=response,
                tests=tests,
                max_tests=max_tests,
                timeout_s=timeout_s,
                max_timeout_records=max_timeout_records,
                require_solution_class=require_solution_class,
            )
        )
        task_meta.append(
            {
                "row_index": row_index,
                "problem_id": problem_id,
                "trajectory_id": row.get("trajectory_id"),
            }
        )
        counters["rows_used"] += 1

    return tasks, counters, total_selected_tests, task_meta, problem_types


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fast evaluator using saved trajectories."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bicycleman15/qwen3_4b_instruct_very_hard_single_turn_pass32",
        help="HuggingFace dataset to load",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument("--max-tests", type=int, default=12)
    parser.add_argument("--eval-workers", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-timeout-s", type=float, default=5.0)
    parser.add_argument(
        "--mode",
        type=str,
        default="parallel",
        choices=["parallel", "sequential"],
        help="Run evaluator using multiprocessing or sequential loop",
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--show-progress",
        dest="show_progress",
        action="store_true",
        default=None,
        help="Show tqdm progress bar (default: on for sequential mode)",
    )
    parser.add_argument(
        "--no-show-progress",
        dest="show_progress",
        action="store_false",
        default=None,
        help="Disable tqdm progress bar",
    )
    parser.add_argument(
        "--show-timeouts",
        action="store_true",
        help="Print per-task timeout indices (relative to selected tests)",
    )
    parser.add_argument(
        "--max-timeout-records",
        type=int,
        default=20,
        help="Max timeout indices to record per task when showing timeouts",
    )
    parser.add_argument(
        "--max-timeout-tasks",
        type=int,
        default=20,
        help="Max tasks to print timeout details for",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Fast evaluator benchmark")
    print(f"Dataset: {args.dataset} ({args.split})")
    print(f"Max rows: {args.max_rows}")
    print(f"Max tests per task: {args.max_tests}")
    print(f"Mode: {args.mode}")
    print(f"Workers: {args.eval_workers} | Batch size: {args.eval_batch_size}")
    print(f"Timeout per test: {args.eval_timeout_s}s")
    print("=" * 60)

    dataset = load_dataset(args.dataset, split=args.split)

    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed)
    elif args.seed:
        random.seed(args.seed)

    timeout_record_limit = args.max_timeout_records if args.show_timeouts else 0
    tasks, counters, total_selected_tests, task_meta, problem_types = _build_tasks(
        dataset,
        max_rows=args.max_rows,
        max_tests=args.max_tests,
        timeout_s=args.eval_timeout_s,
        max_timeout_records=timeout_record_limit,
        require_solution_class=True,
    )

    if not tasks:
        print("No tasks built. Check dataset fields or filters.")
        return

    leetcode_problems = sorted(problem_types.get("leetcode", set()))
    stdin_problems = sorted(problem_types.get("stdin", set()))

    print(
        f"Rows seen: {counters['rows_seen']} | "
        f"Rows used: {counters['rows_used']} | "
        f"Missing tests: {counters['missing_tests']} | "
        f"Missing response: {counters['missing_response']}"
    )
    print(f"Total selected tests: {total_selected_tests}")
    print(
        f"Problem types: LeetCode-style (problem_ids {leetcode_problems[:10]}{'...' if len(leetcode_problems) > 10 else ''}, "
        f"{len(leetcode_problems)} problems) | "
        f"Stdin-based (problem_ids {stdin_problems[:10]}{'...' if len(stdin_problems) > 10 else ''}, "
        f"{len(stdin_problems)} problems)"
    )

    show_progress = args.show_progress
    if show_progress is None:
        show_progress = args.mode == "sequential"

    start = time.perf_counter()
    if args.mode == "parallel":
        eval_results = evaluate_tasks(
            tasks,
            max_workers=args.eval_workers,
            batch_size=args.eval_batch_size,
            show_progress=show_progress,
        )
    else:
        iterator = tasks
        if show_progress:
            try:
                from tqdm import tqdm
            except ImportError:
                tqdm = None
                if args.show_progress is True:
                    print("tqdm not installed; proceeding without progress.")
            if tqdm is not None:
                iterator = tqdm(tasks, desc="Evaluating", total=len(tasks))
        eval_results = [evaluate_task(task) for task in iterator]
    elapsed = time.perf_counter() - start

    rewards = [res.reward for res in eval_results]
    solved = sum(1 for r in rewards if r > 0)
    invalidated = sum(1 for res in eval_results if res.truncated)
    valid_tasks = len(eval_results) - invalidated
    valid_rewards = [res.reward for res in eval_results if not res.truncated]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    avg_reward_valid = (
        sum(valid_rewards) / len(valid_rewards) if valid_rewards else 0.0
    )
    tasks_per_sec = len(eval_results) / elapsed if elapsed > 0 else 0.0
    tests_per_sec = total_selected_tests / elapsed if elapsed > 0 else 0.0
    if args.show_timeouts:
        timeout_total = sum(res.timeout_count for res in eval_results)
        timeout_rate = (
            timeout_total / total_selected_tests if total_selected_tests else 0.0
        )
        print(
            f"Timed-out tests: {timeout_total} "
            f"({timeout_rate:.2%} of selected tests)"
        )
        print("Timeout indices refer to selected tests after max-tests filtering.")
        if timeout_total:
            timeout_tasks = [
                (idx, res)
                for idx, res in enumerate(eval_results)
                if res.timeout_count > 0
            ]
            print(f"Tasks with timeouts: {len(timeout_tasks)}")
            for idx, res in timeout_tasks[: args.max_timeout_tasks]:
                meta = task_meta[idx] if idx < len(task_meta) else {}
                meta_bits = []
                if meta.get("problem_id") is not None:
                    meta_bits.append(f"problem_id={meta['problem_id']}")
                if meta.get("trajectory_id") is not None:
                    meta_bits.append(f"trajectory_id={meta['trajectory_id']}")
                if meta.get("row_index") is not None:
                    meta_bits.append(f"row_index={meta['row_index']}")
                meta_str = " | ".join(meta_bits) if meta_bits else "metadata unavailable"
                indices_display = (
                    list(res.timeout_indices)
                    if res.timeout_indices
                    else "[] (not recorded)"
                )
                print(
                    f"Task {idx} ({meta_str}) "
                    f"timeouts: {res.timeout_count} | "
                    f"indices: {indices_display}"
                )

    print("\n" + "=" * 60)
    print("Benchmark results")
    print(f"Tasks evaluated: {len(eval_results)}")
    print(f"Invalidated (has fn_name but no Solution class): {invalidated}")
    print(f"Valid tasks: {valid_tasks}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Tasks/sec: {tasks_per_sec:.2f}")
    print(f"Tests/sec (selected): {tests_per_sec:.2f}")
    print(f"Avg reward (all tasks): {avg_reward:.4f}")
    if valid_tasks > 0:
        print(f"Avg reward (valid tasks only): {avg_reward_valid:.4f}")
    print(f"Any-pass rate: {solved / len(eval_results):.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
