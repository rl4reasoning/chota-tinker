"""Benchmark fast evaluator using saved trajectories from a HuggingFace dataset.

Example:
  python benchmark_fast_eval_from_dataset.py \
    --dataset bicycleman15/qwen3_4b_instruct_very_hard_single_turn_pass32 \
    --split train \
    --max-rows 256 \
    --eval-workers 8 \
    --eval-batch-size 8 \
    --eval-timeout-s 5

  python benchmark_fast_eval_from_dataset.py \
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
import time
from typing import Any, Iterable

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
) -> tuple[list[EvalTask], dict[str, int], int]:
    tasks: list[EvalTask] = []
    counters = {
        "rows_seen": 0,
        "rows_used": 0,
        "missing_tests": 0,
        "missing_response": 0,
    }
    total_selected_tests = 0

    for row in _iter_rows(dataset):
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

        inputs = tests.get("inputs", [])
        if isinstance(inputs, list):
            total_selected_tests += min(len(inputs), max_tests)

        tasks.append(
            EvalTask(
                response=response,
                tests=tests,
                max_tests=max_tests,
                timeout_s=timeout_s,
            )
        )
        counters["rows_used"] += 1

    return tasks, counters, total_selected_tests


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
    parser.add_argument("--show-progress", action="store_true")
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

    tasks, counters, total_selected_tests = _build_tasks(
        dataset,
        max_rows=args.max_rows,
        max_tests=args.max_tests,
        timeout_s=args.eval_timeout_s,
    )

    if not tasks:
        print("No tasks built. Check dataset fields or filters.")
        return

    print(
        f"Rows seen: {counters['rows_seen']} | "
        f"Rows used: {counters['rows_used']} | "
        f"Missing tests: {counters['missing_tests']} | "
        f"Missing response: {counters['missing_response']}"
    )
    print(f"Total selected tests: {total_selected_tests}")

    start = time.perf_counter()
    if args.mode == "parallel":
        eval_results = evaluate_tasks(
            tasks,
            max_workers=args.eval_workers,
            batch_size=args.eval_batch_size,
            show_progress=args.show_progress,
        )
    else:
        if args.show_progress:
            from tqdm import tqdm

            iterator = tqdm(tasks, desc="Evaluating", total=len(tasks))
        else:
            iterator = tasks
        eval_results = [evaluate_task(task) for task in iterator]
    elapsed = time.perf_counter() - start

    rewards = [res.reward for res in eval_results]
    solved = sum(1 for r in rewards if r > 0)
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    tasks_per_sec = len(eval_results) / elapsed if elapsed > 0 else 0.0
    tests_per_sec = total_selected_tests / elapsed if elapsed > 0 else 0.0

    print("\n" + "=" * 60)
    print("Benchmark results")
    print(f"Tasks evaluated: {len(eval_results)}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Tasks/sec: {tasks_per_sec:.2f}")
    print(f"Tests/sec (selected): {tests_per_sec:.2f}")
    print(f"Avg reward: {avg_reward:.4f}")
    print(f"Any-pass rate: {solved / len(eval_results):.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
