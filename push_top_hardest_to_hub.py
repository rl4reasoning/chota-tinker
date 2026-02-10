#!/usr/bin/env python3
"""
Select the hardest 100, 200, 300, 400 problems (by success_rate@320 on single-turn),
then create and push to HuggingFace:

1. Intellect subsets: anirudhb11/intellect_3_code_very_hard_top_{x}_hardest  (x=100,200,300,400)
2. Single-turn subsets: anirudhb11/qwen3_4b_instruct_top_{x}_hardest_single_turn
3. Multi-turn subsets: anirudhb11/qwen3_4b_instruct_top_{x}_hardest_interations_10_turns

Uses the same ranking logic as plot_best_of_k.py (single-turn + multi-turn configs, common
problem_ids, rank by success_rate@320, take hardest first).
"""
import argparse
from datasets import load_dataset, Dataset, concatenate_datasets
from huggingface_hub import HfApi, repo_exists

# Match plot_best_of_k.py configs
MAX_PROBLEM_ID = 500
SINGLE_TURN_CONFIGS = [
    ("anirudhb11/qwen3_4b_instruct_start_0_end_125_single_turn", 0),
    ("anirudhb11/qwen3_4b_instruct_start_125_end_250_single_turn", 125),
    ("anirudhb11/qwen3_4b_instruct_start_250_end_375_single_turn", 250),
    ("anirudhb11/qwen3_4b_instruct_start_375_end_500_single_turn", 375),
]
MULTITURN_CONFIGS = [
    ("anirudhb11/qwen3_4b_instruct_start_0_end_70_interations_10_turns", 0),
    ("anirudhb11/qwen3_4b_instruct_start_70_end_140_interations_10_turns", 70),
    ("anirudhb11/qwen3_4b_instruct_start_140_end_210_interations_10_turns", 140),
    ("anirudhb11/qwen3_4b_instruct_start_210_end_280_interations_10_turns", 210),
    ("anirudhb11/qwen3_4b_instruct_start_350_end_420_interations_10_turns", 350),
    ("anirudhb11/qwen3_4b_instruct_start_420_end_490_interations_10_turns", 420),
    ("anirudhb11/qwen3_4b_instruct_start_490_end_560_interations_10_turns", 490),
]
INTELLECT_SOURCE = "bicycleman15/intellect_3_code_very_hard"
NUM_HARDEST_LIST = [100, 200, 300, 400]


def _load_rewards_by_problem(dataset_configs, max_problem_id):
    """Return dict[global_problem_id] = list of final_reward (for ranking)."""
    merged = {}
    for name, offset in dataset_configs:
        ds = load_dataset(name, split="train")
        df = ds.to_pandas()
        if "trajectory_id" in df.columns:
            df = df.sort_values(["problem_id", "trajectory_id"])
        for _, row in df.iterrows():
            pid = row["problem_id"] + offset
            if max_problem_id is not None and pid >= max_problem_id:
                continue
            if pid not in merged:
                merged[pid] = []
            merged[pid].append(row["final_reward"])
    return merged


def compute_hardest_problem_ids():
    """Same ranking as plot_best_of_k: common problem_ids, rank by success_rate@320 (ascending)."""
    print("Loading single-turn and multi-turn for ranking...")
    single = _load_rewards_by_problem(SINGLE_TURN_CONFIGS, MAX_PROBLEM_ID)
    multi = _load_rewards_by_problem(MULTITURN_CONFIGS, MAX_PROBLEM_ID)
    common = set(single.keys()) & set(multi.keys())
    single = {p: single[p] for p in common}
    print(f"Common problem_ids: {len(common)}")
    problems_with_320 = [p for p in common if len(single[p]) >= 320]
    rate_at_320 = []
    for pid in problems_with_320:
        rewards = single[pid][:320]
        sr = sum(1 for r in rewards if r == 1.0) / 320.0
        rate_at_320.append((pid, sr))
    rate_at_320.sort(key=lambda x: x[1])
    hardest_ids = [pid for pid, _ in rate_at_320]
    rates = [sr for _, sr in rate_at_320]
    return hardest_ids, rates


def ensure_repo(repo_id, private=False):
    if not repo_exists(repo_id, repo_type="dataset"):
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        print(f"  Created repo: {repo_id}")


def push_intellect_subset(hardest_ids, x, private=False):
    """Subset bicycleman15/intellect_3_code_very_hard by row index (row i = problem i). Select rows at indices hardest_ids[:x], add problem_id 0..x-1, push to anirudhb11/intellect_3_code_very_hard_top_{x}_hardest."""
    repo_id = f"anirudhb11/intellect_3_code_very_hard_top_{x}_hardest"
    hardest_list = hardest_ids[:x]
    print(f"\n--- Intellect top-{x} ---")
    ds = load_dataset(INTELLECT_SOURCE, split="train")
    # Intellect has no problem_id column; row index = problem index. Select those rows.
    indices_to_keep = [i for i in hardest_list if i < len(ds)]
    if not indices_to_keep:
        print(f"  No rows to keep; skipping {repo_id}")
        return
    ds = ds.select(indices_to_keep)
    # Add original_problem_id: indices from the source dataset (hardest first)
    ds = ds.add_column("original_problem_id", indices_to_keep)
    # Add problem_id: sequential 0, 1, 2, 3...
    ds = ds.add_column("problem_id", list(range(len(indices_to_keep))))
    ensure_repo(repo_id, private=private)
    ds.push_to_hub(repo_id, commit_message=f"Top {x} hardest problems (by success_rate@320)")
    print(f"  Pushed {len(ds)} rows -> {repo_id}")


def push_single_turn_subset(hardest_ids, x, private=False):
    """Build single-turn dataset with same schema as qwen3_4b single_turn, only hardest x problems; push to anirudhb11/qwen3_4b_instruct_top_{x}_hardest_single_turn. problem_id = original global index (same as intellect subset)."""
    repo_id = f"anirudhb11/qwen3_4b_instruct_top_{x}_hardest_single_turn"
    print(f"\n--- Single-turn top-{x} ---")
    hardest_set = set(hardest_ids[:x])
    parts = []
    for name, offset in SINGLE_TURN_CONFIGS:
        ds = load_dataset(name, split="train")
        indices = [i for i, row in enumerate(ds) if (row["problem_id"] + offset) in hardest_set]
        if not indices:
            continue
        ds = ds.select(indices)
        # problem_id = original global index (literal hardest problem id), not 0..x-1
        global_pids = [ds[i]["problem_id"] + offset for i in range(len(ds))]
        # ds = ds.remove_columns("problem_id")
        ds = ds.add_column("original_problem_id", global_pids)
        parts.append(ds)
    if not parts:
        print(f"  No rows; skipping {repo_id}")
        return
    out = concatenate_datasets(parts)
    ensure_repo(repo_id, private=private)
    out.push_to_hub(repo_id, commit_message=f"Top {x} hardest problems, single-turn")
    print(f"  Pushed {len(out)} rows -> {repo_id}")


def push_multiturn_subset(hardest_ids, x, private=False):
    """Build multi-turn dataset, only hardest x problems; push to anirudhb11/qwen3_4b_instruct_top_{x}_hardest_interations_10_turns. problem_id = original global index (same as intellect subset)."""
    repo_id = f"anirudhb11/qwen3_4b_instruct_top_{x}_hardest_interations_10_turns"
    print(f"\n--- Multi-turn top-{x} ---")
    hardest_set = set(hardest_ids[:x])
    parts = []
    for name, offset in MULTITURN_CONFIGS:
        ds = load_dataset(name, split="train")
        indices = [i for i, row in enumerate(ds) if (row["problem_id"] + offset) in hardest_set]
        if not indices:
            continue
        ds = ds.select(indices)
        # problem_id = original global index (literal hardest problem id), not 0..x-1
        global_pids = [ds[i]["problem_id"] + offset for i in range(len(ds))]
        # ds = ds.remove_columns("problem_id")
        ds = ds.add_column("original_problem_id", global_pids)
        parts.append(ds)    
    if not parts:
        print(f"  No rows; skipping {repo_id}")
        return
    out = concatenate_datasets(parts)
    ensure_repo(repo_id, private=private)
    out.push_to_hub(repo_id, commit_message=f"Top {x} hardest problems, 10 turns")
    print(f"  Pushed {len(out)} rows -> {repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Push top-x hardest subsets (intellect, single-turn, multi-turn) to HF")
    parser.add_argument("--dry-run", action="store_true", help="Only compute and print hardest IDs, do not push")
    parser.add_argument("--private", action="store_true", help="Push datasets as private")
    parser.add_argument("--x", type=int, default=None, choices=[100, 200, 300, 400],
                        help="Only push for this x (default: all 100,200,300,400)")
    args = parser.parse_args()

    hardest_ids, rates_at_320 = compute_hardest_problem_ids()
    print(f"Ranked {len(hardest_ids)} problems (hardest first).")
    
    # Print first few problem IDs for each top-k
    print("\nFirst 5 problem IDs for each top-k:")
    for x in NUM_HARDEST_LIST:
        ids = hardest_ids[:x]
        print(f"  top-{x}: {ids[:5]} ... (total {len(ids)})")
    # Assertion: first 300 hardest have success_rate@320 == 0
    assert len(rates_at_320) >= 300, f"Need at least 300 problems for assertion, got {len(rates_at_320)}"
    assert all(r == 0.0 for r in rates_at_320[:300]), (
        f"Expected first 300 rate_at_320 to be 0; got {[r for r in rates_at_320[:300] if r != 0.0][:5]}..."
    )

    if args.dry_run:
        for x in NUM_HARDEST_LIST:
            print(f"  Top {x}: problem_ids = {hardest_ids[:x]}...")
        return

    to_run = [args.x] if args.x is not None else NUM_HARDEST_LIST
    for x in to_run:
        if x > len(hardest_ids):
            print(f"Skipping x={x}: only {len(hardest_ids)} problems available.")
            continue
        push_intellect_subset(hardest_ids, x, private=args.private)
        push_single_turn_subset(hardest_ids, x, private=args.private)
        push_multiturn_subset(hardest_ids, x, private=args.private)

    print("\nDone.")


if __name__ == "__main__":
    main()
