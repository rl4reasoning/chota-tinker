"""Merge multiple HuggingFace datasets into a single dataset.

Handles problem_id and trajectory_id offsets to ensure unique IDs across merged dataset.

Usage:
    # Merge multiple datasets:
    python merge_datasets.py \
        --datasets bicycleman15/prompt_v2_single_turn_0_25 \
                   bicycleman15/prompt_v2_single_turn_25_50 \
                   bicycleman15/prompt_v2_single_turn_50_75 \
        --output-dir artifacts/merged_dataset \
        --push-to-hub username/merged_single_turn

    # With custom problem ID offsets (if not auto-detecting from dataset name):
    python merge_datasets.py \
        --datasets dataset1 dataset2 dataset3 \
        --problem-offsets 0 25 50 \
        --output-dir artifacts/merged_dataset
"""

import argparse
import json
import re
from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm import tqdm


def extract_problem_range_from_name(dataset_name: str) -> tuple[int, int] | None:
    """Try to extract problem range from dataset name like 'prefix_0_25' -> (0, 25)."""
    # Match patterns like _0_25, _25_50, etc. at the end of the name
    match = re.search(r'_(\d+)_(\d+)$', dataset_name)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        return start, end
    return None


def merge_datasets(
    dataset_names: list[str],
    problem_offsets: list[int] | None = None,
    split: str = "train",
) -> tuple[Dataset, dict]:
    """Merge multiple datasets with proper ID offsets.
    
    Args:
        dataset_names: List of HuggingFace dataset names to merge
        problem_offsets: Optional list of problem_id offsets for each dataset.
                        If None, will try to auto-detect from dataset names.
        split: Dataset split to load (default: "train")
    
    Returns:
        Tuple of (merged_dataset, metadata_dict)
    """
    if len(dataset_names) == 0:
        raise ValueError("At least one dataset name is required")
    
    # Auto-detect problem offsets if not provided
    if problem_offsets is None:
        problem_offsets = []
        for name in dataset_names:
            range_info = extract_problem_range_from_name(name)
            if range_info:
                problem_offsets.append(range_info[0])
            else:
                # If can't detect, use cumulative offset based on previous datasets
                if not problem_offsets:
                    problem_offsets.append(0)
                else:
                    print(f"Warning: Could not detect problem range from '{name}', "
                          f"using sequential offset")
                    problem_offsets.append(-1)  # Mark for later calculation
    
    if len(problem_offsets) != len(dataset_names):
        raise ValueError(f"Number of offsets ({len(problem_offsets)}) must match "
                        f"number of datasets ({len(dataset_names)})")
    
    print(f"Merging {len(dataset_names)} datasets...")
    
    all_rows = []
    cumulative_problems = 0
    cumulative_trajectories = 0
    dataset_info = []
    
    for idx, (dataset_name, problem_offset) in enumerate(zip(dataset_names, problem_offsets)):
        print(f"\n[{idx + 1}/{len(dataset_names)}] Loading: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split)
        
        # Determine actual problem offset
        if problem_offset == -1:
            # Auto-calculate based on cumulative problems
            actual_problem_offset = cumulative_problems
        else:
            actual_problem_offset = problem_offset
        
        # Find max problem_id and trajectory_id in this dataset
        max_problem_id = max(row.get("problem_id", 0) for row in dataset)
        
        # Count trajectories per problem for trajectory offset calculation
        trajectories_per_problem = {}
        for row in dataset:
            pid = row.get("problem_id", 0)
            tid = row.get("trajectory_id", 0)
            if pid not in trajectories_per_problem:
                trajectories_per_problem[pid] = 0
            trajectories_per_problem[pid] = max(trajectories_per_problem[pid], tid + 1)
        
        num_problems = max_problem_id + 1
        
        print(f"  - {len(dataset)} rows, {num_problems} problems")
        print(f"  - Problem ID offset: {actual_problem_offset}")
        
        # Process rows with offset
        for row in tqdm(dataset, desc=f"  Processing", leave=False):
            new_row = dict(row)
            
            # Apply problem_id offset
            original_problem_id = row.get("problem_id", 0)
            new_row["problem_id"] = original_problem_id + actual_problem_offset
            
            # Keep original IDs for reference
            new_row["original_problem_id"] = original_problem_id
            new_row["original_trajectory_id"] = row.get("trajectory_id", 0)
            new_row["source_dataset"] = dataset_name
            
            all_rows.append(new_row)
        
        # Update cumulative counters
        cumulative_problems = actual_problem_offset + num_problems
        cumulative_trajectories += len(dataset)
        
        dataset_info.append({
            "name": dataset_name,
            "num_rows": len(dataset),
            "num_problems": num_problems,
            "problem_offset": actual_problem_offset,
            "problem_range": [actual_problem_offset, actual_problem_offset + num_problems - 1],
        })
    
    # Create merged dataset
    merged_dataset = Dataset.from_list(all_rows)
    
    # Compute merged statistics
    unique_problems = len(set(row["problem_id"] for row in all_rows))
    successful_rows = sum(1 for row in all_rows if row.get("is_successful", False) or row.get("final_reward", 0) > 0)
    
    metadata = {
        "total_rows": len(all_rows),
        "total_unique_problems": unique_problems,
        "successful_rows": successful_rows,
        "success_rate": successful_rows / len(all_rows) if all_rows else 0,
        "source_datasets": dataset_info,
    }
    
    print(f"\n{'=' * 60}")
    print(f"Merge Complete:")
    print(f"  Total rows: {metadata['total_rows']}")
    print(f"  Unique problems: {metadata['total_unique_problems']}")
    print(f"  Success rate: {metadata['success_rate']:.4f}")
    print(f"{'=' * 60}")
    
    return merged_dataset, metadata


def main():
    parser = argparse.ArgumentParser(description="Merge multiple HuggingFace datasets")
    parser.add_argument("--datasets", type=str, nargs="+", required=True,
                        help="List of HuggingFace dataset names to merge")
    parser.add_argument("--problem-offsets", type=int, nargs="*", default=None,
                        help="Optional problem_id offsets for each dataset. "
                             "If not provided, will try to auto-detect from dataset names.")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to load (default: train)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save merged dataset (optional)")
    parser.add_argument("--push-to-hub", type=str, default=None,
                        help="HuggingFace repo to push merged dataset")
    
    args = parser.parse_args()
    
    merged_dataset, metadata = merge_datasets(
        dataset_names=args.datasets,
        problem_offsets=args.problem_offsets,
        split=args.split,
    )
    
    # Print source dataset info
    print("\nSource datasets:")
    for info in metadata["source_datasets"]:
        print(f"  - {info['name']}")
        print(f"      Rows: {info['num_rows']}, Problems: {info['num_problems']}")
        print(f"      Problem range: {info['problem_range'][0]} - {info['problem_range'][1]}")
    
    # Save if requested
    if args.output_dir:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        merged_dataset.save_to_disk(args.output_dir)
        
        metadata_path = os.path.join(args.output_dir, "merge_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\nSaved merged dataset to: {args.output_dir}")
        print(f"Saved metadata to: {metadata_path}")
    
    # Push to hub if requested
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.push_to_hub}")
        merged_dataset.push_to_hub(args.push_to_hub, private=False)
        print(f"Successfully pushed to: https://huggingface.co/datasets/{args.push_to_hub}")
    
    return merged_dataset, metadata


if __name__ == "__main__":
    main()
