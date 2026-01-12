"""
Create a filtered dataset containing only hard problems from INTELLECT-3-RL
and upload to Hugging Face Hub.
"""

from datasets import load_dataset, Dataset
import argparse


def create_hard_dataset(
    min_score: float = 0.1,
    max_score: float = 0.3,
    config: str = "code",
    split: str = "train",
):
    """
    Filter INTELLECT-3-RL to only include hard problems.
    
    Args:
        min_score: Minimum pass rate to include (default 0.1)
        max_score: Maximum pass rate to include (default 0.3)
    """
    print(f"Loading INTELLECT-3-RL dataset (config={config}, split={split})...")
    dataset = load_dataset("PrimeIntellect/INTELLECT-3-RL", config, split=split)
    
    print(f"Original dataset size: {len(dataset)}")
    
    # Filter for hard problems (pass rate between min and max)
    hard_dataset = dataset.filter(
        lambda x: x["avg@8_qwen3_4b_instruct_2507"] is not None 
                  and min_score <= x["avg@8_qwen3_4b_instruct_2507"] <= max_score
    )
    
    print(f"Filtered dataset size: {len(hard_dataset)} (pass rate {min_score} - {max_score})")
    
    # Show statistics
    scores = [ex["avg@8_qwen3_4b_instruct_2507"] for ex in hard_dataset]
    if scores:
        import statistics
        print(f"\nFiltered dataset statistics:")
        print(f"  Min score: {min(scores):.4f}")
        print(f"  Max score: {max(scores):.4f}")
        print(f"  Mean score: {statistics.mean(scores):.4f}")
    
    return hard_dataset


def upload_to_hub(dataset, repo_id: str, private: bool = False):
    """Upload dataset to Hugging Face Hub."""
    print(f"\nUploading to {repo_id}...")
    
    dataset.push_to_hub(
        repo_id,
        private=private,
        commit_message="Upload hard problems subset of INTELLECT-3-RL (pass rate 0.1 - 0.3)"
    )
    
    print(f"✓ Successfully uploaded to https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and upload hard problems dataset")
    parser.add_argument("--min-score", type=float, default=0.1,
                        help="Minimum pass rate to include (default: 0.1)")
    parser.add_argument("--max-score", type=float, default=0.3,
                        help="Maximum pass rate to include (default: 0.3)")
    parser.add_argument("--repo-id", type=str, default="bicycleman15/intellect_3_code_hard",
                        help="HuggingFace repo ID to upload to")
    parser.add_argument("--private", action="store_true",
                        help="Make the dataset private")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip uploading (just create and show stats)")
    
    args = parser.parse_args()
    
    # Create filtered dataset
    hard_dataset = create_hard_dataset(min_score=args.min_score, max_score=args.max_score)
    
    if not args.no_upload:
        upload_to_hub(hard_dataset, args.repo_id, args.private)
    else:
        print("\nSkipping upload (--no-upload flag set)")
