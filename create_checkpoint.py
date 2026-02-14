"""
Create streaming checkpoints for models and save to cache file.

Usage:
    python graph/create_checkpoint.py --model Qwen/Qwen3-4B-Instruct-2507
    python graph/create_checkpoint.py --model Qwen/Qwen3-8B-Instruct
"""

import argparse
import asyncio
import os

import tinker

CHECKPOINT_CACHE_FILE = os.path.join(os.path.dirname(__file__), "checkpoints.txt")


def load_checkpoints() -> dict[str, str]:
    """Load checkpoint cache from file."""
    checkpoints = {}
    if os.path.exists(CHECKPOINT_CACHE_FILE):
        with open(CHECKPOINT_CACHE_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("\t")
                    if len(parts) == 2:
                        model, path = parts
                        checkpoints[model] = path
    return checkpoints


def save_checkpoints(checkpoints: dict[str, str]):
    """Save checkpoint cache to file."""
    with open(CHECKPOINT_CACHE_FILE, "w") as f:
        f.write("# Model checkpoints for streaming (model_name<TAB>checkpoint_path)\n")
        f.write("# Created by create_checkpoint.py\n")
        f.write("#\n")
        for model, path in sorted(checkpoints.items()):
            f.write(f"{model}\t{path}\n")


def get_checkpoint(model: str) -> str | None:
    """Get cached checkpoint path for a model, or None if not cached."""
    checkpoints = load_checkpoints()
    return checkpoints.get(model)


async def create_checkpoint(model: str) -> str:
    """Create a streaming checkpoint for a model and return the path."""
    print(f"Creating checkpoint for {model}...")
    
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(base_model=model, rank=8)
    
    # Save weights and get the path
    save_result = training_client.save_weights_for_sampler(name="stream-ckpt").result()
    model_path = save_result.path
    
    print(f"  Checkpoint created: {model_path}")
    return model_path


async def main():
    parser = argparse.ArgumentParser(description="Create streaming checkpoints for models")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name to create checkpoint for")
    parser.add_argument("--force", action="store_true",
                        help="Force recreate even if checkpoint exists")
    parser.add_argument("--list", action="store_true", dest="list_checkpoints",
                        help="List all cached checkpoints")
    args = parser.parse_args()
    
    if args.list_checkpoints:
        checkpoints = load_checkpoints()
        if checkpoints:
            print("Cached checkpoints:")
            for model, path in sorted(checkpoints.items()):
                print(f"  {model}")
                print(f"    -> {path}")
        else:
            print("No cached checkpoints found.")
        return
    
    # Check if already cached
    checkpoints = load_checkpoints()
    if args.model in checkpoints and not args.force:
        print(f"Checkpoint already exists for {args.model}:")
        print(f"  {checkpoints[args.model]}")
        print("Use --force to recreate.")
        return
    
    # Create new checkpoint
    model_path = await create_checkpoint(args.model)
    
    # Save to cache
    checkpoints[args.model] = model_path
    save_checkpoints(checkpoints)
    print(f"Saved to {CHECKPOINT_CACHE_FILE}")


if __name__ == "__main__":
    asyncio.run(main())