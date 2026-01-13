"""
Minimal RL training script on IFEval dataset.

Uses in-process vLLM with sleep/wake cycles for GPU memory management
and weight synchronization after each training step.
"""

import gc
import time

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from chota_tinker import (
    TrainingClient,
    SamplingClient,
    SamplingParams,
    ModelInput,
    prepare_rl_batch,
    flatten_rewards,
)
from ifeval import IFEvalDataset


def main():
    # Config
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    batch_size = 4
    num_samples = 4  # GRPO: multiple samples per prompt for advantage estimation
    max_steps = 100
    max_tokens = 256
    
    print("Initializing training client...")
    trainer = TrainingClient(
        model_name=model_name,
        lora_rank=None,  # Full model for now (LoRA sync later)
        learning_rate=1e-6,
        kl_coef=0.001,
        compile_model=False,
    )
    
    # Use tokenizer from trainer for consistency
    tokenizer = trainer.tokenizer
    
    print("Loading dataset: google/IFEval")
    dataset = IFEvalDataset(tokenizer)
    
    print("Initializing vLLM sampling engine...")
    sampler = SamplingClient(
        model_name,
        gpu_memory_utilization=0.35,  # Leave room for training
        enable_prefix_caching=True,
        dtype=torch.bfloat16,
        max_model_len=1024 + max_tokens,  # prompt + response
    )
    
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.95,
    )
    
    print(f"\nStarting training for {max_steps} steps...")
    pbar = tqdm(total=max_steps, desc="Training")
    
    for step in range(max_steps):
        # ─── Sampling Phase ───────────────────────────────────────────
        # Wake up vLLM for inference
        sampler.wake_up()
        
        # Get batch
        batch = dataset.get_batch(batch_size)
        prompts = dataset.format_prompts(batch)
        
        # Sample completions
        results = sampler.sample_batch(
            [ModelInput.from_ints(tokenizer(p)["input_ids"]) for p in prompts],
            sampling_params,
            num_samples=num_samples,
        )
        
        # Put vLLM to sleep to free GPU memory for training
        sampler.sleep()
        
        # ─── Reward Computation ───────────────────────────────────────
        rewards = dataset.compute_rewards(batch, results)
        avg_reward = sum(r for pr in rewards for r in pr) / (batch_size * num_samples)
        
        # Prepare batch for training
        input_ids, prompt_lens, sampling_logprobs = prepare_rl_batch(
            prompts, results, tokenizer
        )
        
        # ─── Training Phase ───────────────────────────────────────────
        # Forward-backward (on-policy: weights match sampling model)
        loss_out = trainer.forward_backward(
            input_ids=input_ids,
            prompt_lens=prompt_lens,
            sampling_logprobs=sampling_logprobs,
            rewards=flatten_rewards(rewards),
            rl_loss_fn="grpo",
        )
        
        # Optimizer step
        trainer.optim_step()
        
        # ─── Weight Sync ──────────────────────────────────────────────
        # Wake up vLLM and sync updated weights for next iteration
        sampler.wake_up()
        sampler.load_weights(trainer.get_state_dict())
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()
        
        pbar.update(1)
        pbar.set_postfix({
            "loss": f"{loss_out.loss:.4f}",
            "kl": f"{loss_out.kl:.4f}",
            "reward": f"{avg_reward:.2f}",
        })
        
        # Save checkpoint every 50 steps
        if (step + 1) % 50 == 0:
            trainer.save_state(f"checkpoints/ifeval_step_{step + 1}")
            print(f"\nSaved checkpoint at step {step + 1}")
    
    pbar.close()
    trainer.save_state("checkpoints/ifeval_final")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
