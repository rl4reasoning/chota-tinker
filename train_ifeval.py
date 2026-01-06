"""
Minimal RL training script on IFEval dataset.

Start vLLM server first:
    vllm serve Qwen/Qwen3-4B --port 8000
"""

from transformers import AutoTokenizer
from tqdm import tqdm

from chota_tinker import (
    TrainingClient,
    ServerSamplingClient,
    SamplingParams,
    ModelInput,
    IFEvalDataset,
    prepare_rl_batch,
    flatten_rewards,
)


def main():
    # Config
    model_name = "Qwen/Qwen3-4B"
    batch_size = 4
    num_samples = 4
    max_steps = 100
    max_tokens = 256
    
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading dataset: google/IFEval")
    dataset = IFEvalDataset(tokenizer)
    
    print("Initializing training client...")
    trainer = TrainingClient(
        model_name=model_name,
        lora_rank=16,
        learning_rate=1e-5,
        kl_coef=0.01,
        compile_model=False,
    )
    
    print("Connecting to vLLM server...")
    sampler = ServerSamplingClient("http://localhost:8000")
    
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.95,
    )
    
    print(f"\nStarting training for {max_steps} steps...")
    pbar = tqdm(total=max_steps, desc="Training")
    
    for step in range(max_steps):
        # Get batch
        batch = dataset.get_batch(batch_size)
        prompts = dataset.format_prompts(batch)
        
        # Sample completions
        try:
            results = sampler.sample_batch(
                [ModelInput.from_ints(tokenizer(p)["input_ids"]) for p in prompts],
                sampling_params,
                num_samples=num_samples,
            )
        except Exception as e:
            print(f"\nSampling error: {e}")
            print(f"Make sure vLLM server is running: vllm serve {model_name} --port 8000")
            break
        
        # Compute rewards
        rewards = dataset.compute_rewards(batch, results)
        avg_reward = sum(r for pr in rewards for r in pr) / (batch_size * num_samples)
        
        # Prepare batch for training
        input_ids, prompt_lens, sampling_logprobs = prepare_rl_batch(
            prompts, results, tokenizer
        )
        
        # Forward-backward
        loss_out = trainer.forward_backward(
            input_ids=input_ids,
            prompt_lens=prompt_lens,
            sampling_logprobs=sampling_logprobs,
            rewards=flatten_rewards(rewards),
            rl_loss_fn="grpo",
        )
        
        # Optimizer step
        trainer.optim_step()
        
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
