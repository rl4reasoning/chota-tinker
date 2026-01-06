"""RL utilities for preparing training batches."""

import torch
from transformers import PreTrainedTokenizer

from .types import SamplingResult


def prepare_rl_batch(
    prompts: list[str],
    results: list[SamplingResult],
    tokenizer: PreTrainedTokenizer,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare a batch for RL training from sampling results.

    Takes prompts and their sampled completions, concatenates them,
    and returns tensors ready for training.

    Args:
        prompts: List of prompt strings
        results: List of SamplingResult, one per prompt
        tokenizer: Tokenizer for encoding prompts
        device: Target device

    Returns:
        input_ids: (batch, seq_len) padded token IDs
        prompt_lens: (batch,) length of each prompt
        sampling_logprobs: (batch, seq_len) log probs from sampling
                          (zeros for prompt tokens, will be masked anyway)

    Note:
        - Flattens num_samples into batch dimension
        - You need to compute rewards separately and pass to forward_backward
    """
    all_input_ids = []
    all_prompt_lens = []

    for prompt, result in zip(prompts, results):
        # Tokenize prompt
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        for seq in result.sequences:
            # Concatenate prompt + completion
            full_ids = prompt_ids + seq.tokens
            all_input_ids.append(full_ids)
            all_prompt_lens.append(prompt_len)

    # Pad to same length
    max_len = max(len(ids) for ids in all_input_ids)
    pad_id = tokenizer.pad_token_id or 0

    padded_ids = []
    for ids in all_input_ids:
        padded = ids + [pad_id] * (max_len - len(ids))
        padded_ids.append(padded)

    input_ids = torch.tensor(padded_ids, device=device)
    prompt_lens = torch.tensor(all_prompt_lens, device=device)

    # Placeholder for sampling logprobs (zeros - masked during loss)
    # In practice, vLLM can return logprobs if needed
    sampling_logprobs = torch.zeros(
        input_ids.size(0), input_ids.size(1) - 1, device=device
    )

    return input_ids, prompt_lens, sampling_logprobs


def flatten_rewards(
    rewards: list[list[float]],
    device: str = "cuda",
) -> torch.Tensor:
    """
    Flatten rewards from [num_prompts, num_samples] to [batch].

    Args:
        rewards: Nested list, rewards[i][j] = reward for prompt i, sample j

    Returns:
        (batch,) tensor of rewards
    """
    flat = [r for prompt_rewards in rewards for r in prompt_rewards]
    return torch.tensor(flat, device=device)

