"""RL loss functions for fine-tuning.

Inspired from:
https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-tx

"""

import torch
import torch.nn.functional as F

# TODO: @claude add memory efficient version ?
def compute_logprobs(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """
    Compute log probabilities for specific tokens.
    
    Args:
        logits: (batch, seq, vocab) model output logits
        tokens: (batch, seq) token IDs
    
    Returns:
        (batch, seq) log probabilities for each token
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)


def create_completion_mask(
    seq_len: int,
    prompt_lens: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Create mask that's 1 for completion tokens, 0 for prompt.
    
    Args:
        seq_len: total sequence length
        prompt_lens: (batch,) length of each prompt
        device: torch device
    
    Returns:
        (batch, seq_len) mask
    """
    batch_size = prompt_lens.shape[0]
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    return (positions >= prompt_lens.unsqueeze(1)).float()


def ppo_loss(
    target_logprobs: torch.Tensor,
    sampling_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_range: float = 0.2,
) -> torch.Tensor:
    """
    PPO clipped policy gradient loss.
    
    Args:
        target_logprobs: log probs from current policy
        sampling_logprobs: log probs from sampling policy
        advantages: per-token advantages
        mask: completion mask (1 for completion, 0 for prompt)
        clip_range: PPO clip range (default 0.2)
    
    Returns:
        scalar loss
    """
    ratio = torch.exp(target_logprobs - sampling_logprobs)
    clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    
    loss_unclipped = -ratio * advantages
    loss_clipped = -clipped_ratio * advantages
    loss = torch.max(loss_unclipped, loss_clipped)
    
    # Apply mask and reduce
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def grpo_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute GRPO (Group Relative Policy Optimization) advantages.
    Normalizes rewards within the group (batch).
    
    Args:
        rewards: (batch,) rewards for each sample
        eps: epsilon for numerical stability
    
    Returns:
        (batch,) normalized advantages
    """
    return (rewards - rewards.mean()) / (rewards.std() + eps)


def grpo_loss(
    target_logprobs: torch.Tensor,
    sampling_logprobs: torch.Tensor,
    rewards: torch.Tensor,
    mask: torch.Tensor,
    clip_range: float = 0.2,
) -> torch.Tensor:
    """
    GRPO loss - PPO with group-normalized advantages.
    
    Args:
        target_logprobs: (batch, seq) log probs from current policy
        sampling_logprobs: (batch, seq) log probs from sampling policy
        rewards: (batch,) reward for each sample
        mask: (batch, seq) completion mask
        clip_range: PPO clip range
    
    Returns:
        scalar loss
    """
    # Compute group-relative advantages and broadcast to sequence
    advantages = grpo_advantages(rewards).unsqueeze(-1).expand_as(target_logprobs)
    return ppo_loss(target_logprobs, sampling_logprobs, advantages, mask, clip_range)


# TODO: @claude add KL safe versions
def kl_penalty(
    current_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    KL divergence penalty between current and reference policy.
    
    Args:
        current_logprobs: log probs from current policy
        reference_logprobs: log probs from reference (frozen) policy
        mask: completion mask
    
    Returns:
        scalar KL penalty
    """
    kl = current_logprobs - reference_logprobs
    return (kl * mask).sum() / mask.sum().clamp(min=1)


# TODO: @claude add fused versions
def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Standard cross-entropy loss with optional masking.
    
    Args:
        logits: (batch, seq, vocab) model logits
        labels: (batch, seq) target token IDs
        mask: optional (batch, seq) mask
    
    Returns:
        scalar loss
    """
    # Shift for causal LM
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='none',
    ).view(shift_labels.shape)
    
    if mask is not None:
        shift_mask = mask[..., 1:]
        return (loss * shift_mask).sum() / shift_mask.sum().clamp(min=1)
    return loss.mean()

