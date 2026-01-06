"""chota-tinker: Minimal Tinker-like training API in PyTorch."""

from .types import ModelInput, SamplingParams, SamplingResult, Sequence, LossOutput
from .client import SamplingClient, ServerSamplingClient
from .training import TrainingClient
from .losses import (
    compute_logprobs,
    create_completion_mask,
    ppo_loss,
    grpo_loss,
    grpo_advantages,
    kl_penalty,
    cross_entropy_loss,
)
from .rl_utils import prepare_rl_batch, flatten_rewards
from .datasets import IFEvalDataset

__all__ = [
    # Sampling
    "SamplingClient",
    "ServerSamplingClient",
    "SamplingParams",
    "ModelInput",
    "SamplingResult",
    "Sequence",
    # Training
    "TrainingClient",
    "LossOutput",
    # RL Losses
    "compute_logprobs",
    "create_completion_mask",
    "ppo_loss",
    "grpo_loss",
    "grpo_advantages",
    "kl_penalty",
    "cross_entropy_loss",
    # RL Utils
    "prepare_rl_batch",
    "flatten_rewards",
    # Datasets
    "IFEvalDataset",
]

