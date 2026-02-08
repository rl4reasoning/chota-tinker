"""Core types for chota-tinker sampling client."""

from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    """Parameters for sampling from the model."""
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 means disabled
    stop: list[str] | None = None


@dataclass
class ModelInput:
    """Wrapper for model input tokens."""
    token_ids: list[int]

    @classmethod
    def from_ints(cls, ids: list[int]) -> "ModelInput":
        return cls(token_ids=ids)


@dataclass
class Sequence:
    """A single generated sequence."""
    tokens: list[int]
    text: str
    finish_reason: str | None = None  # e.g. "length", "stop" (from backend)


@dataclass
class SamplingResult:
    """Result of sampling from the model."""
    sequences: list[Sequence] = field(default_factory=list)


@dataclass
class LossOutput:
    """Output from forward_backward."""
    loss: float
    num_tokens: int
    # RL-specific (optional)
    policy_loss: float | None = None
    kl: float | None = None

