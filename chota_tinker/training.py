"""Training client for fine-tuning LLMs."""

import torch
from transformers import AutoModelForCausalLM
from typing import Callable, Literal

from .types import LossOutput
from .losses import (
    compute_logprobs,
    create_completion_mask,
    ppo_loss,
    grpo_loss,
    kl_penalty,
)


class TrainingClient:
    """Minimal training client for LLM fine-tuning."""

    def __init__(
        self,
        model_name: str,
        learning_rate: float = 1e-5,
        lora_rank: int | None = None,
        kl_coef: float = 0.0,  # 0 = no KL penalty
        torch_dtype: torch.dtype = torch.bfloat16,
        compile_model: bool = True,
        device: str = "cuda",
    ):
        self.device = device
        self.model_name = model_name
        self.lora_rank = lora_rank
        self.kl_coef = kl_coef
        self.torch_dtype = torch_dtype

        # Load model with flash attention
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch_dtype,
        ).to(device)

        # Apply LoRA if requested
        if lora_rank is not None:
            self._apply_lora(lora_rank)

        # Compile for speed
        if compile_model:
            self.model = torch.compile(self.model)

        # Reference model for KL penalty
        self.ref_model = None
        if kl_coef > 0:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation="flash_attention_2",
                torch_dtype=torch_dtype,
            ).to(device)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
        )

        self._accumulated_steps = 0

    def _apply_lora(self, rank: int):
        """Apply LoRA adapters using PEFT."""
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, config)

    def forward_backward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        loss_fn: Callable | None = None,
        # RL-specific args
        prompt_lens: torch.Tensor | None = None,
        sampling_logprobs: torch.Tensor | None = None,
        rewards: torch.Tensor | None = None,
        rl_loss_fn: Literal["grpo", "ppo"] | None = None,
    ) -> LossOutput:
        """
        Forward + backward pass.

        For supervised learning:
            forward_backward(input_ids, labels)

        For RL:
            forward_backward(input_ids, prompt_lens=..., sampling_logprobs=...,
                           rewards=..., rl_loss_fn="grpo")
        """
        self.model.train()

        # RL mode
        if rl_loss_fn is not None:
            return self._forward_backward_rl(
                input_ids, prompt_lens, sampling_logprobs, rewards, rl_loss_fn
            )

        # Supervised mode
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = loss_fn(outputs.logits, labels) if loss_fn else outputs.loss
        loss.backward()

        num_tokens = (labels != -100).sum().item()
        self._accumulated_steps += 1

        return LossOutput(loss=loss.item(), num_tokens=num_tokens)

    def _forward_backward_rl(
        self,
        input_ids: torch.Tensor,
        prompt_lens: torch.Tensor,
        sampling_logprobs: torch.Tensor,
        rewards: torch.Tensor,
        loss_fn: Literal["grpo", "ppo"],
    ) -> LossOutput:
        """RL forward-backward pass."""
        batch_size, seq_len = input_ids.shape

        # Forward pass
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits

        # Compute current policy logprobs (shifted for causal LM)
        current_logprobs = compute_logprobs(logits[:, :-1], input_ids[:, 1:])

        # Create completion mask (shifted)
        mask = create_completion_mask(seq_len - 1, prompt_lens - 1, input_ids.device)

        # Compute policy loss
        if loss_fn == "grpo":
            policy_loss = grpo_loss(
                current_logprobs, sampling_logprobs, rewards, mask
            )
        else:  # ppo
            # Expand rewards to sequence for PPO
            advantages = rewards.unsqueeze(-1).expand_as(current_logprobs)
            policy_loss = ppo_loss(
                current_logprobs, sampling_logprobs, advantages, mask
            )

        total_loss = policy_loss

        # KL penalty
        kl = 0.0
        if self.kl_coef > 0 and self.ref_model is not None:
            with torch.no_grad():
                ref_outputs = self.ref_model(input_ids=input_ids)
                ref_logprobs = compute_logprobs(ref_outputs.logits[:, :-1], input_ids[:, 1:])
            kl = kl_penalty(current_logprobs, ref_logprobs, mask)
            total_loss = total_loss + self.kl_coef * kl

        # Backward
        total_loss.backward()

        num_tokens = int(mask.sum().item())
        self._accumulated_steps += 1

        return LossOutput(
            loss=total_loss.item(),
            num_tokens=num_tokens,
            policy_loss=policy_loss.item(),
            kl=kl.item() if isinstance(kl, torch.Tensor) else kl,
        )

    def optim_step(self):
        """Update model weights and reset gradients."""
        self.optimizer.step()
        self.optimizer.zero_grad()
        self._accumulated_steps = 0

    def save_state(self, path: str):
        """Save model and optimizer state."""
        if self.lora_rank is not None:
            self.model.save_pretrained(path)
        else:
            torch.save({
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }, path)

    def load_state(self, path: str):
        """Load model and optimizer state."""
        if self.lora_rank is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, path)
        else:
            checkpoint = torch.load(path, weights_only=False)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
