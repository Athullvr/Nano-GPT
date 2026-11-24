"""
Optimizer and Learning Rate Scheduler configuration for TouchlineLM.

Implements:
- Weight decay parameter grouping for AdamW:
  - 2D parameters (weights of Linear layers) receive weight decay.
  - 1D parameters (biases, LayerNorm scale and shift) receive 0 weight decay.
  - Tied weights are properly referenced once.
- Cosine Annealing with Linear Warmup Scheduler:
  - Linear warmup from 0 to learning_rate over warmup_steps.
  - Cosine decay from learning_rate to min_lr over (max_steps - warmup_steps).
  - Constant min_lr for steps beyond max_steps.
"""

import math
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW


def configure_optimizers(
    model: nn.Module,
    weight_decay: float = 0.1,
    learning_rate: float = 6e-4,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> AdamW:
    """
    Configures AdamW with decoupled weight decay for 2D vs 1D parameters.

    Args:
        model: Transformer model instance.
        weight_decay: Weight decay factor for matrix weights.
        learning_rate: Peak learning rate.
        betas: Adam beta coefficients.
        eps: Epsilon for numerical stability.

    Returns:
        Configured AdamW optimizer.
    """
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []
    seen_params = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Avoid duplicate entries for tied weights
        if id(param) in seen_params:
            continue
        seen_params.add(id(param))

        # Weight matrices of 2D or higher (Linear, Embedding) receive weight decay
        # 1D tensors (biases, LayerNorm scale/shift) receive no weight decay
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps)
    return optimizer


class CosineWarmupScheduler:
    """
    Cosine learning rate decay scheduler with linear warmup.

    Parameters:
        optimizer: PyTorch optimizer.
        warmup_steps: Number of initial steps with linear warmup.
        max_steps: Total number of training steps.
        min_lr: Minimum learning rate floor after decay.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 200,
        max_steps: int = 3000,
        min_lr: float = 6e-5,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.current_step = 0

        # Store initial learning rate for each param group
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def get_lr(self, step: int, base_lr: float) -> float:
        """Calculates learning rate for a specific step."""
        if step < self.warmup_steps:
            # Linear warmup
            return base_lr * (step + 1) / max(1, self.warmup_steps)
        if step > self.max_steps:
            # Floor
            return self.min_lr

        # Cosine decay down to min_lr
        decay_ratio = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (base_lr - self.min_lr)

    def step(self) -> List[float]:
        """Advance scheduler by one step and update optimizer learning rates."""
        lrs = []
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            lr = self.get_lr(self.current_step, base_lr)
            group["lr"] = lr
            lrs.append(lr)
        self.current_step += 1
        return lrs

    def state_dict(self) -> Dict[str, Any]:
        """Serialize scheduler state."""
        return {
            "current_step": self.current_step,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "min_lr": self.min_lr,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore scheduler state."""
        self.current_step = state["current_step"]
        self.warmup_steps = state["warmup_steps"]
        self.max_steps = state["max_steps"]
        self.min_lr = state["min_lr"]
        self.base_lrs = state["base_lrs"]
        # Sync current lr to optimizer
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self.get_lr(self.current_step, base_lr)
