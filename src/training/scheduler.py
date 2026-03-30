"""
Learning rate schedulers.

PyTorch and HuggingFace cover most cases natively, but this module provides
a few additions that are useful for fine-tuning:

  - WarmupCosineScheduler   : cosine decay with linear warmup (standard recipe)
  - REXScheduler            : REX (Reflection Exponential) — more stable for
                              small datasets / short fine-tuning runs
  - ConstantWithCooldown    : constant LR with a short cosine cooldown at the end
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    Linear warmup, then cosine annealing.

    The minimum LR is  min_lr_ratio × peak_lr.  Setting min_lr_ratio=0.1
    avoids vanishing gradients near the end of training.

    Parameters
    ----------
    num_cycles : float
        Number of cosine half-periods. 0.5 (default) → single trough.
    min_lr_ratio : float
        LR never drops below  min_lr_ratio × base_lr.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        return max(min_lr_ratio, cosine)

    return LambdaLR(optimizer, lr_lambda)


def get_rex_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    a: float = 1.0,
    b: float = 0.6,
) -> LambdaLR:
    """
    REX (Reflection Exponential) learning rate schedule.

    Proposed in "REX: Revisiting Budgeted Training with an Improved Schedule"
    (Budhraja et al., 2023). Outperforms cosine on budget-limited training.

    The schedule is:  lr(t) = a × (1 - t/T)^b  after warmup.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return a * max(0.0, 1.0 - progress) ** b

    return LambdaLR(optimizer, lr_lambda)


def get_constant_with_cooldown(
    optimizer: Optimizer,
    num_training_steps: int,
    cooldown_fraction: float = 0.1,
) -> LambdaLR:
    """
    Constant LR for (1 - cooldown_fraction) of training, then cosine decay to 0.

    Useful when you want to keep the LR high for exploration but avoid
    a sharp ending that can destabilise the final checkpoint.
    """
    cooldown_start = int(num_training_steps * (1.0 - cooldown_fraction))

    def lr_lambda(current_step: int) -> float:
        if current_step < cooldown_start:
            return 1.0
        progress = float(current_step - cooldown_start) / float(
            max(1, num_training_steps - cooldown_start)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Scheduler registry — maps config names to factory functions
# ---------------------------------------------------------------------------

SCHEDULER_REGISTRY: dict[str, Callable] = {
    "cosine":               get_cosine_schedule_with_warmup,
    "cosine_with_warmup":   get_cosine_schedule_with_warmup,
    "rex":                  get_rex_schedule,
    "constant_with_cooldown": get_constant_with_cooldown,
}


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    **kwargs,
) -> LambdaLR:
    """
    Build a scheduler by name.

    Falls back to HuggingFace's ``get_scheduler`` for names not in this
    registry (e.g. 'linear', 'polynomial').
    """
    if name in SCHEDULER_REGISTRY:
        return SCHEDULER_REGISTRY[name](
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **kwargs,
        )

    try:
        from transformers import get_scheduler as hf_get_scheduler
        return hf_get_scheduler(
            name,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    except Exception as e:
        raise ValueError(
            f"Unknown scheduler '{name}'. Available: {list(SCHEDULER_REGISTRY.keys())}"
        ) from e
