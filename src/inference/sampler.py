"""
Sampling strategy implementations and analysis tools.

This module provides:
  - Greedy decoding
  - Temperature sampling
  - Top-k filtering
  - Top-p (nucleus) sampling
  - Min-p sampling
  - Mirostat v2 (adaptive perplexity control)
  - Beam search (reference implementation)

These are primarily educational/analytical — the actual engines delegate
sampling to llama-cpp and HuggingFace Transformers. This module is useful
for understanding and visualising the effect of different sampling parameters
without running a full model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core sampling functions (operate on logit tensors)
# ---------------------------------------------------------------------------

def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Scale logits by temperature before softmax.

    Temperature < 1 → sharper distribution (more deterministic)
    Temperature > 1 → flatter distribution (more random)
    Temperature = 0 → greedy (argmax), handled as a special case elsewhere
    """
    if temperature <= 0:
        raise ValueError("Temperature must be > 0 (use greedy decoding for deterministic output)")
    return logits / temperature


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Zero out all logits except the top-k highest values.

    Prevents sampling from the long tail of low-probability tokens.
    """
    if k <= 0:
        return logits
    top_k_values = torch.topk(logits, min(k, logits.size(-1))).values
    threshold = top_k_values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling: zero out tokens outside the smallest set whose
    cumulative probability ≥ p.

    Unlike top-k, the number of retained tokens adapts to the distribution:
    - Peaked distributions → few tokens pass the threshold
    - Flat distributions → many tokens pass the threshold
    """
    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens where cumulative probability exceeds p
    # Shift right so the token that pushes us over p is kept
    sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits[sorted_indices_to_remove] = float("-inf")

    # Restore original ordering
    logits_filtered = torch.zeros_like(logits)
    logits_filtered.scatter_(-1, sorted_indices, sorted_logits)
    return logits_filtered


def apply_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """
    Min-p sampling (Nguyen et al., 2024).

    Removes tokens with probability < min_p × max_probability.

    Compared to top-p, min-p scales dynamically with the confidence of the
    most likely token: when the model is very confident (high max prob), the
    threshold is high and only near-top tokens survive. When uncertain, more
    tokens survive. This produces more coherent outputs than top-p at the
    same effective diversity level.
    """
    if min_p <= 0:
        return logits

    probs = F.softmax(logits, dim=-1)
    max_prob = probs.max(dim=-1, keepdim=True).values
    threshold = min_p * max_prob
    return logits.masked_fill(probs < threshold, float("-inf"))


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float = 1.0,
) -> torch.Tensor:
    """
    Penalise tokens that already appear in input_ids.

    For already-seen tokens:
      - positive logits are divided by penalty
      - negative logits are multiplied by penalty
    This preserves sign while reducing the probability of repetitions.
    """
    if penalty == 1.0:
        return logits

    score = torch.gather(logits, -1, input_ids)
    score = torch.where(score < 0, score * penalty, score / penalty)
    return logits.scatter_(-1, input_ids, score)


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """Return the argmax token (batch-compatible)."""
    return logits.argmax(dim=-1)


def sample_token(logits: torch.Tensor) -> torch.Tensor:
    """Sample from the categorical distribution defined by logits."""
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ---------------------------------------------------------------------------
# Mirostat v2 (adaptive perplexity control)
# ---------------------------------------------------------------------------

@dataclass
class MirostatState:
    """Mutable state for the Mirostat v2 sampler."""
    mu: float           # running estimate of surprise level

    @classmethod
    def init(cls, target_surprise: float = 3.0) -> "MirostatState":
        return cls(mu=target_surprise * 2)


def apply_mirostat_v2(
    logits: torch.Tensor,
    state: MirostatState,
    tau: float = 3.0,
    eta: float = 0.1,
) -> tuple[torch.Tensor, MirostatState]:
    """
    Mirostat v2 sampling (Basu et al., 2020).

    Adaptively adjusts the truncation threshold to maintain a target
    per-token surprise level (tau), measured in nats.

    Unlike top-p/top-k which use fixed thresholds, Mirostat tracks the
    observed surprise from previous tokens and adjusts dynamically — this
    keeps text quality consistent across domains (code, prose, lists) that
    have very different natural entropy levels.

    Parameters
    ----------
    tau : float
        Target surprise per token in nats. tau ≈ 3 is a good default.
        Lower → more conservative; higher → more creative.
    eta : float
        Learning rate for surprise estimate update.

    Returns
    -------
    (filtered_logits, updated_state)
    """
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)

    # Compute surprise at each cutoff point
    log_probs = torch.log(sorted_probs)
    surprises = -log_probs

    # Keep tokens until expected surprise exceeds mu
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    k = int((cumsum_probs < (1 - math.exp(-state.mu))).sum().item()) + 1
    k = max(1, min(k, logits.size(-1)))

    # Truncate
    top_k_idx = sorted_idx[:k]
    filtered = torch.full_like(logits, float("-inf"))
    filtered[top_k_idx] = logits[top_k_idx]

    # Sample and update surprise estimate
    sampled = sample_token(filtered)
    observed_surprise = -torch.log(probs[sampled]).item()
    new_mu = state.mu - eta * (observed_surprise - tau)

    return filtered, MirostatState(mu=new_mu)


# ---------------------------------------------------------------------------
# Sampling pipeline builder
# ---------------------------------------------------------------------------

def build_sampling_pipeline(
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Compose a sampling pipeline as a single callable.

    The application order follows best practices:
      1. Repetition penalty (on raw logits)
      2. Temperature scaling
      3. Top-k filtering
      4. Min-p filtering
      5. Top-p (nucleus) filtering
      6. Final sample

    Parameters and returns
    ----------------------
    Returns a function (logits, input_ids) → sampled_token_id.
    """
    def pipeline(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        if repetition_penalty != 1.0:
            logits = apply_repetition_penalty(logits, input_ids, repetition_penalty)
        if temperature > 0:
            logits = apply_temperature(logits, temperature)
        else:
            return greedy_sample(logits)
        if top_k > 0:
            logits = apply_top_k(logits, top_k)
        if min_p > 0:
            logits = apply_min_p(logits, min_p)
        if top_p < 1.0:
            logits = apply_top_p(logits, top_p)
        return sample_token(logits)

    return pipeline


# ---------------------------------------------------------------------------
# Entropy / sampling analysis
# ---------------------------------------------------------------------------

def token_entropy(logits: torch.Tensor) -> float:
    """Compute the entropy of the next-token distribution in nats."""
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return -(probs * log_probs).sum().item()


def effective_vocab_size(logits: torch.Tensor, p: float = 0.95) -> int:
    """
    Number of tokens that together account for p of the probability mass.

    A proxy for how 'certain' the model is: low values → model is confident;
    high values → model is uncertain or the distribution is flat.
    """
    probs = F.softmax(logits.float(), dim=-1)
    sorted_probs, _ = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    return int((cumsum < p).sum().item()) + 1
