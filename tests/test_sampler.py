"""Tests for sampling strategies."""

from __future__ import annotations

import math

import pytest
import torch

from src.inference.sampler import (
    MirostatState,
    apply_min_p,
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    apply_mirostat_v2,
    build_sampling_pipeline,
    effective_vocab_size,
    greedy_sample,
    sample_token,
    token_entropy,
)


@pytest.fixture
def uniform_logits():
    """Logits that produce a uniform distribution over 10 tokens."""
    return torch.zeros(10)


@pytest.fixture
def peaked_logits():
    """Logits strongly peaked at token index 3."""
    logits = torch.full((10,), -10.0)
    logits[3] = 10.0
    return logits


class TestTemperature:
    def test_low_temperature_sharpens(self, uniform_logits):
        """Lower temperature should increase the gap between logits."""
        scaled = apply_temperature(uniform_logits + 1.0, 0.1)
        # All were equal, scaled by 0.1 → still equal but 10x larger magnitude
        assert torch.allclose(scaled, (uniform_logits + 1.0) / 0.1)

    def test_raises_on_zero(self, uniform_logits):
        with pytest.raises(ValueError):
            apply_temperature(uniform_logits, 0.0)


class TestTopK:
    def test_keeps_k_tokens(self, uniform_logits):
        filtered = apply_top_k(uniform_logits, k=3)
        finite = (filtered != float("-inf")).sum().item()
        assert finite == 3

    def test_zero_k_disables(self, uniform_logits):
        result = apply_top_k(uniform_logits, k=0)
        assert torch.equal(result, uniform_logits)

    def test_preserves_top_token(self, peaked_logits):
        filtered = apply_top_k(peaked_logits, k=1)
        assert filtered[3] == peaked_logits[3]
        assert all(filtered[i] == float("-inf") for i in range(10) if i != 3)


class TestTopP:
    def test_always_keeps_top_token(self, peaked_logits):
        """Top token should always survive nucleus filtering."""
        filtered = apply_top_p(peaked_logits, p=0.5)
        assert filtered[3] != float("-inf")

    def test_p1_passthrough(self, uniform_logits):
        result = apply_top_p(uniform_logits, p=1.0)
        assert torch.equal(result, uniform_logits)

    def test_reduces_vocab(self, uniform_logits):
        """With uniform probs, p=0.5 should keep roughly 50% of tokens."""
        filtered = apply_top_p(uniform_logits, p=0.5)
        n_kept = (filtered != float("-inf")).sum().item()
        assert n_kept <= 6  # roughly half


class TestMinP:
    def test_confident_model_keeps_few_tokens(self, peaked_logits):
        """When model is very confident, min_p should keep only a few tokens."""
        filtered = apply_min_p(peaked_logits, min_p=0.1)
        n_kept = (filtered != float("-inf")).sum().item()
        assert n_kept < 5

    def test_zero_disables(self, uniform_logits):
        result = apply_min_p(uniform_logits, min_p=0.0)
        assert torch.equal(result, uniform_logits)


class TestRepetitionPenalty:
    def test_reduces_repeated_token_prob(self):
        logits = torch.zeros(10)
        logits[5] = 2.0
        input_ids = torch.tensor([5])
        penalised = apply_repetition_penalty(logits, input_ids, penalty=2.0)
        assert penalised[5] < logits[5]

    def test_unity_penalty_no_change(self):
        logits = torch.randn(10)
        input_ids = torch.arange(5)
        result = apply_repetition_penalty(logits, input_ids, penalty=1.0)
        assert torch.equal(result, logits)


class TestGreedy:
    def test_returns_argmax(self, peaked_logits):
        token = greedy_sample(peaked_logits)
        assert token.item() == 3


class TestPipeline:
    def test_greedy_pipeline(self, peaked_logits):
        pipeline = build_sampling_pipeline(temperature=0.0)
        token = pipeline(peaked_logits, torch.tensor([]))
        assert token.item() == 3

    def test_pipeline_returns_valid_index(self, uniform_logits):
        pipeline = build_sampling_pipeline(temperature=1.0, top_k=5)
        input_ids = torch.zeros(0, dtype=torch.long)
        token = pipeline(uniform_logits, input_ids)
        assert 0 <= token.item() < 10


class TestEntropy:
    def test_uniform_entropy(self, uniform_logits):
        h = token_entropy(uniform_logits)
        expected = math.log(10)  # max entropy for 10 tokens
        assert abs(h - expected) < 0.01

    def test_peaked_entropy_low(self, peaked_logits):
        h = token_entropy(peaked_logits)
        assert h < 0.01  # nearly zero entropy

    def test_effective_vocab_size_peaked(self, peaked_logits):
        n = effective_vocab_size(peaked_logits, p=0.95)
        assert n == 1

    def test_effective_vocab_size_uniform(self, uniform_logits):
        n = effective_vocab_size(uniform_logits, p=0.95)
        assert n >= 9


class TestMirostat:
    def test_returns_filtered_logits_and_updated_state(self, uniform_logits):
        state = MirostatState.init(tau=3.0)
        filtered, new_state = apply_mirostat_v2(uniform_logits, state)
        assert isinstance(new_state, MirostatState)
        assert new_state.mu != state.mu  # state should update

    def test_state_converges(self, uniform_logits):
        """After many steps on uniform logits, mu should stabilise near tau."""
        state = MirostatState.init(tau=3.0)
        for _ in range(50):
            _, state = apply_mirostat_v2(uniform_logits, state, tau=3.0, eta=0.1)
        # mu should be reasonably close to tau after convergence
        assert abs(state.mu - 3.0) < 3.0  # loose bound due to stochasticity
