"""Tests for dataset abstractions."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.data.dataset import (
    IGNORE_INDEX,
    DatasetConfig,
    DatasetFormat,
    FineTuneDataset,
    PackedDataset,
    TokenisedSample,
    _handle_chat,
    _handle_completion,
    _handle_instruct,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.eos_token = "</s>"
    tok.eos_token_id = 2

    def tokenize(text, **kwargs):
        # Deterministic fake tokenisation: one token per character, IDs = ord values % 100
        ids = [ord(c) % 100 + 3 for c in text[:kwargs.get("max_length", 512)]]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    tok.side_effect = tokenize
    tok.__call__ = lambda self, text, **kwargs: tokenize(text, **kwargs)

    def apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"<|{role}|>{content}<|end|>")
        return "".join(parts)

    tok.apply_chat_template = apply_chat_template
    return tok


@pytest.fixture
def completion_config():
    return DatasetConfig(
        name_or_path="dummy",
        format=DatasetFormat.COMPLETION,
        text_column="text",
        min_length=5,
    )


@pytest.fixture
def instruct_config():
    return DatasetConfig(
        name_or_path="dummy",
        format=DatasetFormat.INSTRUCT,
        instruction_column="instruction",
        response_column="response",
        min_length=5,
    )


# ---------------------------------------------------------------------------
# Completion format
# ---------------------------------------------------------------------------

class TestCompletionFormat:
    def test_full_sequence_loss(self, mock_tokenizer, completion_config):
        """Labels should match input_ids for completion format."""
        record = {"text": "Hello world"}
        sample = _handle_completion(record, completion_config, mock_tokenizer, 512)
        assert sample.input_ids == sample.labels
        assert len(sample.input_ids) == len(sample.attention_mask)

    def test_truncation(self, mock_tokenizer, completion_config):
        """Sequences should be truncated to max_seq_length."""
        long_text = "a" * 1000
        sample = _handle_completion(record={"text": long_text}, config=completion_config,
                                    tokenizer=mock_tokenizer, max_seq_length=100)
        assert len(sample.input_ids) <= 100


# ---------------------------------------------------------------------------
# Instruct format
# ---------------------------------------------------------------------------

class TestInstructFormat:
    def test_prompt_tokens_masked(self, mock_tokenizer, instruct_config):
        """Prompt tokens should have IGNORE_INDEX in labels."""
        record = {"instruction": "What is 2+2?", "response": "4"}
        sample = _handle_instruct(record, instruct_config, mock_tokenizer, 512)

        # At least some labels should be IGNORE_INDEX (prompt mask)
        assert IGNORE_INDEX in sample.labels

        # Non-masked labels should equal the corresponding input_ids
        for idx, (iid, label) in enumerate(zip(sample.input_ids, sample.labels)):
            if label != IGNORE_INDEX:
                assert label == iid, f"Token mismatch at position {idx}"

    def test_lengths_consistent(self, mock_tokenizer, instruct_config):
        record = {"instruction": "Describe AI", "response": "AI is amazing."}
        sample = _handle_instruct(record, instruct_config, mock_tokenizer, 512)
        assert len(sample.input_ids) == len(sample.labels) == len(sample.attention_mask)


# ---------------------------------------------------------------------------
# FineTuneDataset
# ---------------------------------------------------------------------------

class TestFineTuneDataset:
    def _make_samples(self, n: int = 10) -> list[TokenisedSample]:
        return [
            TokenisedSample(
                input_ids=list(range(20 + i)),
                attention_mask=[1] * (20 + i),
                labels=list(range(20 + i)),
            )
            for i in range(n)
        ]

    def test_len(self, completion_config):
        samples = self._make_samples(5)
        ds = FineTuneDataset(samples, completion_config)
        assert len(ds) == 5

    def test_getitem_returns_tensors(self, completion_config):
        import torch
        samples = self._make_samples(3)
        ds = FineTuneDataset(samples, completion_config)
        item = ds[0]
        assert "input_ids" in item
        assert "labels" in item
        assert "attention_mask" in item
        assert isinstance(item["input_ids"], torch.Tensor)

    def test_token_stats(self, completion_config):
        samples = self._make_samples(10)
        ds = FineTuneDataset(samples, completion_config)
        stats = ds.token_stats()
        assert "count" in stats
        assert stats["count"] == 10
        assert stats["min"] <= stats["mean"] <= stats["max"]

    def test_fingerprint_deterministic(self, completion_config):
        samples = self._make_samples(5)
        ds = FineTuneDataset(samples, completion_config)
        assert ds.fingerprint() == ds.fingerprint()

    def test_fingerprint_changes_with_data(self, completion_config):
        samples_a = self._make_samples(5)
        samples_b = self._make_samples(5)
        samples_b[0] = TokenisedSample([99, 88, 77], [1, 1, 1], [99, 88, 77])
        ds_a = FineTuneDataset(samples_a, completion_config)
        ds_b = FineTuneDataset(samples_b, completion_config)
        assert ds_a.fingerprint() != ds_b.fingerprint()


# ---------------------------------------------------------------------------
# PackedDataset
# ---------------------------------------------------------------------------

class TestPackedDataset:
    def test_chunk_size(self, completion_config):
        samples = [
            TokenisedSample(
                input_ids=list(range(100)),
                attention_mask=[1] * 100,
                labels=list(range(100)),
            )
            for _ in range(20)
        ]
        base = FineTuneDataset(samples, completion_config)
        packed = PackedDataset(base, chunk_size=256, eos_token_id=2)

        for i in range(len(packed)):
            item = packed[i]
            assert len(item["input_ids"]) == 256

    def test_attention_mask_all_ones(self, completion_config):
        import torch
        samples = [
            TokenisedSample(list(range(50)), [1] * 50, list(range(50)))
            for _ in range(10)
        ]
        base = FineTuneDataset(samples, completion_config)
        packed = PackedDataset(base, chunk_size=128, eos_token_id=2)

        if len(packed) > 0:
            item = packed[0]
            assert item["attention_mask"].sum() == len(item["attention_mask"])
