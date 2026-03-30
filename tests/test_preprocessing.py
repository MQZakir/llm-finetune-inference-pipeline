"""Tests for text preprocessing and quality filtering."""

from __future__ import annotations

import pytest

from src.data.preprocessing import (
    QualityFilterConfig,
    PreprocessingConfig,
    TextPreprocessor,
    filter_chat_quality,
    _fix_encoding,
    _strip_html,
)


@pytest.fixture
def preprocessor():
    return TextPreprocessor(PreprocessingConfig())


class TestQualityFilter:
    def test_passes_normal_text(self, preprocessor):
        text = "This is a perfectly normal sentence with multiple words and some punctuation."
        assert preprocessor._passes_quality(text) is True

    def test_rejects_too_short(self, preprocessor):
        assert preprocessor._passes_quality("Hi") is False

    def test_rejects_empty(self, preprocessor):
        assert preprocessor._passes_quality("") is False

    def test_rejects_repetitive(self, preprocessor):
        # All same word → low unique ratio
        text = "word " * 100
        assert preprocessor._passes_quality(text) is False

    def test_rejects_all_caps(self, preprocessor):
        text = "THIS IS ALL CAPS AND SHOULD BE REJECTED BECAUSE IT LOOKS LIKE SPAM " * 3
        assert preprocessor._passes_quality(text) is False

    def test_rejects_numeric_heavy(self, preprocessor):
        text = "1234567890 " * 50
        assert preprocessor._passes_quality(text) is False

    def test_rejects_line_repetition(self, preprocessor):
        line = "The same line repeated over and over again.\n"
        text = line * 20
        assert preprocessor._passes_quality(text) is False

    def test_custom_reject_pattern(self):
        cfg = PreprocessingConfig(
            quality=QualityFilterConfig(reject_patterns=[r"SPAM_MARKER"])
        )
        pp = TextPreprocessor(cfg)
        text = "Normal text but it contains SPAM_MARKER somewhere."
        assert pp._passes_quality(text) is False


class TestNormalisation:
    def test_collapses_whitespace(self, preprocessor):
        text = "Hello   world    with   spaces"
        result = preprocessor._normalise(text)
        assert "   " not in result

    def test_collapses_blank_lines(self, preprocessor):
        text = "Para one\n\n\n\n\nPara two"
        result = preprocessor._normalise(text)
        assert "\n\n\n" not in result

    def test_strips_html_tags(self, preprocessor):
        text = "<p>Hello <b>world</b></p>"
        result = preprocessor._normalise(text)
        assert "<p>" not in result
        assert "<b>" not in result
        assert "Hello" in result

    def test_fixes_encoding_artifacts(self, preprocessor):
        text = "It\u2019s a test"  # smart apostrophe — should survive NFKC
        result = preprocessor._normalise(text)
        assert result  # should not be empty


class TestHTMLStripping:
    def test_removes_tags(self):
        assert "Hello" in _strip_html("<p>Hello</p>")
        assert "<p>" not in _strip_html("<p>Hello</p>")

    def test_replaces_entities(self):
        assert "&amp;" not in _strip_html("Hello &amp; world")
        assert "&" in _strip_html("Hello &amp; world")


class TestEncodingFix:
    def test_fixes_curly_quotes(self):
        result = _fix_encoding("â€™")
        assert result == "'"


class TestDeduplication:
    def test_removes_exact_duplicates(self):
        from datasets import Dataset
        records = [{"text": f"Sentence number {i}"} for i in range(5)]
        records += records[:2]  # add 2 duplicates
        ds = Dataset.from_list(records)

        pp = TextPreprocessor(PreprocessingConfig(deduplicate=True))
        deduped = pp._exact_dedup(ds, "text")
        assert len(deduped) == 5


class TestChatQualityFilter:
    def test_valid_conversation(self):
        msgs = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a high-level programming language known for its readability."},
        ]
        assert filter_chat_quality(msgs) is True

    def test_rejects_empty(self):
        assert filter_chat_quality([]) is False

    def test_rejects_no_assistant_turn(self):
        msgs = [{"role": "user", "content": "Hello"}]
        assert filter_chat_quality(msgs) is False

    def test_rejects_last_turn_not_assistant(self):
        msgs = [
            {"role": "user",      "content": "How are you?"},
            {"role": "assistant", "content": "I am doing well, thank you!"},
            {"role": "user",      "content": "Great!"},
        ]
        assert filter_chat_quality(msgs) is False

    def test_rejects_short_assistant_response(self):
        msgs = [
            {"role": "user",      "content": "Explain quantum mechanics"},
            {"role": "assistant", "content": "ok"},  # too short
        ]
        assert filter_chat_quality(msgs, min_assistant_chars=20) is False
