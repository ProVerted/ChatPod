"""
Tests for the Phase 2 summarizer. All LLM calls are mocked — no API key or
network access needed to run these, and no cost incurred.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.summarizer import _chunk_text, summarize  # noqa: E402


def _make_fake_client(return_values: list[str]) -> MagicMock:
    """Build a fake OpenAI client whose chat.completions.create() returns
    the given strings in sequence, one per call."""
    client = MagicMock()
    responses = []
    for text in return_values:
        message = MagicMock()
        message.content = text
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        responses.append(response)
    client.chat.completions.create.side_effect = responses
    return client


def test_summarize_empty_text_returns_empty_string():
    client = _make_fake_client([])
    assert summarize("", client=client) == ""
    assert summarize("   ", client=client) == ""
    client.chat.completions.create.assert_not_called()


def test_summarize_short_text_makes_one_call():
    client = _make_fake_client(["This is a short summary."])

    result = summarize("Some short transcript text.", client=client)

    assert result == "This is a short summary."
    assert client.chat.completions.create.call_count == 1


def test_summarize_long_text_uses_map_reduce():
    long_text = "Sentence one. " * 2000  # well over the chunk limit
    num_chunks = len(_chunk_text(long_text))
    assert num_chunks > 1  # sanity check that this test text is actually "long"

    # one summary per chunk, plus one final combine call
    client = _make_fake_client([f"Summary of chunk {i}." for i in range(num_chunks)] + ["Final combined summary."])

    result = summarize(long_text, client=client)

    assert result == "Final combined summary."
    assert client.chat.completions.create.call_count == num_chunks + 1


def test_chunk_text_under_limit_returns_single_chunk():
    text = "short text"
    assert _chunk_text(text, limit=1000) == [text]


def test_chunk_text_splits_on_sentence_boundaries():
    text = "First sentence. Second sentence. Third sentence."
    chunks = _chunk_text(text, limit=20)

    assert len(chunks) > 1
    # every chunk should be non-empty and under (or near) the limit
    for c in chunks:
        assert c.strip() != ""
    # rejoining should reconstruct the original content (modulo whitespace)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")
