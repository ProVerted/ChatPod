"""
Tests for Phase 3: Embeddings, Semantic Search, and RAG Q&A.

All OpenAI API calls (embeddings and chat completions) are mocked so these tests
run completely offline, deterministically, fast, and at zero API cost.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.embeddings import (  # noqa: E402
    Chunk,
    TranscriptIndex,
    chunk_segments,
    cosine_similarity,
    format_timestamp,
    generate_embeddings,
)
from app.qa import Citation, QAResponse, answer_question, answer_question_from_segments  # noqa: E402
from app.transcriber import Segment  # noqa: E402


# ---------------------------------------------------------------------------
# Test Fixtures & Mock Helpers
# ---------------------------------------------------------------------------

def _make_mock_client(embeddings_map: dict[str, list[float]] | None = None, answer_text: str = "Test Answer") -> MagicMock:
    """
    Build a fake OpenAI client that returns predetermined embeddings for given texts,
    and returns a mock completion message for chat.completions.create.
    """
    client = MagicMock()
    embeddings_map = embeddings_map or {}

    def fake_embeddings_create(model, input):
        data = []
        for i, text in enumerate(input):
            vec = embeddings_map.get(text, [0.1, 0.2, 0.3])
            item = SimpleNamespace(index=i, embedding=vec)
            data.append(item)
        return SimpleNamespace(data=data)

    client.embeddings.create.side_effect = fake_embeddings_create

    message = MagicMock()
    message.content = answer_text
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    client.chat.completions.create.return_value = completion

    return client


# ---------------------------------------------------------------------------
# Timestamp & Chunker Unit Tests
# ---------------------------------------------------------------------------

def test_format_timestamp():
    assert format_timestamp(0.0) == "00:00"
    assert format_timestamp(45.2) == "00:45"
    assert format_timestamp(65.0) == "01:05"
    assert format_timestamp(3600.0) == "01:00:00"
    assert format_timestamp(3665.0) == "01:01:05"


def test_chunk_segments_empty():
    assert chunk_segments([]) == []


def test_chunk_segments_groups_by_max_chars():
    segments = [
        Segment(start=0.0, end=2.0, text="Hello world."),
        Segment(start=2.0, end=4.5, text="Welcome to the podcast."),
        Segment(start=4.5, end=8.0, text="Today we discuss artificial intelligence."),
    ]
    # Small max_chars to force split
    chunks = chunk_segments(segments, max_chars=35, overlap_segments=0)
    assert len(chunks) >= 2
    assert chunks[0].start == 0.0
    assert chunks[0].end >= 2.0
    assert "Hello world." in chunks[0].text


def test_chunk_segments_preserves_start_and_end_boundaries():
    segments = [
        Segment(start=10.0, end=15.0, text="Segment 1 text."),
        Segment(start=15.0, end=25.0, text="Segment 2 text."),
    ]
    chunks = chunk_segments(segments, max_chars=500, overlap_segments=0)
    assert len(chunks) == 1
    assert chunks[0].start == 10.0
    assert chunks[0].end == 25.0
    assert chunks[0].text == "Segment 1 text. Segment 2 text."
    assert chunks[0].timestamp_label == "[00:10 - 00:25]"


# ---------------------------------------------------------------------------
# Vector Math & Similarity Tests
# ---------------------------------------------------------------------------

def test_cosine_similarity_orthogonal_and_identical():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert round(cosine_similarity([1.0, 2.0], [1.0, 2.0]), 5) == 1.0
    assert round(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), 5) == -1.0


def test_cosine_similarity_edge_cases():
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# ---------------------------------------------------------------------------
# Embeddings & Index Search Tests
# ---------------------------------------------------------------------------

def test_generate_embeddings_calls_api():
    client = _make_mock_client({"hello": [0.5, 0.5]})
    embs = generate_embeddings(["hello"], client=client)
    assert len(embs) == 1
    assert embs[0] == [0.5, 0.5]
    client.embeddings.create.assert_called_once()


def test_transcript_index_search():
    # Setup two chunks: chunk 1 is closer to query vector [1.0, 0.0]
    chunk1 = Chunk(chunk_id=0, text="AI discussion", start=0.0, end=10.0, embedding=[1.0, 0.0])
    chunk2 = Chunk(chunk_id=1, text="Cooking tips", start=10.0, end=20.0, embedding=[0.0, 1.0])

    index = TranscriptIndex(chunks=[chunk1, chunk2])
    client = _make_mock_client({"Tell me about AI": [0.95, 0.05]})

    results = index.search("Tell me about AI", client=client, top_k=1)
    assert len(results) == 1
    best_chunk, score = results[0]
    assert best_chunk.chunk_id == 0
    assert best_chunk.text == "AI discussion"
    assert score > 0.9


# ---------------------------------------------------------------------------
# Q&A RAG Pipeline Tests
# ---------------------------------------------------------------------------

def test_answer_question_empty_inputs():
    client = _make_mock_client()
    index = TranscriptIndex(chunks=[])

    # Empty question
    resp = answer_question("", index=index, client=client)
    assert resp.answer == ""
    assert resp.citations == []

    # Empty index
    resp2 = answer_question("What is AI?", index=index, client=client)
    assert "No transcript content" in resp2.answer


def test_answer_question_end_to_end_mocked():
    chunk = Chunk(
        chunk_id=0,
        text="The team launched Pegasus fine-tuning in October.",
        start=65.0,
        end=90.0,
        embedding=[1.0, 0.0, 0.0],
    )
    index = TranscriptIndex(chunks=[chunk])
    client = _make_mock_client(
        embeddings_map={"When did they launch?": [0.99, 0.0, 0.0]},
        answer_text="They launched Pegasus fine-tuning in October [01:05].",
    )

    resp = answer_question("When did they launch?", index=index, client=client, top_k=1)

    assert isinstance(resp, QAResponse)
    assert "October [01:05]" in resp.answer
    assert len(resp.citations) == 1
    assert resp.citations[0].start == 65.0
    assert resp.citations[0].end == 90.0
    assert resp.citations[0].formatted_time == "01:05 - 01:30"
    assert client.chat.completions.create.call_count == 1


def test_answer_question_from_segments():
    segments = [
        Segment(start=0.0, end=15.0, text="Deep learning architectures have revolutionized NLP."),
        Segment(start=15.0, end=30.0, text="Transformers rely heavily on self-attention mechanisms."),
    ]
    client = _make_mock_client(
        embeddings_map={
            "Deep learning architectures have revolutionized NLP. Transformers rely heavily on self-attention mechanisms.": [0.8, 0.2],
            "What do transformers rely on?": [0.85, 0.15],
        },
        answer_text="Transformers rely on self-attention mechanisms [00:15].",
    )

    resp = answer_question_from_segments(
        question="What do transformers rely on?",
        segments=segments,
        client=client,
        top_k=1,
    )

    assert "self-attention" in resp.answer
    assert len(resp.citations) >= 1
    assert resp.citations[0].start == 0.0
