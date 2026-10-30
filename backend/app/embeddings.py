"""
Phase 3: Vector Embeddings and Semantic Retrieval for Podcast Transcripts.

Splits timestamped audio segments into semantic chunks, generates vector
embeddings via OpenAI's embedding model, and performs cosine similarity search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from openai import OpenAI

from app.config import settings

if TYPE_CHECKING:
    from app.transcriber import Segment


def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS format."""
    total_seconds = max(0, int(round(seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass
class Chunk:
    chunk_id: int
    text: str
    start: float
    end: float
    embedding: list[float] = field(default_factory=list)

    @property
    def timestamp_label(self) -> str:
        return f"[{format_timestamp(self.start)} - {format_timestamp(self.end)}]"


def chunk_segments(
    segments: list[Segment],
    max_chars: int = 600,
    overlap_segments: int = 1,
) -> list[Chunk]:
    """
    Group consecutive audio segments into larger contextual chunks while
    preserving start and end timestamps.

    Args:
        segments: List of timestamped Segments from transcriber.
        max_chars: Approximate maximum character limit per chunk.
        overlap_segments: Number of trailing segments from previous chunk to
                          carry into next chunk for semantic continuity.

    Returns:
        List of Chunks with accurate start/end intervals.
    """
    if not segments:
        return []

    chunks: list[Chunk] = []
    chunk_id = 0
    idx = 0
    n = len(segments)

    while idx < n:
        current_texts: list[str] = []
        start_time = segments[idx].start
        end_time = segments[idx].end
        current_char_count = 0
        step_end = idx

        while step_end < n:
            seg = segments[step_end]
            seg_text = seg.text.strip()
            if not seg_text:
                step_end += 1
                continue

            # If adding this segment exceeds max_chars and we already have content, stop
            if current_char_count > 0 and (current_char_count + len(seg_text) + 1) > max_chars:
                break

            current_texts.append(seg_text)
            current_char_count += len(seg_text) + 1
            end_time = seg.end
            step_end += 1

        if current_texts:
            merged_text = " ".join(current_texts).strip()
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=merged_text,
                    start=start_time,
                    end=end_time,
                )
            )
            chunk_id += 1

        # Advance window: step_end minus overlap, ensuring at least +1 forward movement
        next_idx = max(idx + 1, step_end - overlap_segments)
        if next_idx >= n and step_end >= n:
            break
        idx = next_idx

    return chunks


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def generate_embeddings(
    texts: list[str],
    client: OpenAI | None = None,
) -> list[list[float]]:
    """Generate vector embeddings for a list of texts using OpenAI."""
    if not texts:
        return []

    client = client or OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
    )
    # Ensure ordered extraction by index
    sorted_data = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in sorted_data]


class TranscriptIndex:
    """In-memory vector index for fast semantic retrieval over podcast chunks."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

    @classmethod
    def build(
        cls,
        segments: list[Segment],
        client: OpenAI | None = None,
        max_chars: int = 600,
    ) -> TranscriptIndex:
        """Create and populate a vector index from audio segments."""
        chunks = chunk_segments(segments, max_chars=max_chars)
        if not chunks:
            return cls(chunks=[])

        texts = [c.text for c in chunks]
        embeddings = generate_embeddings(texts, client=client)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        return cls(chunks=chunks)

    def search(
        self,
        query: str,
        client: OpenAI | None = None,
        top_k: int = 3,
    ) -> list[tuple[Chunk, float]]:
        """
        Search for top_k most semantically relevant chunks for a given query.
        Returns tuples of (Chunk, similarity_score).
        """
        if not self.chunks or not query.strip():
            return []

        query_embeddings = generate_embeddings([query], client=client)
        if not query_embeddings:
            return []
        query_vec = query_embeddings[0]

        scored: list[tuple[Chunk, float]] = []
        for chunk in self.chunks:
            if not chunk.embedding:
                continue
            score = cosine_similarity(query_vec, chunk.embedding)
            scored.append((chunk, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]
