"""
Phase 3: Retrieval-Augmented Generation (RAG) Q&A for Podcast Transcripts.

Given a user question, retrieves the most semantically relevant transcript
segments using vector search, augments the prompt with timestamped context,
and synthesizes an accurate answer with citations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from openai import OpenAI

from app.config import settings
from app.embeddings import TranscriptIndex, format_timestamp

if TYPE_CHECKING:
    from app.transcriber import Segment


@dataclass
class Citation:
    start: float
    end: float
    text: str
    formatted_time: str

    @classmethod
    def from_chunk(cls, chunk) -> Citation:
        return cls(
            start=chunk.start,
            end=chunk.end,
            text=chunk.text,
            formatted_time=f"{format_timestamp(chunk.start)} - {format_timestamp(chunk.end)}",
        )


@dataclass
class QAResponse:
    answer: str
    citations: list[Citation] = field(default_factory=list)


_QA_SYSTEM_PROMPT = (
    "You are an intelligent, precise podcast assistant. Your task is to answer "
    "questions about a podcast episode based strictly on the provided transcript excerpts.\n\n"
    "Guidelines:\n"
    "1. Answer concisely, directly, and factually using the excerpts.\n"
    "2. Include timestamp references in your answer (e.g. [03:15]) whenever citing specific information.\n"
    "3. If the answer cannot be determined from the excerpts, state clearly: "
    "'I could not find information about that in this podcast episode.'"
)

_QA_USER_TEMPLATE = (
    "Here are relevant excerpts from the podcast transcript:\n\n"
    "{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def _format_context(ranked_chunks: list[tuple]) -> str:
    """Format retrieved chunks into a prompt-friendly string with timestamps."""
    context_blocks = []
    for chunk, _score in ranked_chunks:
        header = f"[{format_timestamp(chunk.start)} - {format_timestamp(chunk.end)}]"
        context_blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(context_blocks)


def answer_question(
    question: str,
    index: TranscriptIndex,
    client: OpenAI | None = None,
    top_k: int = 3,
) -> QAResponse:
    """
    Answer a question using a pre-built TranscriptIndex.

    Args:
        question: User query string.
        index: TranscriptIndex instance containing chunk embeddings.
        client: Optional OpenAI client override for dependency injection / mocking.
        top_k: Number of relevant chunks to retrieve.

    Returns:
        QAResponse containing the LLM answer and timestamped citations.
    """
    clean_question = question.strip()
    if not clean_question:
        return QAResponse(answer="", citations=[])

    if not index.chunks:
        return QAResponse(
            answer="No transcript content available to answer this question.",
            citations=[],
        )

    client = client or OpenAI(api_key=settings.openai_api_key)

    # 1. Retrieve top-k semantic matches
    ranked_chunks = index.search(clean_question, client=client, top_k=top_k)
    if not ranked_chunks:
        return QAResponse(
            answer="I could not find information about that in this podcast episode.",
            citations=[],
        )

    # 2. Build context and citations
    context_str = _format_context(ranked_chunks)
    citations = [Citation.from_chunk(chunk) for chunk, _score in ranked_chunks]

    # 3. Prompt LLM
    user_prompt = _QA_USER_TEMPLATE.format(context=context_str, question=clean_question)

    completion = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": _QA_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    answer_text = completion.choices[0].message.content.strip()

    return QAResponse(answer=answer_text, citations=citations)


def answer_question_from_segments(
    question: str,
    segments: list[Segment],
    client: OpenAI | None = None,
    top_k: int = 3,
) -> QAResponse:
    """
    Convenience function: builds an on-the-fly index from segments and answers the question.
    """
    index = TranscriptIndex.build(segments, client=client)
    return answer_question(question, index=index, client=client, top_k=top_k)
