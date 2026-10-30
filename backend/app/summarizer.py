"""
Phase 2: transcript text -> summary, via an LLM API.

Kept to the same "one function, one job" shape as transcriber.py.
Long transcripts are chunked and summarized in a simple map-reduce style
so this doesn't fall over on a full-length podcast; short transcripts
just take the "map" step's single call.
"""

from __future__ import annotations

from openai import OpenAI

from app.config import settings

# Rough chars-per-token safety margin. Not exact, just a simple guard so we
# don't send more text than the model can comfortably handle in one call.
_CHUNK_CHAR_LIMIT = 12_000

_SUMMARIZE_CHUNK_PROMPT = (
    "Summarize the following podcast transcript excerpt in 3-5 sentences. "
    "Focus on the key points, arguments, and any conclusions. "
    "Write plainly, no bullet points.\n\n"
    "Transcript excerpt:\n{chunk}"
)

_COMBINE_PROMPT = (
    "The following are summaries of consecutive excerpts of the same podcast episode, "
    "in order. Combine them into a single coherent summary of the full episode, "
    "3-6 sentences, no bullet points, no mention that this is a summary of summaries.\n\n"
    "{combined}"
)


def _chunk_text(text: str, limit: int = _CHUNK_CHAR_LIMIT) -> list[str]:
    """Split text into chunks under `limit` chars, breaking on sentence boundaries where possible."""
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        split_at = remaining.rfind(". ", 0, limit)
        if split_at == -1:
            split_at = limit
        else:
            split_at += 1  # keep the period
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _call_llm(prompt: str, client: OpenAI) -> str:
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def summarize(text: str, client: OpenAI | None = None) -> str:
    """
    Summarize a (possibly long) transcript.

    Args:
        text: full transcript text
        client: optional OpenAI client override, mainly for testing/DI

    Returns:
        A short plain-text summary.
    """
    if not text.strip():
        return ""

    client = client or OpenAI(api_key=settings.openai_api_key)

    chunks = _chunk_text(text)

    if len(chunks) == 1:
        return _call_llm(_SUMMARIZE_CHUNK_PROMPT.format(chunk=chunks[0]), client)

    # Map: summarize each chunk independently
    chunk_summaries = [_call_llm(_SUMMARIZE_CHUNK_PROMPT.format(chunk=c), client) for c in chunks]

    # Reduce: combine chunk summaries into one
    combined = "\n\n".join(chunk_summaries)
    return _call_llm(_COMBINE_PROMPT.format(combined=combined), client)
