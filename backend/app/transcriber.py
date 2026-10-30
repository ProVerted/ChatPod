"""
Phase 1: Audio -> text transcription using faster-whisper.

Kept deliberately small: one function, one job. No server, no summarization,
no Q&A here — those are Phase 2+.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from faster_whisper import WhisperModel


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    language: str
    segments: list[Segment] = field(default_factory=list)


@lru_cache(maxsize=1)
def _get_model(model_size: str = "base") -> WhisperModel:
    """
    Load (and cache) the whisper model. Cached so repeated calls in the same
    process — e.g. multiple CLI invocations in a test run — don't reload the
    model from disk every time.

    model_size options (speed/accuracy tradeoff): tiny, base, small, medium, large-v3
    Start with "base" for local dev; it's fast and good enough to validate the pipeline.
    """
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe(audio_path: str, model_size: str = "base") -> TranscriptionResult:
    """
    Transcribe an audio file to text.

    Args:
        audio_path: path to an audio file (mp3, wav, m4a, etc.)
        model_size: which whisper model to use, see _get_model docstring

    Returns:
        TranscriptionResult with full text, detected language, and timed segments.
    """
    model = _get_model(model_size)
    segments_iter, info = model.transcribe(audio_path)

    segments: list[Segment] = []
    full_text_parts: list[str] = []

    for seg in segments_iter:
        clean_text = seg.text.strip()
        segments.append(Segment(start=seg.start, end=seg.end, text=clean_text))
        full_text_parts.append(clean_text)

    return TranscriptionResult(
        text=" ".join(full_text_parts).strip(),
        language=info.language,
        segments=segments,
    )
