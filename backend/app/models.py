"""
Phase 4: Pydantic v2 Request & Response Data Contracts for ChatPod API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SegmentModel(BaseModel):
    start: float = Field(..., description="Start timestamp in seconds")
    end: float = Field(..., description="End timestamp in seconds")
    text: str = Field(..., description="Transcribed segment text")


class TranscribeUrlRequest(BaseModel):
    url: str = Field(..., description="Direct HTTP/HTTPS link to an audio file")
    model_size: str = Field("base", description="Whisper model size (tiny, base, small, medium)")
    generate_summary: bool = Field(True, description="Whether to also summarize the transcript")


class TranscribeResponse(BaseModel):
    text: str = Field(..., description="Full combined transcript text")
    language: str = Field(..., description="Detected audio language")
    segments: list[SegmentModel] = Field(default_factory=list, description="Timestamped segments")
    summary: str | None = Field(None, description="Generated summary of the podcast, if requested")


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Transcript text to summarize")


class SummarizeResponse(BaseModel):
    summary: str = Field(..., description="Generated concise episode summary")


class CitationModel(BaseModel):
    start: float = Field(..., description="Start timestamp in seconds")
    end: float = Field(..., description="End timestamp in seconds")
    text: str = Field(..., description="Retrieved excerpt text")
    formatted_time: str = Field(..., description="Formatted timestamp interval e.g. '01:15 - 01:45'")


class QARequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question about the podcast")
    segments: list[SegmentModel] | None = Field(
        None,
        description="Optional list of transcript segments. If omitted, uses active episode memory.",
    )


class QAResponseModel(BaseModel):
    answer: str = Field(..., description="Direct factual answer synthesized from the podcast excerpts")
    citations: list[CitationModel] = Field(
        default_factory=list,
        description="Timestamped citations referenced in the podcast audio",
    )


class HealthResponse(BaseModel):
    status: str = Field("ok", description="API health status")
    version: str = Field("1.0.0", description="API version")
