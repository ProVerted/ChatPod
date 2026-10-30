"""
Phase 4: FastAPI Production Service Gateway for ChatPod.

Provides RESTful endpoints for:
- /health: Service health and readiness probe
- /transcribe: Audio URL or file transcription with optional auto-summarization
- /transcribe/upload: Direct audio multipart file upload
- /summarize: Long-form text summarization via Map-Reduce LLM pipeline
- /qa: Conversational RAG Q&A over podcast transcript with timestamp citations
"""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from app.embeddings import TranscriptIndex
from app.models import (
    CitationModel,
    HealthResponse,
    QARequest,
    QAResponseModel,
    SegmentModel,
    SummarizeRequest,
    SummarizeResponse,
    TranscribeResponse,
    TranscribeUrlRequest,
)
from app.qa import answer_question
from app.summarizer import summarize
from app.transcriber import Segment, transcribe


# In-memory store for the active podcast episode session
class SessionState:
    def __init__(self):
        self.active_segments: list[Segment] = []
        self.active_index: TranscriptIndex | None = None
        self.active_summary: str | None = None

    def update(self, segments: list[Segment], summary: str | None = None):
        self.active_segments = segments
        self.active_summary = summary
        self.active_index = None  # Lazy-load vector index on first query


session_state = SessionState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle handler."""
    yield
    session_state.active_segments.clear()
    session_state.active_index = None


app = FastAPI(
    title="ChatPod API",
    description="High-performance podcast transcription, summarization, and RAG Q&A service.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for React frontend (localhost:3000, 5173, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Utility Helpers
# ---------------------------------------------------------------------------

def _download_audio_to_tempfile(url: str) -> str:
    """Download audio stream to a temporary local file, returning the file path."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download audio from URL: {exc}",
        ) from exc

    # Determine suffix from URL or fallback to .mp3
    suffix = ".mp3"
    clean_path = url.split("?")[0].lower()
    for ext in [".mp3", ".wav", ".m4a", ".ogg", ".flac"]:
        if clean_path.endswith(ext):
            suffix = ext
            break

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                tmp_file.write(chunk)
        return tmp_file.name


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Service health check probe."""
    return HealthResponse(status="ok", version="1.0.0")


@app.post("/transcribe", response_model=TranscribeResponse, tags=["Audio Processing"])
def transcribe_url(payload: TranscribeUrlRequest) -> TranscribeResponse:
    """
    Transcribe a podcast from an audio URL.
    Optionally generates a summary and indexes the transcript for interactive chat.
    """
    temp_path = _download_audio_to_tempfile(payload.url)
    try:
        result = transcribe(temp_path, model_size=payload.model_size)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription error: {exc}",
        ) from exc
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    summary_text = None
    if payload.generate_summary and result.text:
        try:
            summary_text = summarize(result.text)
        except Exception:
            summary_text = "Summary generation failed or LLM API key not configured."

    # Cache active episode in memory
    session_state.update(result.segments, summary=summary_text)

    segment_models = [
        SegmentModel(start=seg.start, end=seg.end, text=seg.text)
        for seg in result.segments
    ]

    return TranscribeResponse(
        text=result.text,
        language=result.language,
        segments=segment_models,
        summary=summary_text,
    )


@app.post("/transcribe/upload", response_model=TranscribeResponse, tags=["Audio Processing"])
async def transcribe_upload(
    file: UploadFile = File(...),
    generate_summary: bool = True,
    model_size: str = "base",
) -> TranscribeResponse:
    """Upload a local audio file (.mp3, .wav, etc.) for transcription and summarization."""
    suffix = os.path.splitext(file.filename or "audio.mp3")[1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        temp_path = tmp_file.name

    try:
        result = transcribe(temp_path, model_size=model_size)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription error: {exc}",
        ) from exc
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    summary_text = None
    if generate_summary and result.text:
        try:
            summary_text = summarize(result.text)
        except Exception:
            summary_text = "Summary generation failed or LLM API key not configured."

    session_state.update(result.segments, summary=summary_text)

    segment_models = [
        SegmentModel(start=seg.start, end=seg.end, text=seg.text)
        for seg in result.segments
    ]

    return TranscribeResponse(
        text=result.text,
        language=result.language,
        segments=segment_models,
        summary=summary_text,
    )


@app.post("/summarize", response_model=SummarizeResponse, tags=["LLM Services"])
def summarize_text(payload: SummarizeRequest) -> SummarizeResponse:
    """Generate a concise summary from any arbitrary transcript text."""
    try:
        summary_text = summarize(payload.text)
        return SummarizeResponse(summary=summary_text)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization error: {exc}",
        ) from exc


@app.post("/qa", response_model=QAResponseModel, tags=["Conversational AI"])
def ask_question_endpoint(payload: QARequest) -> QAResponseModel:
    """
    Ask a question against the active podcast transcript using vector RAG.
    Returns the answer and timestamped citations.
    """
    segments_to_use: list[Segment] = []

    if payload.segments:
        # Use user-supplied segments
        segments_to_use = [
            Segment(start=s.start, end=s.end, text=s.text) for s in payload.segments
        ]
    elif session_state.active_segments:
        # Fallback to active episode session
        segments_to_use = session_state.active_segments
    else:
        return QAResponseModel(
            answer="No podcast transcript loaded. Please transcribe a podcast episode first.",
            citations=[],
        )

    # Build or reuse cached index
    try:
        if not session_state.active_index or payload.segments:
            session_state.active_index = TranscriptIndex.build(segments_to_use)

        qa_res = answer_question(payload.question, index=session_state.active_index)

        citations = [
            CitationModel(
                start=c.start,
                end=c.end,
                text=c.text,
                formatted_time=c.formatted_time,
            )
            for c in qa_res.citations
        ]
        return QAResponseModel(answer=qa_res.answer, citations=citations)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Q&A service error: {exc}",
        ) from exc
