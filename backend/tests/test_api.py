"""
Integration tests for Phase 4 FastAPI REST endpoints.

Mocks heavy ML/transcription and external API calls so these tests run
in memory, quickly, offline, and deterministically.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app, session_state  # noqa: E402
from app.qa import Citation, QAResponse  # noqa: E402
from app.transcriber import Segment, TranscriptionResult  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_session():
    """Reset session state before every test."""
    session_state.active_segments.clear()
    session_state.active_index = None
    session_state.active_summary = None
    yield
    session_state.active_segments.clear()
    session_state.active_index = None
    session_state.active_summary = None


def test_health_check(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.0.0"}


@patch("app.main.summarize")
def test_summarize_endpoint(mock_summarize, client: TestClient):
    mock_summarize.return_value = "This is a mocked summary of the podcast episode."

    payload = {"text": "Long transcript text about AI."}
    response = client.post("/summarize", json=payload)

    assert response.status_code == 200
    assert response.json() == {"summary": "This is a mocked summary of the podcast episode."}
    mock_summarize.assert_called_once_with("Long transcript text about AI.")


@patch("app.main.summarize")
@patch("app.main.transcribe")
@patch("app.main._download_audio_to_tempfile")
def test_transcribe_url_endpoint(mock_download, mock_transcribe, mock_summarize, client: TestClient):
    mock_download.return_value = "fake_temp.mp3"
    mock_transcribe.return_value = TranscriptionResult(
        text="Welcome to ChatPod.",
        language="en",
        segments=[Segment(start=0.0, end=3.5, text="Welcome to ChatPod.")],
    )
    mock_summarize.return_value = "Episode introduction."

    payload = {
        "url": "https://example.com/podcast.mp3",
        "model_size": "base",
        "generate_summary": True,
    }
    response = client.post("/transcribe", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Welcome to ChatPod."
    assert data["language"] == "en"
    assert data["summary"] == "Episode introduction."
    assert len(data["segments"]) == 1
    assert data["segments"][0]["text"] == "Welcome to ChatPod."

    # Verify session state was populated
    assert len(session_state.active_segments) == 1
    assert session_state.active_summary == "Episode introduction."


@patch("app.main.summarize")
@patch("app.main.transcribe")
def test_transcribe_upload_endpoint(mock_transcribe, mock_summarize, client: TestClient):
    mock_transcribe.return_value = TranscriptionResult(
        text="Audio upload test content.",
        language="en",
        segments=[Segment(start=0.0, end=2.0, text="Audio upload test content.")],
    )
    mock_summarize.return_value = "Uploaded audio summary."

    fake_file_content = b"fake audio byte payload"
    files = {"file": ("test.mp3", io.BytesIO(fake_file_content), "audio/mpeg")}

    response = client.post("/transcribe/upload", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Audio upload test content."
    assert data["summary"] == "Uploaded audio summary."


def test_qa_endpoint_without_active_session_returns_helpful_message(client: TestClient):
    payload = {"question": "What is discussed?"}
    response = client.post("/qa", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "No podcast transcript loaded" in data["answer"]
    assert data["citations"] == []


@patch("app.main.answer_question")
@patch("app.main.TranscriptIndex.build")
def test_qa_endpoint_with_active_session(mock_build_index, mock_answer, client: TestClient):
    # Pre-populate session state
    session_state.active_segments = [Segment(start=0.0, end=10.0, text="AI discuss.")]
    mock_index = MagicMock()
    mock_build_index.return_value = mock_index

    mock_answer.return_value = QAResponse(
        answer="The podcast talks about AI.",
        citations=[
            Citation(start=0.0, end=10.0, text="AI discuss.", formatted_time="00:00 - 00:10")
        ],
    )

    payload = {"question": "What is discussed?"}
    response = client.post("/qa", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The podcast talks about AI."
    assert len(data["citations"]) == 1
    assert data["citations"][0]["formatted_time"] == "00:00 - 00:10"
