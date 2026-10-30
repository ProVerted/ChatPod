"""
Tests for the Phase 1 transcriber.

These tests mock the underlying WhisperModel so they run fast, offline, and
in CI without downloading model weights. There's a separate, manual
smoke-test instruction at the bottom of this file for verifying against a
real audio file on your own machine (which does need model weights and
therefore network access once).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.transcriber import transcribe  # noqa: E402


def _fake_segment(start, end, text):
    return SimpleNamespace(start=start, end=end, text=text)


@patch("app.transcriber._get_model")
def test_transcribe_joins_segments_into_full_text(mock_get_model):
    fake_model = MagicMock()
    fake_model.transcribe.return_value = (
        [
            _fake_segment(0.0, 1.5, " Hello there."),
            _fake_segment(1.5, 3.0, " This is a test podcast."),
        ],
        SimpleNamespace(language="en"),
    )
    mock_get_model.return_value = fake_model

    result = transcribe("fake_path.mp3")

    assert result.text == "Hello there. This is a test podcast."
    assert result.language == "en"
    assert len(result.segments) == 2
    assert result.segments[0].text == "Hello there."
    assert result.segments[1].start == 1.5


@patch("app.transcriber._get_model")
def test_transcribe_handles_empty_audio(mock_get_model):
    fake_model = MagicMock()
    fake_model.transcribe.return_value = ([], SimpleNamespace(language="en"))
    mock_get_model.return_value = fake_model

    result = transcribe("fake_silence.mp3")

    assert result.text == ""
    assert result.segments == []


@patch("app.transcriber._get_model")
def test_transcribe_strips_whitespace_from_segments(mock_get_model):
    fake_model = MagicMock()
    fake_model.transcribe.return_value = (
        [_fake_segment(0.0, 1.0, "   padded text   ")],
        SimpleNamespace(language="en"),
    )
    mock_get_model.return_value = fake_model

    result = transcribe("fake_path.mp3")

    assert result.segments[0].text == "padded text"
    assert result.text == "padded text"


# ---------------------------------------------------------------------------
# Manual smoke test (not run automatically — needs a real audio file and,
# on first run, network access to download whisper weights):
#
#   python scripts/cli_demo.py path/to/any_short_audio.mp3
#
# Confirms the real faster-whisper integration works end to end.
# ---------------------------------------------------------------------------
