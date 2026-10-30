"""
Phase 1 CLI demo: transcribe an audio file and print the result.

Usage:
    python scripts/cli_demo.py path/to/audio.mp3
    python scripts/cli_demo.py path/to/audio.mp3 --model small
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script directly without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.summarizer import summarize  # noqa: E402
from app.transcriber import transcribe  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="ChatPod CLI: transcribe (and now summarize) an audio file.")
    parser.add_argument("audio_path", help="Path to an audio file (mp3, wav, m4a, ...)")
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model size: tiny, base, small, medium, large-v3 (default: base)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip summarization and only print the transcript (useful if you don't have an API key set yet).",
    )
    args = parser.parse_args()

    if not Path(args.audio_path).exists():
        print(f"Error: file not found: {args.audio_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Transcribing '{args.audio_path}' with model '{args.model}'...\n")
    result = transcribe(args.audio_path, model_size=args.model)

    print(f"Detected language: {result.language}")
    print(f"Segments: {len(result.segments)}\n")
    print("--- Transcript ---")
    print(result.text)

    if not args.no_summary:
        print("\nSummarizing...\n")
        summary = summarize(result.text)
        print("--- Summary ---")
        print(summary)


if __name__ == "__main__":
    main()
