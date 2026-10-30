import whisper

def transcribe(audio_path: str, model_size: str = "base") -> dict:
    """
    Transcribes audio using Whisper.
    Replaces Google Speech Recognition from the original notebook.
    
    model_size options:
    - "tiny"   → fastest, less accurate
    - "base"   → good balance (recommended)
    - "small"  → better accuracy, slower
    """
    print(f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)

    print(f"Transcribing: {audio_path}")
    result = model.transcribe(audio_path, verbose=False)

    return {
        "text": result["text"],
        "segments": result["segments"],
        "language": result["language"]
    }