import whisper

def transcribe(audio_path: str, model_size: str = "base") -> dict:
    """
    Transcribes an audio file using Whisper.
    Returns the full text AND word-level timestamps.
    
    model_size options: "tiny", "base", "small", "medium", "large"
    - tiny   → fastest, least accurate
    - base   → good balance for most podcasts  ← start here
    - medium → better accuracy, slower
    """
    print(f"Loading Whisper ({model_size})...")
    model = whisper.load_model(model_size)

    print(f"Transcribing {audio_path}...")
    result = model.transcribe(audio_path, verbose=False)

    return {
        "text": result["text"],           # full transcript as one string
        "segments": result["segments"],   # list of chunks with timestamps
        "language": result["language"]    # auto-detected language
    }


if __name__ == "__main__":
    # Quick test — point this at any mp3/wav file you have
    output = transcribe(r"C:\Users\JANARTHTHAN\Desktop\Mine\SnapTube Audio\Architects - _Animals_ (Orchestral Version) - Live at Abbey Road(MP3_320K).mp3")
    print("\n--- TRANSCRIPT ---")
    print(output["text"][:500])  # print first 500 characters
    print("\n--- DETECTED LANGUAGE ---")
    print(output["language"])