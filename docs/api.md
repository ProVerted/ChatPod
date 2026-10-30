# ChatPod REST API Documentation

The ChatPod API provides production-grade endpoints for transcribing audio files or streams, generating Map-Reduce AI summaries, and performing conversational Retrieval-Augmented Generation (RAG) over timestamped transcript segments.

- **Base URL**: `http://localhost:8000`
- **Interactive OpenAPI Documentation**: `http://localhost:8000/docs`
- **ReDoc Specification**: `http://localhost:8000/redoc`

---

## 1. System Endpoints

### `GET /health`
Probe service readiness and API health.

**Response `200 OK`**:
```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

---

## 2. Transcription & Summarization

### `POST /transcribe`
Transcribe audio from a direct streaming URL (e.g., MP3/WAV/M4A), optionally generating an episode summary and priming active session memory for chat.

**Request Body**:
```json
{
  "url": "https://example.com/episode.mp3",
  "model_size": "base",
  "generate_summary": true
}
```

**Response `200 OK`**:
```json
{
  "text": "Full transcript of the podcast episode...",
  "language": "en",
  "segments": [
    {
      "start": 0.0,
      "end": 4.5,
      "text": "Welcome to today's episode."
    }
  ],
  "summary": "Concise 3-5 sentence synthesis of the episode key takeaways."
}
```

---

### `POST /transcribe/upload`
Upload an audio file directly via multipart form data.

**Parameters**:
- `file`: Audio file (`.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`)
- `model_size`: Whisper model size (`tiny`, `base`, `small`, `medium`)
- `generate_summary`: Boolean (default `true`)

---

### `POST /summarize`
Summarize arbitrary text transcripts using Map-Reduce chunking.

**Request Body**:
```json
{
  "text": "Arbitrary transcript text to summarize..."
}
```

**Response `200 OK`**:
```json
{
  "summary": "Cohesive summary of the provided text."
}
```

---

## 3. Conversational RAG Q&A

### `POST /qa`
Query the podcast using semantic vector retrieval and grounded LLM synthesis. Automatically queries the active episode loaded in memory or accepts explicit segments.

**Request Body**:
```json
{
  "question": "What did the guest say about model training latency?"
}
```

**Response `200 OK`**:
```json
{
  "answer": "The guest highlighted that quantizing weights to int8 reduced latency by 65% [04:15].",
  "citations": [
    {
      "start": 255.0,
      "end": 280.0,
      "text": "By applying int8 quantization, we cut down inferencing latency significantly.",
      "formatted_time": "04:15 - 04:40"
    }
  ]
}
```
