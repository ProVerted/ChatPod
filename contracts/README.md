# ChatPod Architectural & API Data Contracts

This document formalizes the internal interface contracts between ChatPod's ingestion pipeline, RAG retrieval engine, API layer, and client interface.

---

## 1. Pipeline Data Contracts

```
[Audio Ingest] 
      │
      ▼ (CTranslate2 Whisper int8)
[TranscriptionResult]
      ├── text: str
      ├── language: str
      └── segments: list[Segment(start, end, text)]
            │
            ├──────────────────────────┐
            ▼ (Map-Reduce Chunking)    ▼ (Sliding Window & OpenAI Embeddings)
     [Episode Summary]          [TranscriptIndex]
                                       │
                                       ▼ (Cosine Similarity Retrieval)
                                [Ranked Chunks + Timestamps]
                                       │
                                       ▼ (Grounded Synthesis)
                                [QAResponse + Citations]
```

### Ingestion Data Model
- **`Segment`**: Atomic transcribed unit with float second precision:
  - `start: float` (e.g. `0.0`)
  - `end: float` (e.g. `4.5`)
  - `text: str` (e.g. `"Welcome to the episode."`)
- **`Chunk`**: Aggregated semantic window for embedding and retrieval:
  - `chunk_id: int`
  - `start: float`
  - `end: float`
  - `text: str`
  - `embedding: list[float]` (1536-dimensional vector for `text-embedding-3-small`)

---

## 2. API Contract Endpoints

| Method | Route | Description | Request Body | Response Body |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` | Liveness & readiness probe | *None* | `HealthResponse` |
| `POST` | `/transcribe` | Ingest audio from stream URL | `TranscribeUrlRequest` | `TranscribeResponse` |
| `POST` | `/transcribe/upload` | Ingest audio via multipart upload | `multipart/form-data` | `TranscribeResponse` |
| `POST` | `/summarize` | Map-Reduce text summarization | `SummarizeRequest` | `SummarizeResponse` |
| `POST` | `/qa` | Conversational RAG Q&A query | `QARequest` | `QAResponseModel` |

All schemas are strictly validated via **Pydantic v2** in `backend/app/models.py`.
