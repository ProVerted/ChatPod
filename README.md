---
title: ChatPod
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
---

ChatPod
Overview

ChatPod is an AI-powered system designed to automatically transcribe podcasts and long-form audio, generate high-quality abstractive summaries, and allow users to ask questions about the audio content using a Large Language Model (LLM).

This solution is ideal for students, developers, researchers, podcast listeners, and anyone who needs fast and reliable extraction of key information from long audio files. Developers can also reuse and extend this system to build their own domain-specific audio summarization tools.

Features
🔊 Podcast & Audio Transcription

Automatically converts speech to text using Google Speech Recognition with support for long audio chunking, retry logic, and preprocessing.

🧠 LLM-Powered Summarization

Uses the PEGASUS transformer model to generate clean, human-readable summaries of lengthy audio segments.

❓ Intelligent Q&A

Users can ask questions about the content. ChatPod uses DeepSeek LLM to produce contextual answers based on the transcript.

📁 Vector Storage and Retrieval

The system generates semantic embeddings using sentence transformers and stores them in ChromaDB for efficient search and retrieval.

🌐 Flask API

Provides simple REST API endpoints for uploading audio, generating summaries, and performing question answering. Ngrok can be used to temporarily expose the API to the internet.

Usage

Once the Flask server is running, users can:

Upload an audio file via /upload_audio

Receive the transcript and summary

Ask questions using /ask
This system can be integrated into web apps, mobile apps, or CLI workflows.

Installation Steps
1. Clone the repository
git clone https://github.com/ProVerted/ChatPod.git
cd ChatPod

2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

3. Install dependencies
pip install -r requirements.txt

Running ChatPod
1. Start the Flask server
python main.py

2. Optional – Expose via ngrok
ngrok http 5000

API Endpoints
POST /upload_audio

Upload an audio file and automatically:

Transcribe the audio

Generate summaries

Store sentence embeddings in ChromaDB

POST /ask

Ask a question related to the uploaded audio.

Example request:

{
  "question": "What main topic was discussed?"
}

Tools & Technologies

Google Speech Recognition – Speech-to-text transcription

PEGASUS – Abstractive summarization

DeepSeek LLM – Natural language question answering

ChromaDB – Vector storage and semantic retrieval

Sentence Transformers – Embedding generation

Flask – REST API

Ngrok – Public API tunneling

Project Flow
Audio Input  
      ↓  
Preprocessing & Chunking  
      ↓  
Google Speech-to-Text → Transcript  
      ↓  
PEGASUS Summarization  
      ↓  
Embedding Generation → ChromaDB  
      ↓  
DeepSeek LLM → Q&A  

Future Enhancements

Add a full GUI dashboard

Multi-language support

Offline transcription model

Speaker recognition

Better ranking of retrieved context chunks
