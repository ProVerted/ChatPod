import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

function App() {
  const [ingestMode, setIngestMode] = useState('url'); // 'url' | 'upload'
  const [url, setUrl] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [audioSrc, setAudioSrc] = useState(null);

  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState('');
  const [error, setError] = useState(null);

  const [podcastData, setPodcastData] = useState(null);
  const [showTranscript, setShowTranscript] = useState(false);

  const [chatHistory, setChatHistory] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);

  const chatEndRef = useRef(null);
  const audioRef = useRef(null);

  // Auto-scroll chat to bottom
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatHistory, chatLoading]);

  // Jump to specific audio timestamp
  const seekTo = (seconds) => {
    if (audioRef.current) {
      audioRef.current.currentTime = seconds;
      audioRef.current.play().catch(() => {});
    }
  };

  // Handle URL submission
  const handleUrlSubmit = async (e) => {
    e.preventDefault();
    if (!url.trim()) return;

    setLoading(true);
    setLoadingStep('Downloading and transcribing audio...');
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/transcribe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          url: url.trim(),
          model_size: 'base',
          generate_summary: true,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned error status ${response.status}`);
      }

      const data = await response.json();
      setPodcastData(data);
      setAudioSrc(url.trim());
    } catch (err) {
      console.error('Transcription error:', err);
      setError(err.message || 'Failed to process podcast URL. Please verify backend connection.');
    } finally {
      setLoading(false);
      setLoadingStep('');
    }
  };

  // Handle File upload submission
  const handleFileUploadSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) return;

    setLoading(true);
    setLoadingStep('Uploading audio and generating transcription...');
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${API_BASE_URL}/transcribe/upload?generate_summary=true&model_size=base`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Upload failed with status ${response.status}`);
      }

      const data = await response.json();
      setPodcastData(data);
      setAudioSrc(URL.createObjectURL(selectedFile));
    } catch (err) {
      console.error('File upload error:', err);
      setError(err.message || 'Failed to upload and transcribe audio.');
    } finally {
      setLoading(false);
      setLoadingStep('');
    }
  };

  // Handle Chat message submission
  const handleChatSubmit = async (e) => {
    e.preventDefault();
    const query = chatInput.trim();
    if (!query) return;

    // Append user message
    const userMessage = { role: 'user', content: query };
    setChatHistory((prev) => [...prev, userMessage]);
    setChatInput('');
    setChatLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/qa`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error ${response.status}`);
      }

      const data = await response.json();
      setChatHistory((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer || 'No answer generated.',
          citations: data.citations || [],
        },
      ]);
    } catch (err) {
      console.error('Chat error:', err);
      setChatHistory((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `Error: ${err.message || 'Unable to retrieve answer.'}`,
          citations: [],
        },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  // Reset and return to initial state
  const handleReset = () => {
    setPodcastData(null);
    setUrl('');
    setSelectedFile(null);
    setAudioSrc(null);
    setChatHistory([]);
    setError(null);
    setShowTranscript(false);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>CHAT_POD</h1>
        <p className="app-subtitle">AI-Powered Podcast Transcription, Summarization & Semantic Q&A</p>
      </header>

      <main className="app-main">
        {/* Step 1: Ingestion Panel */}
        {!podcastData && !loading && (
          <div className="url-submission">
            <div className="tab-switcher">
              <button
                type="button"
                className={`tab-btn ${ingestMode === 'url' ? 'active' : ''}`}
                onClick={() => setIngestMode('url')}
              >
                Podcast URL
              </button>
              <button
                type="button"
                className={`tab-btn ${ingestMode === 'upload' ? 'active' : ''}`}
                onClick={() => setIngestMode('upload')}
              >
                Upload Audio File
              </button>
            </div>

            {ingestMode === 'url' ? (
              <form onSubmit={handleUrlSubmit} className="ingest-form">
                <input
                  type="url"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="Enter direct audio stream URL (MP3, WAV, M4A)..."
                  className="url-input"
                  required
                />
                <button type="submit" className="submit-button">
                  Analyze
                </button>
              </form>
            ) : (
              <form onSubmit={handleFileUploadSubmit} className="upload-form">
                <label className="file-dropzone">
                  <input
                    type="file"
                    accept="audio/*"
                    onChange={(e) => setSelectedFile(e.target.files[0] || null)}
                    className="file-input-hidden"
                    required
                  />
                  <span className="file-dropzone-text">
                    {selectedFile ? `Selected: ${selectedFile.name}` : '📁 Choose an audio file (.mp3, .wav, .m4a)'}
                  </span>
                </label>
                <button type="submit" className="submit-button upload-btn" disabled={!selectedFile}>
                  Transcribe & Summarize
                </button>
              </form>
            )}

            {error && (
              <div className="error-box">
                <p>⚠️ {error}</p>
                <small>Ensure the FastAPI server is running: <code>uvicorn app.main:app --port 8000</code></small>
              </div>
            )}
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="loading-card">
            <div className="pulse-loader" />
            <p className="loading-title">Processing Podcast Audio</p>
            <p className="loading-desc">{loadingStep || 'Analyzing speech frames with Whisper...'}</p>
          </div>
        )}

        {/* Step 2: Active Episode Summary & Controls */}
        {podcastData && (
          <div className="episode-dashboard">
            <div className="dashboard-header">
              <button onClick={handleReset} className="back-button">
                ← Process Another Episode
              </button>
              <div className="meta-tags">
                <span className="badge">Language: {podcastData.language?.toUpperCase() || 'EN'}</span>
                <span className="badge">Segments: {podcastData.segments?.length || 0}</span>
              </div>
            </div>

            {/* Synchronized Audio Player */}
            {audioSrc && (
              <div className="audio-player-box">
                <span className="audio-label">🎧 Audio Player:</span>
                <audio ref={audioRef} controls src={audioSrc} className="native-player" />
              </div>
            )}

            {/* AI Summary Card */}
            <div className="summary-container">
              <h2>Executive Episode Summary</h2>
              <p className="summary-body">{podcastData.summary || 'Summary unavailable.'}</p>

              {/* Full Transcript Accordion */}
              {podcastData.segments && podcastData.segments.length > 0 && (
                <div className="transcript-accordion">
                  <button
                    type="button"
                    className="toggle-transcript-btn"
                    onClick={() => setShowTranscript(!showTranscript)}
                  >
                    {showTranscript ? 'Hide Full Transcript ▲' : 'View Full Transcript & Timestamps ▼'}
                  </button>

                  {showTranscript && (
                    <div className="transcript-list">
                      {podcastData.segments.map((seg, i) => (
                        <div key={i} className="transcript-segment" onClick={() => seekTo(seg.start)}>
                          <span className="timestamp-badge">
                            {Math.floor(seg.start / 60)}:{(Math.floor(seg.start % 60)).toString().padStart(2, '0')}
                          </span>
                          <span className="segment-text">{seg.text}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Conversational RAG Chat Interface */}
            <div className="chat-container">
              <div className="chat-header">
                <h2>Interactive Podcast Q&A</h2>
                <p className="chat-subtitle">Ask any question grounded in the podcast with timestamp citations</p>
              </div>

              <div className="chat-history">
                {chatHistory.length === 0 && (
                  <div className="chat-empty">
                    <p>💡 Ask questions like:</p>
                    <div className="sample-queries">
                      <span onClick={() => setChatInput('What are the key conclusions?')}>
                        "What are the key conclusions?"
                      </span>
                      <span onClick={() => setChatInput('Who were the main speakers or guests?')}>
                        "Who were the main speakers or guests?"
                      </span>
                    </div>
                  </div>
                )}

                {chatHistory.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`chat-message ${msg.role === 'user' ? 'user-message' : 'assistant-message'}`}
                  >
                    <div className="message-content">{msg.content}</div>

                    {/* Timestamp Citations */}
                    {msg.citations && msg.citations.length > 0 && (
                      <div className="citations-container">
                        <span className="citations-label">Sources:</span>
                        {msg.citations.map((c, cIdx) => (
                          <button
                            key={cIdx}
                            type="button"
                            className="citation-pill"
                            onClick={() => seekTo(c.start)}
                            title={`Quote: "${c.text}"`}
                          >
                            ⏱️ {c.formatted_time}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                ))}

                {chatLoading && (
                  <div className="chat-message assistant-message loading-message">
                    <span className="loading-dots">
                      Searching podcast context<span>.</span><span>.</span><span>.</span>
                    </span>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              <form onSubmit={handleChatSubmit} className="chat-form">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Ask a question about the episode..."
                  className="chat-input"
                  disabled={chatLoading}
                />
                <button type="submit" className="chat-button" disabled={chatLoading || !chatInput.trim()}>
                  Send
                </button>
              </form>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
