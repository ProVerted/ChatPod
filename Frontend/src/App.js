import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const [error, setError] = useState(null);
  const chatEndRef = useRef(null);

  // Scroll to bottom of chat whenever messages change
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatHistory]);

  // Apply animation delay to messages
  useEffect(() => {
    const messages = document.querySelectorAll('.chat-message');
    messages.forEach((message, index) => {
      message.style.animationDelay = `${0.1 + (index * 0.1)}s`;
    });
  }, [chatHistory]);

  // Handle URL submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!url.trim()) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('https://8217-34-125-59-169.ngrok-free.app/transcribe', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url }),
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      
      const data = await response.json();
      setSummary(data);
      setShowChat(true);
    } catch (error) {
      console.error('Error:', error);
      setError('Failed to process podcast. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  // Handle chat message submission
  const handleChatSubmit = async (e) => {
    e.preventDefault();
    
    if (!chatInput.trim()) return;
    
    // Add user message to chat history
    const userMessage = { role: 'user', content: chatInput };
    setChatHistory([...chatHistory, userMessage]);
    setChatInput('');
    setChatLoading(true);
    
    try {
      const response = await fetch('https://8217-34-125-59-169.ngrok-free.app/qa', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: chatInput
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }
      
      const data = await response.json();
      setChatHistory(prev => [...prev, { 
        role: 'assistant', 
        content: data.answer || 'Sorry, I couldn\'t process that request.'
      }]);
    } catch (error) {
      console.error('Error:', error);
      setChatHistory(prev => [...prev, { 
        role: 'assistant', 
        content: 'Sorry, there was an error processing your request.' 
      }]);
    } finally {
      setChatLoading(false);
    }
  };

  // Handle going back to URL input
  const handleGoBack = () => {
    setSummary(null);
    setShowChat(false);
    setChatHistory([]);
    setError(null);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Podcast Summarizer</h1>
      </header>
      
      <main className="app-main">
        {!summary && !loading && (
          <div className="url-submission">
            <h2>Enter Podcast URL</h2>
            <form onSubmit={handleSubmit}>
              <input
                type="text"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="Enter podcast URL"
                className="url-input"
              />
              <button type="submit" className="submit-button">Analyze</button>
            </form>
            {error && <p className="error-message">{error}</p>}
          </div>
        )}

        {loading && (
          <div className="loading">
            <p>Processing podcast...</p>
          </div>
        )}

        {summary && (
          <div className="summary-container">
            <div className="summary-header">
              <button onClick={handleGoBack} className="back-button">← Back</button>
            </div>
            <p>{summary.summary || 'No summary available.'}</p>
          </div>
        )}

        {showChat && (
          <div className="chat-container">
            <div className="chat-header">
              <h2>Chat about this podcast</h2>
            </div>
            <div className="chat-history">
              {chatHistory.map((message, index) => (
                <div 
                  key={index}
                  className={`chat-message ${message.role === 'user' ? 'user-message' : 'assistant-message'}`}
                >
                  {message.content}
                </div>
              ))}
              {chatLoading && (
                <div className="chat-message assistant-message loading-message">
                  <span className="loading-dots">Thinking<span>.</span><span>.</span><span>.</span></span>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>
            
            <form onSubmit={handleChatSubmit} className="chat-form">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder="Ask a question about the podcast..."
                className="chat-input"
                disabled={chatLoading}
              />
              <button type="submit" className="chat-button" disabled={chatLoading}>
                {chatLoading ? 'Sending...' : 'Send'}
              </button>
            </form>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;