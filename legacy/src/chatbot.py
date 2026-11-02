import os
import requests

# Using DeepSeek API instead of the local 7B model
# The local model required a GPU which Hugging Face free tier doesn't have
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"


def ask(question: str, transcript: str, summary: str) -> str:
    """
    Answers a question about the podcast using DeepSeek API.
    Uses both the summary and a portion of the transcript as context.
    """
    if not DEEPSEEK_API_KEY:
        return "Error: DEEPSEEK_API_KEY not set in environment variables."

    # Build context — summary first, then first 3000 chars of transcript
    context = f"Podcast Summary:\n{summary}\n\nTranscript excerpt:\n{transcript[:3000]}"

    prompt = f"""You are a helpful assistant that answers questions about podcast episodes.
Use the context below to answer the question accurately and concisely.
If the answer isn't in the context, say so honestly.

Context:
{context}

Question: {question}
Answer:"""

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300,
        "temperature": 0.7
    }

    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    except requests.exceptions.RequestException as e:
        return f"Error calling DeepSeek API: {str(e)}"