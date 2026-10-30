import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from src.transcriber import transcribe
from src.summarizer import summarize
from src.chatbot import ask

load_dotenv()

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# In-memory cache (same pattern as original notebook)
cache = {
    "transcript": "",
    "summary": ""
}

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "downloads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    data = request.get_json()
    if not data or "audio_path" not in data:
        return jsonify({"error": "Missing 'audio_path' in request body"}), 400

    try:
        logging.info("Transcribing audio...")
        result = transcribe(data["audio_path"])
        cache["transcript"] = result["text"]

        logging.info("Summarizing transcript...")
        cache["summary"] = summarize(result["text"])

        return jsonify({
            "message": "Transcription and summarization complete",
            "summary": cache["summary"],
            "language": result["language"]
        }), 200

    except Exception as e:
        logging.exception("Pipeline error")
        return jsonify({"error": str(e)}), 500


@app.route("/qa", methods=["POST"])
def qa():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    if not cache["transcript"]:
        return jsonify({
            "error": "No transcript available. Run /transcribe first."
        }), 400

    answer = ask(
        question=data["question"],
        transcript=cache["transcript"],
        summary=cache["summary"]
    )

    return jsonify({"answer": answer}), 200


if __name__ == "__main__":
    app.run(port=5000, debug=True)