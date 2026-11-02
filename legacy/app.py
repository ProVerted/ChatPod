import gradio as gr
from src.transcriber import transcribe
from src.summarizer import summarize    # your existing PEGASUS logic
from src.chatbot import ask             # your existing DeepSeek logic

# We'll store the transcript in memory between interactions
transcript_store = {"text": "", "segments": []}

def process_audio(audio_file):
    """Step 1: User uploads audio → we transcribe it"""
    result = transcribe(audio_file)
    transcript_store["text"] = result["text"]
    transcript_store["segments"] = result["segments"]
    
    summary = summarize(result["text"])
    return result["text"][:1000] + "...", summary

def chat(user_question):
    """Step 2: User asks a question → chatbot answers from transcript"""
    if not transcript_store["text"]:
        return "Please upload and process an audio file first!"
    
    answer = ask(user_question, transcript_store["text"])
    return answer

# Build the UI
with gr.Blocks(title="ChatPod") as app:
    gr.Markdown("# 🎙️ ChatPod — Chat with any Podcast")
    
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Upload your podcast")
        process_btn = gr.Button("Transcribe & Summarize", variant="primary")
    
    with gr.Row():
        transcript_box = gr.Textbox(label="Transcript Preview", lines=8)
        summary_box = gr.Textbox(label="Summary", lines=8)
    
    process_btn.click(process_audio, inputs=audio_input, 
                      outputs=[transcript_box, summary_box])
    
    gr.Markdown("---")
    gr.Markdown("### 💬 Ask anything about this podcast")
    
    question_box = gr.Textbox(label="Your question", placeholder="Who was the guest? What was the main topic?")
    answer_box = gr.Textbox(label="Answer", lines=4)
    ask_btn = gr.Button("Ask", variant="primary")
    
    ask_btn.click(chat, inputs=question_box, outputs=answer_box)

app.launch()