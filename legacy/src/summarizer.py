from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import os

MODEL_PATH = os.getenv("PEGASUS_MODEL_PATH", "google/pegasus-cnn_dailymail")

print(f"Loading PEGASUS from: {MODEL_PATH}")
tokenizer = PegasusTokenizer.from_pretrained(MODEL_PATH)
model = PegasusForConditionalGeneration.from_pretrained(MODEL_PATH)


def chunk_text(text: str, max_tokens: int = 900) -> list:
    """
    Splits long transcript into chunks.
    Fixes the silent truncation bug from the original notebook
    where PEGASUS was only seeing the first ~1024 tokens.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_chunk.append(word)
        current_length += 1
        if current_length >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def summarize(text: str) -> str:
    """
    Summarizes transcript using fine-tuned PEGASUS.
    Uses chunking so the full transcript is summarized,
    not just the first few minutes like before.
    """
    print("Chunking transcript...")
    chunks = chunk_text(text)
    print(f"Split into {len(chunks)} chunks")

    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)}...")
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )
        summary_ids = model.generate(
            inputs.input_ids,
            num_beams=4,
            min_length=40,
            max_length=150,
            length_penalty=0.5,
            early_stopping=True
        )
        chunk_summary = tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )
        chunk_summaries.append(chunk_summary)

    # If only one chunk, return directly
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    # Otherwise merge chunk summaries and summarize again
    print("Merging chunk summaries into final summary...")
    merged = " ".join(chunk_summaries)
    inputs = tokenizer(
        merged,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )
    final_ids = model.generate(
        inputs.input_ids,
        num_beams=4,
        min_length=75,
        max_length=200,
        length_penalty=0.5,
        early_stopping=True
    )
    final_summary = tokenizer.decode(
        final_ids[0],
        skip_special_tokens=True
    )

    return final_summary