import os
from typing import List
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# nltk imports
import nltk
nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))
from nltk.tokenize import sent_tokenize, word_tokenize

# Sentence Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# def extract_text_from_pdf(file_path: str) -> str:
#     """Extracts text from a PDF file."""
#     reader = PdfReader(file_path)
#     text = ""
#     for page in reader.pages:
#         if page.extract_text():
#             text += page.extract_text() + "\n"
#     return text.strip()


def load_file(file_path: str) -> str:
    """Loads text from a PDF or DOCX file."""
    if file_path.lower().endswith(".pdf"):
        return extract_text(file_path)
    else:
        raise ValueError("Unsupported file format. Use PDF.")

def sliding_window_chunk(text: str, chunk_size: int = 300, overlap: int = 100) -> List[str]:
    """
    Chunking strategy that splits pdf into 300 word sizes with 100 word overlaps with
    adjacent chunks.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    return chunks

def dynamic_semantic_chunk(text: str, min_words: int = 150, max_words: int = 350, sim_threshold: float = 0.75, overlap: int = 1) -> List[str]:
    """
    Split text into chunks using semantic similarity between sentences.
    A new chunk starts when similarity between sentences drops below a threshold or current chunk
    excess max_words.
    """
    text = re.sub(r'\s+', ' ', text.strip())
    sentences = sent_tokenize(text)
    embeddings = model.encode(sentences)

    chunks = []
    current_chunk = []
    current_chunk_words = 0

    for i, sentence in enumerate(sentences):
        sent_words = word_tokenize(sentence)
        sent_len = len(sent_words)

        if not current_chunk:
            current_chunk.append(sentence)
            current_chunk_words += sent_len
            continue

        # Semantic similarity to current chunk so far
        combined_text = " ".join(current_chunk)
        combined_embedding = model.encode([combined_text])[0]
        sim = cosine_similarity([embeddings[i]], [combined_embedding])[0][0]

        # Compare sim to decide new chunk or continue
        if sim >= sim_threshold or current_chunk_words < min_words:
            current_chunk.append(sentence)
            current_chunk_words += sent_len
        else: # split the chunk
            chunks.append(" ".join(current_chunk))
            # overlap last N sentences
            overlap_sentence = current_chunk[-overlap:] if overlap > 0 else []
            current_chunk = overlap_sentence + [sentence]
            current_chunk_words = sum(len(word_tokenize(s)) for s in current_chunk)
        
        # Force split if chunk is too long
        if current_chunk_words >= max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_chunk_words = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def ingest(file_path: str) -> List[str]:
    """
    Full ingestion pipeline: load + chunk.
    Returns a list of text chunks.
    """
    print(f"[INFO] Loading file: {file_path}")
    text = load_file(file_path)
    print(f"[INFO] Document length: {len(text)} characters")

    chunks = dynamic_semantic_chunk(text)
    print(f"[INFO] Split into {len(chunks)} chunks")
    return chunks

if __name__ == "__main__":
    file_path = "data/notes.pdf"
    chunks = ingest(file_path)

    print(f"\n[INFO] {len(chunks)} chunks generated:")
    
    import random
    for i in random.sample(range(len(chunks)), 5):
        print(f"\n--- Chunk {i} ---\n{chunks[i]}\n")

    # print("\n--- First Chunk ---\n")
    # print(chunks[70])