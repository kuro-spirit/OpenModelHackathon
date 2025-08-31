import os
import json
import numpy as np
import faiss
from typing import List
from sentence_transformers import SentenceTransformer, util

# Paths (should match embed.py)
CHUNK_METADATA_PATH = "embeddings/chunk_metadata.json"
FAISS_INDEX_PATH = "embeddings/index.faiss"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

def load_embedding_model(model_name=EMBED_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(model_name)

def load_metadata() -> List[str]:
    with open(CHUNK_METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
    
def load_faiss_index():
    return faiss.read_index(FAISS_INDEX_PATH)

def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    return np.array([model.encode(query)])

def retrieve_top_k(query: str, top_k: int = 5, initial_k: int = 20, sim_threshold: float = 0.4) -> List[str]:
    """
    Implemented using a re-ranking method where top 20 related chunks within threshold are retrieved,
    then refined and top 5 of the top 20 are returned to llama for context.
    """
    print(f"\n[DEBUG] Running retrieve_top_k()")
    print(f"[DEBUG] Query: {query}")
    print(f"[DEBUG] CHUNK_METADATA_PATH exists? {os.path.exists(CHUNK_METADATA_PATH)}")
    print(f"[DEBUG] FAISS_INDEX_PATH exists? {os.path.exists(FAISS_INDEX_PATH)}")
    model = load_embedding_model()
    query_vec = embed_query(query, model)

    index = load_faiss_index()
    D, I = index.search(query_vec, initial_k)

    # Clean and fetch valid chunk candidates
    retrieved = []
    all_chunks = load_metadata()
    print(f"\n[DEBUG] Top {initial_k} retrieved chunks:")
    retrieved_chunks = [all_chunks[i] for i in I[0]]

    print(f"[DEBUG] Retrieved {len(retrieved_chunks)} chunks from FAISS")

    # Re-rank using cosine similarity
    chunk_embeddings = model.encode(retrieved_chunks)
    query_embedding = model.encode(query)
    similarities = util.cos_sim(query_embedding, chunk_embeddings)[0]

    # Apply threshold filter
    filtered = [(chunk, sim.item()) for chunk, sim in zip(retrieved_chunks, similarities) if sim >= sim_threshold]

    if not filtered:
        print("[DEBUG] No chunks passed similarity threshold. Falling back to top-k FAISS results.")
        return retrieved_chunks[:top_k]
    # Sort by similarity (descending)
    filtered_sorted = sorted(filtered, key=lambda x: x[1], reverse=True)

    results = [chunk for chunk, _ in filtered_sorted[:top_k]]

    print(f"[DEBUG] Final results returned: {len(results)} chunks")

    return results

if __name__ == "__main__":
    query = "what is a rank"
    results = retrieve_top_k(query)

    print("\n--- Retrieved Chunks ---")
    for i, chunk in enumerate(results):
        print(f"\n[{i+1}] {chunk}\n")