import os
import requests
from datetime import datetime
from retrieve import retrieve_top_k

HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
HF_MODEL = "openai/gpt-oss-20b"
API_URL = f"https://huggingface.co/models/openai/gpt-oss-20b"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_hf_model(prompt: str, max_tokens: int = 512) -> str:
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.0,  # deterministic responses
            "top_p": 1.0
        },
        "options": {"wait_for_model": True}
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"[ERROR] Failed to query model: {e}"

def build_prompt(query: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    return f"""
        You are an expert study assistant helping a university student understand their course material.
        Use ONLY the provided context to answer the question.
        If you cannot find the answer, reply exactly with: "I cannot find any context in your notes".

        Context:
        {context}

        Question: {query}

        Answer:"""

def answer_question(query: str) -> str:
    # Refining user query for better accuracy
    refined_query = refine_query(query)
    print(f"[DEBUG] Refined query: {refined_query}")

    # Retrieve relevant context
    chunks = retrieve_top_k(refined_query)

    if not chunks:
        chunks = "I cannot find any context in your notes"

    # Build prompt
    prompt = build_prompt(query, chunks) + "\n\n"

    # Generate response
    answer = query_hf_model(prompt, max_tokens=512)
    return answer

def refine_query(query: str) -> str:
    """
    Takes in user query and refines it so it is inserted into the vector db with 
    more accuracy
    """
    prompt = f"""
    You are a helpful assistant. Rewrite the following question to make it clearer,
    more specific and optimized for retrieving relevant information from acadamic notes.
    Do not answer the question. Only return the imrpoved version.

    Original question: {query}
    Refined question:
    """
    return query_hf_model(prompt, max_tokens=64)

if __name__ == "__main__":
    asking = True
    while asking:
        question = input("Write your question: ")
        start_time = datetime.now()
        answer = answer_question(question)
        end_time = datetime.now()
        print("\n--- Answer ---\n")
        print(answer)
        print(f"Time to response: {end_time - start_time}")
        cont = input("Continue asking? input y for Yes and n for No: ")
        if cont == "n":
            asking = not asking
        