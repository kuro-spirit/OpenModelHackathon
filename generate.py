# from llama_cpp import Llama
from retrieve import retrieve_top_k
import os
from datetime import datetime

os.environ["LLAMA_CUBLAS"] = "1"

# llm = Llama(
#     model_path="models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
#     n_gpu_layers=24,
#     n_ctx=8192,
#     n_threads=os.cpu_count(),
#     verbose=True
# )

def build_prompt(query: str, context_chunks: list) -> str:
    context = "\n\n".join(context_chunks)
    prompt = f"""
    You are an expert study assistant helping an university student understand course material. 
    Only use the provided context below to answer the question.
    If technical terms appear, briefly define them.
    If the answer cannot be found in the context, respond with "I cannot find any context in your
    notes". It is okay to not find any answer in the context.

    Context:
    {context}

    Question: {query}
    Answer:"""
    prompt2 = f"""
    You are a helpful study assistant. Use only the context provided below to answer the user's question.
    # If the answer is not in the context, say "I cannot find any context in your notes."

    Context:
    {context}

    Question: {query}
    """
    print(f"\n[DEBUG] Prompt Sent to LLaMA:\n{prompt2[:1000]}...\n")
    return prompt2

def answer_question(query: str) -> str:
    # Refining user query for better accuracy
    refined_query = refine_query(query)
    print(f"[DEBUG] Refined query: {refined_query}")

    # Retrieve relevant context
    chunks = retrieve_top_k(refined_query)

    if not chunks:
        chunks = "No relevant context was found."

    # Build prompt
    prompt = build_prompt(query, chunks) + "\n\n"

    # Generate response
    response = llm(prompt, max_tokens=512)
    print("\n[DEBUG] Raw LLaMA response:\n", response)

    # clean response
    answer = response["choices"][0].get("text", "").strip()
    print("\n[DEBUG] Cleaned response:\n", answer)
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
    response = llm(prompt, max_tokens=64)
    return response["choices"][0]["text"].strip()

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
        