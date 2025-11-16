# llm.py

import os
from typing import List, Dict
from groq import Groq


# CLIENT

def get_groq_client() -> Groq:
    """
    Returns an initialized Groq client.
    Requires GROQ_API_KEY in environment variables.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in environment variables.")
    return Groq(api_key=api_key)


# CONTEXT

def build_context(retrieval_results: List[Dict], max_chars: int = 6000) -> str:
    """
    Build a context string from retrieval results.
    Each result is expected to be a dict with keys: 'text', 'metadata', 'score'.
    """
    context_parts = []
    total_len = 0

    for r in retrieval_results:
        text = r.get("text", "")
        meta = r.get("metadata", {})
        source = meta.get("source", "")
        chunk_id = meta.get("chunk_id", "")

        piece = f"[chunk_id={chunk_id} source={source}]\n{text}\n\n"

        if total_len + len(piece) > max_chars:
            break

        context_parts.append(piece)
        total_len += len(piece)

    return "".join(context_parts).strip()


# RAG ANSWERS

def rag_answer(
    question: str,
    retrieval_results: List[Dict],
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.1,
) -> Dict:
   

    if not retrieval_results:
        return {
            "answer": "I couldn't find any relevant information in the documents for your question.",
            "context": "",
            "used_chunks": [],
        }

    client = get_groq_client()

    context = build_context(retrieval_results)

    system_prompt = (
        "You are a helpful medical assistant that answers strictly based on the given context.\n"
        "If the answer is not present in the context, say you don't know or suggest that the "
        "information is not available in the documents.\n"
        "Do not hallucinate or invent facts."
    )

    user_prompt = f"""
Use the provided context to answer the question thoroughly and with detail if there is a detailed answer in the context.
Include explanation where relevant.Do not cite metadata. Do not reference pages/snippets

Context:
{context}

Question: {question}

Answer in a clear, concise way. If the context is insufficient, say so.
"""

    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer_text = completion.choices[0].message.content.strip()

    return {
        "answer": answer_text,
        "context": context,
        "used_chunks": retrieval_results,
    }
