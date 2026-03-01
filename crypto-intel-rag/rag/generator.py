"""
LLM response generator using Google Gemini API.
Builds a grounded prompt from retrieved chunks and returns answer + sources.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _is_key_set() -> bool:
    return bool(GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here")


def _demo_response(query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fallback response when Gemini API key is not configured."""
    if not chunks:
        return {
            "answer": "⚠️ No relevant documents found in the knowledge base.",
            "sources": [],
            "model": "demo",
        }
    top = chunks[0]
    answer = (
        f"**[Demo Mode — Gemini API key not set]**\n\n"
        f"Based on the retrieved documents, the most relevant chunk is from "
        f"**{top.get('source', 'Unknown')}** dated {top.get('date', 'N/A')}:\n\n"
        f"> {top.get('text', '')[:500]}...\n\n"
        f"Set `GEMINI_API_KEY` in your `.env` file to get full AI-generated answers."
    )
    from rag.prompt_template import build_sources_list
    return {
        "answer": answer,
        "sources": build_sources_list(chunks),
        "model": "demo",
    }


def generate(query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a grounded answer using Gemini.

    Args:
        query: User's question
        chunks: Retrieved context chunks from Endee

    Returns:
        { answer: str, sources: List[str], model: str }
    """
    from rag.prompt_template import SYSTEM_PROMPT, build_prompt, build_sources_list

    if not _is_key_set():
        return _demo_response(query, chunks)

    try:
        from google import genai
        from google.genai import types

        # Initialize the new genai client
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        user_message = build_prompt(query, chunks)
        
        # Use the new generate_content API with system instructions
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
            )
        )
        answer = response.text.strip()

        return {
            "answer": answer,
            "sources": build_sources_list(chunks),
            "model": GEMINI_MODEL,
        }

    except Exception as e:
        return {
            "answer": f"❌ Gemini API error: {str(e)}\n\nCheck your `GEMINI_API_KEY` in `.env`.",
            "sources": build_sources_list(chunks) if chunks else [],
            "model": "error",
        }


def ask(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Full RAG pipeline: retrieve → generate.
    Returns { answer, sources, model, chunks, error? }
    """
    from rag.retriever import retrieve

    try:
        chunks = retrieve(query, top_k=top_k)
    except Exception as e:
        return {
            "answer": (
                f"❌ Could not connect to Endee vector database.\n\n"
                f"Error: {e}\n\n"
                "Please start Endee with Docker:\n"
                "```\ndocker run -p 8080:8080 endeeio/endee-server:latest\n```"
            ),
            "sources": [],
            "model": "error",
            "chunks": [],
        }

    result = generate(query, chunks)
    result["chunks"] = chunks
    return result


if __name__ == "__main__":
    q = "What is the current state of Bitcoin price?"
    print(f"Query: {q}")
    r = ask(q)
    print(f"\nAnswer:\n{r['answer']}")
    print(f"\nSources:")
    for s in r["sources"]:
        print(f"  - {s}")
