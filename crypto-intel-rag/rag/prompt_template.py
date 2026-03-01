"""
Prompt templates for the Crypto RAG system.
Enforces grounding: model must cite only retrieved context.
"""

from __future__ import annotations

from typing import Any, Dict, List


SYSTEM_PROMPT = """You are CryptoInsight, an expert cryptocurrency analyst AI assistant.
You answer questions STRICTLY based on the provided context documents.

Rules:
1. Only use information from the provided context. Do NOT use external knowledge.
2. If the answer is not found in the context, respond with:
   "Not found in knowledge base. Please try rephrasing, or ingest more documents."
3. Always cite your sources as [Source: <source_name>] inline in your answer.
4. Be concise, clear, and analytical — like a professional crypto research report.
5. Never speculate beyond what the documents say.
"""


def build_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Build the user message with query + context for the LLM."""
    if not chunks:
        return (
            f"Question: {query}\n\n"
            "Context: [No relevant documents found in the knowledge base]"
        )

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "Unknown")
        date = chunk.get("date", "")
        title = chunk.get("title", "No title")
        text = chunk.get("text", "")
        context_parts.append(
            f"[Document {i}]\n"
            f"Source: {source} ({date})\n"
            f"Title: {title}\n"
            f"Content: {text}"
        )

    context_block = "\n\n" + "\n\n---\n\n".join(context_parts) + "\n"

    return (
        f"Context Documents:\n{context_block}\n"
        f"Question: {query}\n\n"
        "Please answer the question using ONLY the information in the context above. "
        "Cite sources as [Source: <source_name>] inline."
    )


def build_sources_list(chunks: List[Dict[str, Any]]) -> List[str]:
    """Build a deduplicated list of source citation strings."""
    seen = set()
    sources = []
    for chunk in chunks:
        source = chunk.get("source", "Unknown")
        date = chunk.get("date", "")
        title = chunk.get("title", "")
        label = f"{source} ({date}) — {title}" if date else f"{source} — {title}"
        if label not in seen:
            seen.add(label)
            sources.append(label)
    return sources
