"""
RAG Retriever: embeds user query and searches Endee for top-K chunks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

TOP_K = int(os.getenv("TOP_K", "5"))


def retrieve(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Embed query → search Endee → return ranked chunks.

    Returns list of:
      { id, similarity, text, source, date, title, chunk_index }
    """
    from embeddings.embedder import embed_text
    from vector_store.endee_client import search

    query_vec = embed_text(query)
    results = search(query_vec, top_k=top_k)
    return results


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a readable context block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source_label = f"{chunk.get('source', 'Unknown')} — {chunk.get('date', '')}"
        parts.append(
            f"[{i}] Source: {source_label}\n"
            f"Title: {chunk.get('title', '')}\n"
            f"Content: {chunk.get('text', '')}"
        )
    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    query = "What is happening with Bitcoin price?"
    print(f"[Retriever] Query: {query}")
    results = retrieve(query, top_k=3)
    for r in results:
        print(f"\n  [{r['similarity']:.3f}] {r['source']} | {r['title']}")
        print(f"  {r['text'][:120]}...")
