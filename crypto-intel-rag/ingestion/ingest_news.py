"""
Document ingestion pipeline.
Loads raw JSON news articles, chunks them with overlapping windows,
embeds each chunk, and upserts into Endee in batches.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

DATA_RAW_DIR      = Path(__file__).parent.parent / "data" / "raw"
DATA_PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

CHUNK_SIZE   = int(os.getenv("CHUNK_SIZE", "128"))    # words per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "20")) # overlap words
MAX_ARTICLES = int(os.getenv("MAX_ARTICLES", "500"))  # cap for demo speed
UPSERT_BATCH = int(os.getenv("UPSERT_BATCH", "50"))   # upsert batch size


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(text.split())


def recursive_chunk(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


# ---------------------------------------------------------------------------
# Load + chunk
# ---------------------------------------------------------------------------

def load_articles(path: Path, max_articles: int = MAX_ARTICLES) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")
    articles = data[:max_articles]
    print(f"[Ingest] Loaded {len(articles)} articles (cap={max_articles}, total={len(data)})")
    return articles


def build_chunks(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert articles into chunk dicts with metadata."""
    all_chunks = []
    for article in articles:
        # Support various field name conventions
        content = clean_text(
            article.get("content") or article.get("text") or article.get("body") or ""
        )
        if not content:
            continue

        source = (
            article.get("source") or article.get("publisher") or
            article.get("author") or "Unknown"
        )
        title   = article.get("title", "No title")
        date    = article.get("date") or article.get("published_at") or "Unknown"
        cat     = article.get("category", "")
        art_id  = str(article.get("id", f"article_{len(all_chunks)}"))

        raw_chunks = recursive_chunk(content)
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{art_id}_chunk_{idx}"
            all_chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "meta": {
                    "article_id": art_id,
                    "title": title,
                    "source": source,
                    "date": date,
                    "category": cat,
                    "chunk_index": idx,
                    "token_count": _word_count(chunk_text),
                    "text": chunk_text,
                },
            })
    print(f"[Ingest] Built {len(all_chunks)} chunks from {len(articles)} articles")
    return all_chunks


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------

def embed_chunks(chunks: List[Dict[str, Any]], batch_size: int = 64) -> List[Dict[str, Any]]:
    """Embed all chunks in batches."""
    from embeddings.embedder import embed_batch

    texts = [c["text"] for c in chunks]
    print(f"[Ingest] Embedding {len(texts)} chunks (batch={batch_size})...")
    vectors = embed_batch(texts, batch_size=batch_size)
    for chunk, vec in zip(chunks, vectors):
        chunk["vector"] = vec
    return chunks


def save_processed(chunks: List[Dict[str, Any]]) -> None:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED_DIR / "chunks.json"
    saveable = [{k: v for k, v in c.items() if k != "vector"} for c in chunks]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(saveable, f, indent=2, ensure_ascii=False)
    print(f"[Ingest] ✓ Saved metadata → {out_path}")


# ---------------------------------------------------------------------------
# Upsert in batches
# ---------------------------------------------------------------------------

def upsert_in_batches(chunks: List[Dict[str, Any]], batch_size: int = UPSERT_BATCH) -> int:
    """Upsert chunks into Endee using the stable Endee Python SDK client."""
    from vector_store.endee_client import upsert_chunks
    return upsert_chunks(chunks)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_ingestion(
    max_articles: int = MAX_ARTICLES,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Full ingestion pipeline: load → chunk → embed → upsert.
    progress_callback(step, total, message) for UI updates.
    """
    from vector_store.endee_client import create_index, health_check

    steps  = 5
    result = {"status": "error", "chunks": 0, "articles": 0, "error": None}

    # Step 1: Health check
    if progress_callback:
        progress_callback(1, steps, "Checking Endee connection...")
    if not health_check():
        result["error"] = (
            "Cannot reach Endee at "
            + os.getenv("ENDEE_URL", "http://localhost:8080/api/v1")
            + "\n\nStart Endee with:\n"
            "  docker run -d -p 8080:8080 endeeio/endee-server:latest"
        )
        return result

    # Step 2: Create index
    if progress_callback:
        progress_callback(2, steps, "Creating Endee index...")
    if not create_index():
        result["error"] = "Failed to create Endee index."
        return result

    # Step 3: Load + chunk
    raw_path = DATA_RAW_DIR / "crypto_news.json"
    if not raw_path.exists():
        result["error"] = f"Data file not found: {raw_path}"
        return result

    if progress_callback:
        progress_callback(3, steps, f"Loading & chunking articles (cap={max_articles})...")
    articles = load_articles(raw_path, max_articles=max_articles)
    chunks   = build_chunks(articles)
    result["articles"] = len(articles)

    # Step 4: Embed
    if progress_callback:
        progress_callback(4, steps, f"Embedding {len(chunks)} chunks (this takes a minute)...")
    chunks = embed_chunks(chunks)
    save_processed(chunks)

    # Step 5: Upsert
    if progress_callback:
        progress_callback(5, steps, f"Upserting {len(chunks)} chunks into Endee...")
    n = upsert_in_batches(chunks)
    result["chunks"]  = n
    result["status"]  = "success" if n > 0 else "partial"
    print(f"[Ingest] ✓ Done. {len(articles)} articles → {n} chunks in Endee.")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest crypto news into Endee")
    parser.add_argument("--max", type=int, default=MAX_ARTICLES,
                        help=f"Max articles to ingest (default: {MAX_ARTICLES})")
    args = parser.parse_args()

    print(f"[Ingest] Starting pipeline (max_articles={args.max})")
    run_ingestion(
        max_articles=args.max,
        progress_callback=lambda s, t, m: print(f"  [{s}/{t}] {m}"),
    )
