"""
Endee vector store client wrapper.
Uses the official Endee Python SDK with CORRECTLY mapped field names.

Key SDK facts (from source inspection):
  - create_index payload: index_name, dim, M, ef_con, checksum, precision, version
  - get_index endpoint:   /index/{name}/info
  - upsert endpoint:      /index/{name}/vector/insert  (msgpack binary)
  - search endpoint:      /index/{name}/search
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

ENDEE_URL     = os.getenv("ENDEE_URL", "http://localhost:8080/api/v1")
INDEX_NAME    = os.getenv("ENDEE_INDEX_NAME", "crypto_news")
AUTH_TOKEN    = os.getenv("ENDEE_AUTH_TOKEN", "")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))


def _new_client():
    """Return a fresh Endee SDK client pointed at local server."""
    from endee import Endee
    c = Endee()           # token=None → open mode
    c.set_base_url(ENDEE_URL)
    return c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def health_check() -> bool:
    try:
        r = requests.get(f"{ENDEE_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def index_exists() -> bool:
    try:
        c = _new_client()
        indexes = c.list_indexes()   # returns list or dict
        items = indexes if isinstance(indexes, list) else indexes.get("indexes", [])
        for item in items:
            name = item.get("name", item) if isinstance(item, dict) else item
            if name == INDEX_NAME:
                return True
    except Exception:
        pass
    return False


def create_index() -> bool:
    """Create the Endee index using the SDK (which uses the correct field names)."""
    from endee import Endee, Precision

    if index_exists():
        print(f"[EndeeClient] Index '{INDEX_NAME}' already exists.")
        return True

    try:
        c = _new_client()
        # SDK create_index signature:
        #   name, dimension, space_type, M=16, ef_con=128, precision=Precision.INT16,
        #   version=None, sparse_dim=0
        c.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            space_type="cosine",
            precision=Precision.FLOAT32,   # INT8/INT16 rejected by endeeio/endee-server:latest
        )
        print(f"[EndeeClient] ✓ Index '{INDEX_NAME}' created.")
        return True
    except Exception as e:
        err = str(e)
        if "already exists" in err.lower() or "exist" in err.lower():
            print(f"[EndeeClient] Index '{INDEX_NAME}' already exists.")
            return True
        print(f"[EndeeClient] ✗ create_index failed: {err}")
        return False


def get_index():
    """
    Returns an Index object. Calls /index/{name}/info via the SDK.
    NOTE: SDK caches this with @lru_cache — create a new client each time
    to avoid stale cache between calls.
    """
    c = _new_client()
    return c.get_index(INDEX_NAME)


# ---------------------------------------------------------------------------
# Upsert — uses SDK index.upsert() which handles msgpack serialization
# ---------------------------------------------------------------------------

def upsert_chunks(chunks: List[Dict[str, Any]]) -> int:
    """
    Upsert chunks via the Endee SDK in sub-batches (max 1000 per call).
    Each chunk must have: id (str), vector (List[float]), meta (dict).
    """
    MAX_BATCH = 500   # safe batch size

    try:
        index = get_index()
    except Exception as e:
        print(f"[EndeeClient] ✗ get_index failed: {e}")
        return 0

    total = 0
    for i in range(0, len(chunks), MAX_BATCH):
        batch = chunks[i : i + MAX_BATCH]
        items = [
            {
                "id":     c["id"],
                "vector": c["vector"],
                "meta":   {str(k): str(v) for k, v in c.get("meta", {}).items()},
            }
            for c in batch
        ]
        try:
            index.upsert(items)
            total += len(items)
            print(f"  [Upsert] {total}/{len(chunks)} chunks done ({i + len(batch)}/{len(chunks)})")
        except Exception as e:
            print(f"  [Upsert] Batch starting at {i} failed: {e}")

    print(f"[EndeeClient] ✓ Upserted {total} total chunks.")
    return total


# ---------------------------------------------------------------------------
# Search — uses SDK index.query()
# ---------------------------------------------------------------------------

def search(query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Semantic search via Endee SDK.
    Returns: [{ id, similarity, text, source, date, title, chunk_index }]
    """
    try:
        index = get_index()
        results = index.query(vector=query_vector, top_k=top_k)
        output = []
        for r in results:
            meta = r.get("meta", {}) if isinstance(r, dict) else {}
            output.append({
                "id":          r.get("id", "")          if isinstance(r, dict) else "",
                "similarity":  r.get("similarity", 0.0) if isinstance(r, dict) else 0.0,
                "text":        meta.get("text", ""),
                "source":      meta.get("source", "Unknown"),
                "date":        meta.get("date", ""),
                "title":       meta.get("title", ""),
                "chunk_index": meta.get("chunk_index", 0),
            })
        return output
    except Exception as e:
        print(f"[EndeeClient] ✗ Search failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def get_index_stats() -> Optional[Dict[str, Any]]:
    try:
        return get_index().describe()
    except Exception:
        return None


if __name__ == "__main__":
    print("[Test] Health:", health_check())
    print("[Test] Index exists:", index_exists())
    if not index_exists():
        create_index()
    print("[Test] Index exists now:", index_exists())
