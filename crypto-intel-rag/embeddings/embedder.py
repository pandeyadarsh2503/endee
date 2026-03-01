"""
Embeddings module using sentence-transformers (free, local, no API key needed).
Model: all-MiniLM-L6-v2 → 384-dimensional cosine embeddings.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Lazily load the model once and cache it."""
    print(f"[Embedder] Loading model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> List[float]:
    """Embed a single text string → list of floats."""
    model = _load_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_batch(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Embed a list of texts in batches → list of float lists."""
    model = _load_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 10,
    )
    return [v.tolist() for v in vectors]


if __name__ == "__main__":
    sample = "Bitcoin surged past $95,000 as institutional demand accelerates."
    vec = embed_text(sample)
    print(f"[✓] Embedding dim: {len(vec)}")
    print(f"[✓] First 5 values: {vec[:5]}")
