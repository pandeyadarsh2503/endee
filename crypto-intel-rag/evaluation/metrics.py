"""
Evaluation metrics for the RAG pipeline:
- Precision@K
- Retrieval Relevance Score (keyword overlap)
- Hallucination detection (basic)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Precision@K: fraction of top-K retrieved items that are relevant.

    Args:
        retrieved_ids: ordered list of retrieved doc IDs
        relevant_ids:  set of ground-truth relevant doc IDs
        k:             cutoff

    Returns:
        float in [0, 1]
    """
    if k == 0 or not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for id_ in top_k if id_ in relevant_set)
    return hits / k


def retrieval_relevance_score(query: str, chunks: List[Dict[str, Any]]) -> float:
    """
    Keyword overlap score: what fraction of query keywords appear in each chunk?
    Returns average score across all chunks.
    """
    if not chunks:
        return 0.0

    query_keywords = set(
        w.lower().strip(".,!?\"'")
        for w in query.split()
        if len(w) > 3
    )
    if not query_keywords:
        return 0.0

    scores = []
    for chunk in chunks:
        text = chunk.get("text", "").lower()
        chunk_words = set(text.split())
        overlap = query_keywords & chunk_words
        scores.append(len(overlap) / len(query_keywords))

    return round(sum(scores) / len(scores), 4)


def hallucination_risk(answer: str, chunks: List[Dict[str, Any]]) -> float:
    """
    Rough hallucination risk estimator.
    Calculates the fraction of noun-like words in the answer
    that are NOT grounded in the retrieved context.

    Returns:
        float 0.0 (fully grounded) → 1.0 (fully ungrounded)
    """
    if not answer or not chunks:
        return 1.0

    # Build a superset of all words in context
    context_text = " ".join(c.get("text", "") for c in chunks).lower()
    context_words = set(context_text.split())

    # Filter answer words that are "significant" (length > 4, not stop words)
    stop_words = {"from", "with", "this", "that", "have", "been", "were",
                  "they", "their", "which", "about", "also", "into"}
    answer_words = [
        w.lower().strip(".,!?\"'()[]")
        for w in answer.split()
        if len(w) > 4 and w.lower() not in stop_words
    ]

    if not answer_words:
        return 0.0

    ungrounded = sum(1 for w in answer_words if w not in context_words)
    return round(ungrounded / len(answer_words), 4)


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    {
        "query": "What happened to Bitcoin price recently?",
        "expected_sources": ["coindesk_btc_2026_02_20"],
    },
    {
        "query": "What is the Ethereum Petra upgrade?",
        "expected_sources": ["ethereum_shapella_upgrade_2026"],
    },
    {
        "query": "How is DeFi performing in terms of TVL?",
        "expected_sources": ["defi_tvl_recovery_2026"],
    },
    {
        "query": "What is the Solana TPS milestone?",
        "expected_sources": ["solana_network_2026_performance"],
    },
    {
        "query": "What are the new US crypto regulations?",
        "expected_sources": ["crypto_regulation_usa_2026"],
    },
]


def run_evaluation(
    top_k: int = 5,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Run the full evaluation suite.
    Returns a dict with per-query results and aggregate metrics.
    """
    from rag.retriever import retrieve

    results = []
    total_queries = len(TEST_QUERIES)

    for i, test in enumerate(TEST_QUERIES):
        if progress_callback:
            progress_callback(i + 1, total_queries, f"Testing: {test['query'][:50]}...")

        try:
            chunks = retrieve(test["query"], top_k=top_k)
            retrieved_ids = [
                c.get("id", "").split("_chunk_")[0] for c in chunks
            ]
            p_at_k = precision_at_k(retrieved_ids, test["expected_sources"], k=top_k)
            relevance = retrieval_relevance_score(test["query"], chunks)

            results.append(
                {
                    "query": test["query"],
                    "precision_at_k": p_at_k,
                    "relevance_score": relevance,
                    "retrieved_count": len(chunks),
                    "top_source": chunks[0]["source"] if chunks else "N/A",
                    "top_similarity": chunks[0]["similarity"] if chunks else 0.0,
                    "status": "ok",
                }
            )
        except Exception as e:
            results.append(
                {
                    "query": test["query"],
                    "precision_at_k": 0.0,
                    "relevance_score": 0.0,
                    "retrieved_count": 0,
                    "top_source": "ERROR",
                    "top_similarity": 0.0,
                    "status": f"error: {e}",
                }
            )

    # Aggregate
    ok_results = [r for r in results if r["status"] == "ok"]
    avg_precision = (
        sum(r["precision_at_k"] for r in ok_results) / len(ok_results)
        if ok_results else 0.0
    )
    avg_relevance = (
        sum(r["relevance_score"] for r in ok_results) / len(ok_results)
        if ok_results else 0.0
    )

    return {
        "results": results,
        "avg_precision_at_k": round(avg_precision, 4),
        "avg_relevance_score": round(avg_relevance, 4),
        "total_queries": total_queries,
        "successful": len(ok_results),
    }


if __name__ == "__main__":
    print("[Evaluation] Running test suite...")
    report = run_evaluation(
        top_k=5,
        progress_callback=lambda s, t, m: print(f"  [{s}/{t}] {m}"),
    )
    print(f"\n{'='*50}")
    print(f"Average Precision@5:    {report['avg_precision_at_k']:.4f}")
    print(f"Average Relevance:      {report['avg_relevance_score']:.4f}")
    print(f"Queries tested:         {report['successful']}/{report['total_queries']}")
    print(f"{'='*50}")
