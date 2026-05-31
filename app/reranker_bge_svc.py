"""
Cross-Encoder Reranker service singleton.

Responsibilities:
  - Load cross-encoder/ms-marco-MiniLM-L-6-v2 once (lazily or eagerly via get_reranker())
  - rerank(query, papers) -> (sorted_papers, compute_time_ms)
  - CPU float32, graceful fallback if model loading or computation fails
  - Thread-safe (model is read-only after load)

Model notes:
  - MiniLM-L-6 is a 22M-param cross-encoder (~20x lighter than bge-reranker-v2-m3)
  - Achieves ~200-400ms for 10 pairs on CPU (vs ~7s for BGE-reranker)
  - Trained on MS MARCO passage ranking — generalizes well to academic search
  - Scores are logits: higher = more relevant (no sigmoid needed for ranking)
"""
from __future__ import annotations

import threading
import time
from app import config

# ── Module-level singleton ────────────────────────────────────────────────────

_reranker = None
_reranker_lock = threading.Lock()


def get_reranker():
    """
    Return the cross-encoder reranker singleton. Thread-safe, loads once.
    Called eagerly in main.py lifespan so the first request doesn't pay the load cost.
    """
    global _reranker
    if _reranker is not None:
        return _reranker

    with _reranker_lock:
        # Double-check after acquiring lock
        if _reranker is not None:
            return _reranker

        from sentence_transformers import CrossEncoder

        print(f"[reranker_svc] Loading {config.RERANKER_MODEL} on {config.BGE_M3_DEVICE}...")

        _reranker = CrossEncoder(
            config.RERANKER_MODEL,
            max_length=512,
            device=config.BGE_M3_DEVICE,
        )
        print("[reranker_svc] Cross-encoder reranker model loaded successfully")
        return _reranker


def rerank(
    query: str,
    papers: list[dict],
    top_n: int = config.SEARCH_RERANK_TOP_N,
) -> tuple[list[dict], int]:
    """
    Rerank candidates using Cross-Encoder.

    Args:
        query: User's search query (raw text).
        papers: List of paper dicts with 'title' and 'abstract' keys.
        top_n: Capping limit for reranking. Rerank only the top_n elements.

    Returns:
        (sorted_papers, compute_time_ms) where:
          sorted_papers: Reranked papers (top_n reranked first, rest appended).
          compute_time_ms: Time taken in milliseconds.
    """
    if not papers or not query.strip():
        return papers, 0

    # Separate candidates to rerank from the rest to keep latency low on CPU
    to_rerank = papers[:top_n]
    rest = papers[top_n:]

    start_time = time.perf_counter()
    try:
        model = get_reranker()
        
        # Prepare [query, passage] pairs
        pairs = []
        for p in to_rerank:
            title = p.get("title") or ""
            abstract = p.get("abstract") or ""
            # Cross-encoder formats query and document text clearly
            doc_text = f"Title: {title}\nAbstract: {abstract}"
            pairs.append([query, doc_text])

        # Inference — CrossEncoder.predict() returns numpy array of scores
        scores = model.predict(pairs, show_progress_bar=False)

        # Normalize to list of floats
        if hasattr(scores, 'tolist'):
            scores = scores.tolist()
        if isinstance(scores, (float, int)):
            scores = [float(scores)]

        # Attach score and sort
        for p, score in zip(to_rerank, scores):
            p["bge_rerank_score"] = float(score)

        to_rerank.sort(key=lambda x: x.get("bge_rerank_score", -9999.0), reverse=True)

    except Exception as e:
        print(f"[reranker_svc] WARNING: Reranking failed: {e} -- falling back to input order")
        # Keep original scores / order but mark as fallback
        for p in to_rerank:
            if "bge_rerank_score" not in p:
                p["bge_rerank_score"] = 0.0

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    print(f"[reranker_svc] Reranked {len(to_rerank)} candidates in {elapsed_ms}ms")

    # Combine back the reranked part and the remaining unranked papers
    return to_rerank + rest, elapsed_ms
