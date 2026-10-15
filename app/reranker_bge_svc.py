"""
Cross-Encoder Reranker service singleton.

Responsibilities:
  - Load config.RERANKER_MODEL once (lazily or eagerly via get_reranker())
  - rerank(query, papers) -> (sorted_papers, compute_time_ms)
  - CPU float32, graceful fallback if model loading or computation fails
  - Thread-safe (model is read-only after load)

Model notes:
  - Default mixedbread-ai/mxbai-rerank-xsmall-v1 (70M, DeBERTa-v3), read at
    RERANKER_MAX_LENGTH=256 tokens. Previously ms-marco-MiniLM-L-6-v2 (22M).
  - Scores are always returned as raw LOGITS. Some cross-encoders (mxbai,
    gte-reranker) ship a Sigmoid activation and return probabilities; the
    search pipeline applies its own sigmoid, and a second one squeezes every
    score into ~0.5-0.73, where the title/citation boosts overwhelm the
    reranker. Measured: with the double sigmoid mxbai looked WORSE than
    MiniLM. predict(activation_fn=Identity) gives logits for every model.
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

        from app import compute_pool
        compute_pool.configure_torch()
        _reranker = CrossEncoder(
            config.RERANKER_MODEL,
            max_length=config.RERANKER_MAX_LENGTH,
            device=config.BGE_M3_DEVICE,
            revision=config.RERANKER_REVISION,
        )
        print("[reranker_svc] Cross-encoder reranker model loaded successfully")
        return _reranker


def _identity():
    """Identity activation, so predict() returns logits. None without torch,
    which only happens where no cross-encoder can be loaded anyway."""
    try:
        import torch
    except ImportError:
        return None
    return torch.nn.Identity()


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

        # Inference — CrossEncoder.predict() returns numpy array of scores.
        # Identity activation: logits for every model (see module docstring).
        scores = model.predict(pairs, show_progress_bar=False,
                               batch_size=max(1, len(pairs)),
                               activation_fn=_identity())

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
