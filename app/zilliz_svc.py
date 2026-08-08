"""
Zilliz Cloud sparse search client — Phase 3.

Responsibilities:
  - Connect to Zilliz Cloud serverless via pymilvus MilvusClient
  - search_sparse(sparse_dict, limit) → list[dict] with arxiv_id + score
  - Handle gRPC reconnects on closed-channel errors
  - Collection: arxiv_bgem3_sparse
  - Schema: id (INT64 auto PK), arxiv_id (VARCHAR), sparse_vector (SPARSE_FLOAT_VECTOR)
  - Index: SPARSE_INVERTED_INDEX, metric_type=IP
"""
from __future__ import annotations

import asyncio
import threading
from functools import lru_cache

from app import config

# ── Client singleton ─────────────────────────────────────────────────────────

_client = None
_client_lock = threading.Lock()


def _get_client():
    """Return or create the MilvusClient singleton.  Thread-safe."""
    global _client
    if _client is not None:
        return _client

    with _client_lock:
        if _client is not None:
            return _client

        from pymilvus import MilvusClient

        _client = MilvusClient(
            uri=config.ZILLIZ_URI,
            token=config.ZILLIZ_TOKEN,
        )
        print(f"[zilliz_svc] Connected to {config.ZILLIZ_COLLECTION}")
        return _client


def _reset_client():
    """Force reconnect on next call.  Used after gRPC errors."""
    global _client
    with _client_lock:
        _client = None


# ── Sparse search ────────────────────────────────────────────────────────────

def _run_sparse_search(
    sparse_dict: dict[int, float],
    limit: int,
) -> list[dict]:
    """
    Sync helper: execute sparse vector search on Zilliz.

    Args:
        sparse_dict: {token_id_int: weight_float} from BGE-M3 lexical_weights
        limit: max results to return

    Returns:
        list of {'arxiv_id': str, 'score': float} dicts, sorted by score desc
    """
    client = _get_client()

    results = client.search(
        collection_name=config.ZILLIZ_COLLECTION,
        data=[sparse_dict],
        anns_field="sparse_vector",
        search_params={"metric_type": "IP"},
        limit=limit,
        output_fields=["arxiv_id"],
    )

    # pymilvus returns list[list[dict]] — first list is for first query vector
    if not results or not results[0]:
        return []

    return [
        {"arxiv_id": hit["entity"]["arxiv_id"], "score": hit["distance"]}
        for hit in results[0]
        if hit.get("entity", {}).get("arxiv_id")
    ]


async def search_sparse(
    sparse_dict: dict[int, float],
    limit: int = 50,
) -> list[dict]:
    """
    Async sparse search — runs the sync MilvusClient in a thread executor.

    Args:
        sparse_dict: BGE-M3 lexical weights {int_token_id: float_weight}
        limit: max results

    Returns:
        list of {'arxiv_id': str, 'score': float} sorted by score desc.
        Returns empty list on error (graceful degradation).
    """
    if not sparse_dict:
        return []

    loop = asyncio.get_event_loop()

    try:
        results = await loop.run_in_executor(
            None, _run_sparse_search, sparse_dict, limit
        )
        return results
    except Exception as e:
        error_msg = str(e).lower()
        # Retry once on gRPC channel closed errors
        if "closed" in error_msg or "unavailable" in error_msg or "connect" in error_msg:
            print(f"[zilliz_svc] Connection error, retrying: {e}")
            _reset_client()
            try:
                results = await loop.run_in_executor(
                    None, _run_sparse_search, sparse_dict, limit
                )
                return results
            except Exception as e2:
                print(f"[zilliz_svc] Retry failed: {e2}")
                return []
        else:
            print(f"[zilliz_svc] search_sparse error: {e}")
            return []
