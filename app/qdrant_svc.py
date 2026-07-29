"""
Qdrant service layer.

Phase 1: Recommend API (BEST_SCORE with positive/negative IDs)
Phase 2a: Vector search using EWMA profile embeddings
Phase 2b: Multi-interest prefetch + RRF fusion (multiple ANN queries in one call)

The collection is 'arxiv_bgem3_dense' with integer point IDs and 1024-dim BGE-M3 vectors.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
    RecommendStrategy,
    RecommendQuery,
    RecommendInput,
    Prefetch,
    FusionQuery,
    Fusion,
    SearchParams,
    QuantizationSearchParams,
)

from app import config, db


# ── Quantization search params ───────────────────────────────────────────────
#
# The collection has binary quantization with always_ram, so the 1-bit codes
# live in memory while the float16 vectors (3.27 GB) stay on disk. Rescoring
# reads those vectors back, and the number read scales with `oversampling` —
# which is where the latency was going: the cumulative vector-IO counter on the
# cluster stood at 56.7 GB.
#
# Sending no params at all is NOT the same as sending oversampling=1.0.
# Measured on the live collection, 3 real paper vectors, recall@10 against
# exact (un-quantized) search as ground truth:
#
#   no params (previous behaviour)      2810 ms    100% recall
#   rescore=False                        615 ms     57% recall   <- unusable
#   rescore=True, oversampling=1.0       608 ms    100% recall   <- chosen
#   rescore=True, oversampling=2.0       783 ms    100% recall
#
# So 4.6x faster at identical recall. Rescore stays ON deliberately: binary
# codes alone lose 43% of the correct results, because 1024 dims compressed to
# 1024 bits is far too lossy to rank on directly.
_SEARCH_PARAMS = SearchParams(
    quantization=QuantizationSearchParams(rescore=True, oversampling=1.0),
)

# ── Client (sync, thread-safe, reused across requests) ───────────────────────

@lru_cache(maxsize=1)
def _client() -> QdrantClient:
    return QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        timeout=30,
        check_compatibility=False,
    )


# ── ID lookup ─────────────────────────────────────────────────────────────────

async def lookup_qdrant_ids(arxiv_ids: list[str]) -> dict[str, int]:
    """
    Return {arxiv_id: qdrant_point_id} for every id that exists in the
    collection.  Checks the local SQLite cache first; fetches missing ones
    via Qdrant payload filter (requires the arxiv_id keyword index).
    """
    if not arxiv_ids:
        return {}

    # 1. Pull what we already know from SQLite
    cached = await db.get_qdrant_ids_batch(arxiv_ids)
    missing = [aid for aid in arxiv_ids if aid not in cached]

    if missing:
        # 2. Ask Qdrant: filter by arxiv_id
        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                None,
                _scroll_by_arxiv_ids,
                missing,
            )
        except Exception:
            results = {}

        # 3. Persist new mappings
        for arxiv_id, point_id in results.items():
            await db.save_qdrant_id(arxiv_id, point_id)
            cached[arxiv_id] = point_id

    return cached


def _scroll_by_arxiv_ids(arxiv_ids: list[str]) -> dict[str, int]:
    """
    Sync helper: scroll Qdrant filtering by arxiv_id payload.
    Requires the keyword index created during setup.
    """
    client = _client()
    pts, _ = client.scroll(
        collection_name=config.QDRANT_COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="arxiv_id", match=MatchAny(any=arxiv_ids))]
        ),
        limit=len(arxiv_ids),
        with_payload=True,
        with_vectors=False,
    )
    return {p.payload["arxiv_id"]: p.id for p in pts}


# ── Recommend ─────────────────────────────────────────────────────────────────

async def recommend(
    positive_arxiv_ids: list[str],
    negative_arxiv_ids: list[str],
    seen_arxiv_ids: set[str],
    limit: int = config.REC_LIMIT,
) -> list[str]:
    """
    Call Qdrant Recommend API.

    Returns a list of arxiv_ids (up to `limit`) sorted by Qdrant score,
    excluding papers the user has already seen.
    """
    # Translate arxiv_ids → integer point IDs
    all_ids = list(dict.fromkeys(positive_arxiv_ids + negative_arxiv_ids))
    id_map = await lookup_qdrant_ids(all_ids)

    pos_ids = [id_map[aid] for aid in positive_arxiv_ids if aid in id_map]
    neg_ids = [id_map[aid] for aid in negative_arxiv_ids if aid in id_map]

    if not pos_ids:
        return []

    # Build must-not filter: exclude already-seen papers
    # We can only filter on payload fields — seen list applied in Python
    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(
            None,
            _run_recommend,
            pos_ids,
            neg_ids,
            limit * 2,   # fetch extra so we can filter seen in Python
        )
    except Exception as e:
        # Log and return empty rather than crashing the page
        print(f"[qdrant_svc] recommend error: {e}")
        return []

    # Filter out seen papers, return top `limit`
    filtered = [
        r.payload["arxiv_id"]
        for r in results
        if r.payload.get("arxiv_id") and r.payload["arxiv_id"] not in seen_arxiv_ids
    ]
    return filtered[:limit]


def _run_recommend(
    pos_ids: list[int],
    neg_ids: list[int],
    limit: int,
) -> list:
    """Sync helper — uses query_points with RecommendQuery (modern API)."""
    client = _client()
    result = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=RecommendQuery(
            recommend=RecommendInput(
                positive=pos_ids,
                negative=neg_ids if neg_ids else [],
                strategy=RecommendStrategy.BEST_SCORE,
            )
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False,
        search_params=_SEARCH_PARAMS,
    )
    return result.points


# ── Phase 2a: Vector retrieval + vector search ───────────────────────────────
#
# In-process LRU vector cache.
# Profiling showed Qdrant Cloud free tier reads candidate vectors from
# disk on every retrieve(), which dominated Tier 1 latency (9-18s for
# 120 vectors). Vectors are 1024 floats = 4KB each. A 25K cap = ~100MB
# RAM ceiling. Same papers appear across users' candidate sets (Zipf),
# so steady-state hit rate is high.
#
# Vectors don't change once uploaded, so no TTL.

_VECTOR_CACHE: "OrderedDict[str, list[float]]" = OrderedDict()
_VECTOR_CACHE_MAX = 25_000


def _vec_cache_get(arxiv_id: str) -> list[float] | None:
    val = _VECTOR_CACHE.get(arxiv_id)
    if val is not None:
        _VECTOR_CACHE.move_to_end(arxiv_id)
    return val


def _vec_cache_put(arxiv_id: str, vec: list[float]) -> None:
    if arxiv_id in _VECTOR_CACHE:
        _VECTOR_CACHE.move_to_end(arxiv_id)
        _VECTOR_CACHE[arxiv_id] = vec
        return
    _VECTOR_CACHE[arxiv_id] = vec
    if len(_VECTOR_CACHE) > _VECTOR_CACHE_MAX:
        _VECTOR_CACHE.popitem(last=False)


def vector_cache_stats() -> dict:
    return {"size": len(_VECTOR_CACHE), "max": _VECTOR_CACHE_MAX}


async def get_paper_vectors(arxiv_ids: list[str]) -> dict[str, list[float]]:
    """
    Fetch BGE-M3 embedding vectors for papers from Qdrant.
    Returns {arxiv_id: vector_list} for papers found.

    Cached in-process by arxiv_id; only un-cached IDs hit Qdrant. The
    Qdrant retrieve() that pulls the actual stored vectors is the
    single most expensive call in the pipeline (BQ -> disk read), so
    absorbing repeats here is a big win.

    Used by:
      - EWMA profile updates on save (events.py)
      - Cluster medoid embedding load (recommendations.py)
      - Tier 1 candidate vector fetch (recommendations.py, ~120 IDs)
    """
    if not arxiv_ids:
        return {}

    # Cache check first — pull anything we already know.
    result: dict[str, list[float]] = {}
    misses: list[str] = []
    for aid in arxiv_ids:
        cached = _vec_cache_get(aid)
        if cached is not None:
            result[aid] = cached
        else:
            misses.append(aid)

    if not misses:
        return result

    id_map = await lookup_qdrant_ids(misses)
    if not id_map:
        return result

    point_ids = list(id_map.values())
    arxiv_by_point = {v: k for k, v in id_map.items()}

    loop = asyncio.get_event_loop()
    try:
        points = await loop.run_in_executor(
            None, _get_vectors_by_ids, point_ids
        )
    except Exception as e:
        print(f"[qdrant_svc] get_paper_vectors error: {e}")
        return result

    for p in points:
        aid = p.payload.get("arxiv_id") or arxiv_by_point.get(p.id)
        if aid and p.vector:
            # p.vector may be a dict if named vectors are used
            vec = p.vector if isinstance(p.vector, list) else p.vector.get("dense", p.vector)
            if isinstance(vec, list):
                result[aid] = vec
                _vec_cache_put(aid, vec)
    return result


def _get_vectors_by_ids(point_ids: list[int]) -> list:
    """Sync helper: retrieve points with their vectors."""
    client = _client()
    points = client.retrieve(
        collection_name=config.QDRANT_COLLECTION,
        ids=point_ids,
        with_payload=True,
        with_vectors=True,
    )
    return points


async def search_by_vector(
    query_vector: list[float],
    limit: int = 20,
    exclude_ids: set[str] | None = None,
) -> list[str]:
    """
    Raw vector search — find papers similar to a given embedding.
    Returns list of arxiv_ids, excluding any in exclude_ids.

    Used when EWMA profile vectors are available (Phase 2a+).
    """
    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(
            None, _run_vector_search, query_vector, (limit * 2) if exclude_ids else limit,
        )
    except Exception as e:
        print(f"[qdrant_svc] search_by_vector error: {e}")
        return []

    exclude = exclude_ids or set()
    filtered = [
        r.payload["arxiv_id"]
        for r in results
        if r.payload.get("arxiv_id") and r.payload["arxiv_id"] not in exclude
    ]
    return filtered[:limit]


async def search_by_vector_with_scores(
    query_vector: list[float],
    limit: int = 20,
    exclude_ids: set[str] | None = None,
    with_vectors: bool = False,
) -> list[dict]:
    """
    Vector search returning both arxiv_ids AND cosine scores.

    Returns list of {'arxiv_id': str, 'score': float} dicts sorted by
    score desc, excluding any in exclude_ids.

    If `with_vectors=True`, each dict also has a 'vector' key holding the
    1024-dim BGE-M3 embedding. Returning vectors in the search response
    avoids a separate `client.retrieve()` round-trip later — that retrieve
    was ~9-18s on cold candidates because BQ rescore reads from disk.
    """
    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(
            None, _run_vector_search, query_vector,
            (limit * 2) if exclude_ids else limit,
            with_vectors,
        )
    except Exception as e:
        print(f"[qdrant_svc] search_by_vector_with_scores error: {e}")
        return []

    exclude = exclude_ids or set()
    out: list[dict] = []
    for r in results:
        aid = r.payload.get("arxiv_id")
        if not aid or aid in exclude:
            continue
        item = {"arxiv_id": aid, "score": float(r.score)}
        if with_vectors and r.vector:
            # Named vectors return a dict; unnamed returns a list.
            vec = r.vector if isinstance(r.vector, list) else r.vector.get("dense", r.vector)
            if isinstance(vec, list):
                item["vector"] = vec
        out.append(item)
        if len(out) >= limit:
            break
    return out


def _run_vector_search(
    query_vector: list[float], limit: int, with_vectors: bool = False,
) -> list:
    """Sync helper: nearest-neighbour search by vector."""
    client = _client()
    result = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
        with_payload=True,
        with_vectors=with_vectors,
        search_params=_SEARCH_PARAMS,
    )
    return result.points


# ── Phase 3: Dense search for hybrid search pipeline ─────────────────────────

async def search_dense(
    dense_vec: list[float],
    limit: int = 50,
) -> list[dict]:
    """
    ANN dense search for the hybrid search pipeline (Phase 3).

    Returns list of {'arxiv_id': str, 'score': float} dicts sorted by
    score desc.  Different from search_by_vector() which returns only
    arxiv_ids — this version returns scores needed for RRF fusion.
    """
    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(
            None, _run_dense_search, dense_vec, limit,
        )
    except Exception as e:
        print(f"[qdrant_svc] search_dense error: {e}")
        return []

    return [
        {"arxiv_id": r.payload["arxiv_id"], "score": r.score}
        for r in results
        if r.payload.get("arxiv_id")
    ]


def _run_dense_search(query_vector: list[float], limit: int) -> list:
    """Sync helper: ANN search returning scored results for RRF."""
    client = _client()
    result = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
        with_payload=True,
        with_vectors=False,
        search_params=_SEARCH_PARAMS,
    )
    return result.points


# ── Phase 2b: Multi-interest prefetch + RRF fusion ───────────────────────────

async def multi_interest_search(
    interest_vectors: list[tuple[list[float], int]],
    short_term_vector: list[float] | None = None,
    exclude_ids: set[str] | None = None,
    total_limit: int = 100,
) -> list[str]:
    """
    Multi-interest retrieval using Qdrant prefetch + RRF fusion.

    Sends multiple ANN queries (one per interest cluster + optional session
    vector) in a SINGLE API call.  Qdrant runs them in parallel server-side
    and fuses results via Reciprocal Rank Fusion (k=60).

    Args:
        interest_vectors: list of (medoid_embedding, per_cluster_limit) tuples,
                          ordered by importance (highest first)
        short_term_vector: optional EWMA short-term session embedding
        exclude_ids: arxiv_ids to filter out (already seen)
        total_limit: how many candidates to return after fusion

    Returns:
        list of arxiv_ids sorted by fused relevance

    Latency: ~15-25ms for 3-4 prefetch queries (single network round-trip)
    """
    # Build prefetch queries — one per interest cluster
    prefetches = []
    for vec, limit in interest_vectors:
        prefetches.append(Prefetch(
            query=vec,
            limit=limit,
        ))

    # Add short-term session vector if available
    if short_term_vector is not None:
        prefetches.append(Prefetch(
            query=short_term_vector,
            limit=25,
        ))

    if not prefetches:
        return []

    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(
            None,
            _run_prefetch_rrf,
            prefetches,
            total_limit * 2 if exclude_ids else total_limit,
        )
    except Exception as e:
        print(f"[qdrant_svc] multi_interest_search error: {e}")
        return []

    exclude = exclude_ids or set()
    filtered = [
        r.payload["arxiv_id"]
        for r in results
        if r.payload.get("arxiv_id") and r.payload["arxiv_id"] not in exclude
    ]
    return filtered[:total_limit]


def _run_prefetch_rrf(prefetches: list[Prefetch], limit: int) -> list:
    """Sync helper: execute prefetch queries with RRF fusion."""
    client = _client()
    result = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        prefetch=prefetches,
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        with_payload=True,
        with_vectors=False,
        search_params=_SEARCH_PARAMS,
    )
    return result.points
