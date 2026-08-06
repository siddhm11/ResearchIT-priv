"""
Map-position lookup: the replacement for api.soarxiv.org/locate.

The spatial map's coordinates live in their own Qdrant collection
(`arxiv_map_positions`), loaded from the SoArXiv chunk corpus. The vector *is*
the 3D position; arxiv_id carries a keyword payload index.

Two things this does that the upstream endpoint cannot:

  * Batch. Semantic search resolves up to 100 ids at once, which upstream costs
    100 round trips. A payload filter with MatchAny answers all of them in one.

  * Answer for papers the map never contained. Roughly 10-20% of search results
    are not in the spatial corpus at all -- upstream simply 404s. Here a missing
    paper is placed at the centroid of its nearest neighbours in BGE-M3 space,
    which is a real semantic estimate rather than a guess, and is written back
    so the placement is stable from then on.

Inferred placements are flagged `inferred: true` in the payload and never
silently mixed with real ones: a caller has to be able to tell "this is where
the paper is" from "this is where a paper like it would sit".
"""
from __future__ import annotations

import asyncio
import hashlib

from qdrant_client.models import FieldCondition, Filter, MatchAny, PointStruct

from app import config, qdrant_svc

MAP_COLLECTION = "arxiv_map_positions"

# How many semantic neighbours vote on an inferred position. Small enough that
# one outlier cannot drag the estimate far, large enough to average out the
# projection's local noise.
INFER_NEIGHBOURS = 12
# Ids are allocated above the SoArXiv paper_id range so an inferred point can
# never collide with a real one.
INFERRED_ID_BASE = 1_000_000_000


def _map_client():
    return qdrant_svc._client_for(config.QDRANT_URL, config.QDRANT_API_KEY)


def _row(point) -> dict:
    payload = point.payload or {}
    vector = list(point.vector or [])
    return {
        "arxivId": payload.get("arxiv_id"),
        "position": [float(v) for v in vector] if len(vector) == 3 else None,
        "clusterId": int(payload.get("cluster_id") or 0),
        "tileId": payload.get("tile_id"),
        "title": payload.get("title") or "",
        "source": "inferred" if payload.get("inferred") else "map",
    }


def _fetch_known(arxiv_ids: list[str]) -> dict[str, dict]:
    """One filtered scroll for the whole batch, rather than one call per id."""
    if not arxiv_ids:
        return {}
    client = _map_client()
    found: dict[str, dict] = {}
    # Chunked so a large batch cannot build an unbounded filter.
    for start in range(0, len(arxiv_ids), 128):
        window = arxiv_ids[start:start + 128]
        points, _ = client.scroll(
            collection_name=MAP_COLLECTION,
            scroll_filter=Filter(must=[FieldCondition(key="arxiv_id", match=MatchAny(any=window))]),
            limit=len(window),
            with_payload=True,
            with_vectors=True,
        )
        for point in points:
            row = _row(point)
            if row["arxivId"]:
                found[row["arxivId"]] = row
    return found


async def _infer_one(arxiv_id: str, vector: list[float]) -> dict | None:
    """
    Place a paper the map never contained, using its semantic neighbours.

    Neighbours come from the BGE-M3 collections; their *positions* come from the
    map collection. Averaging those positions puts the paper where papers like
    it already live, which is the only defensible answer -- a random or default
    coordinate would place it beside unrelated work and make the map lie.
    """
    # Public API rather than the private helper: it returns {arxiv_id, score}
    # dicts and fans out across all three dense shards, so the neighbourhood is
    # drawn from the whole corpus instead of one collection.
    hits = await qdrant_svc.search_by_vector_with_scores(
        vector, limit=INFER_NEIGHBOURS + 1, exclude_ids={arxiv_id},
    )
    neighbour_ids = [h["arxiv_id"] for h in hits if h.get("arxiv_id")]
    if not neighbour_ids:
        return None
    loop = asyncio.get_running_loop()
    known = await loop.run_in_executor(None, _fetch_known, neighbour_ids[:INFER_NEIGHBOURS])
    placed = [row for row in known.values() if row["position"] and row["source"] == "map"]
    if not placed:
        return None

    count = len(placed)
    centroid = [sum(row["position"][axis] for row in placed) / count for axis in range(3)]
    cluster = max(
        {row["clusterId"] for row in placed},
        key=lambda c: sum(1 for row in placed if row["clusterId"] == c),
    )
    return {
        "arxivId": arxiv_id,
        "position": centroid,
        "clusterId": cluster,
        "tileId": None,
        "title": "",
        "source": "inferred",
        "inferredFrom": count,
    }


def _inferred_point_id(arxiv_id: str) -> int:
    digest = hashlib.sha256(arxiv_id.encode("utf-8")).digest()
    return INFERRED_ID_BASE + int.from_bytes(digest[:6], "big") % INFERRED_ID_BASE


def _persist_inferred(rows: list[dict]) -> None:
    """Write inferred placements back so a paper does not move between sessions."""
    if not rows:
        return
    client = _map_client()
    points = [
        PointStruct(
            # SHA-256, not hash(): Python randomises string hashing per process,
            # so hash() would give the same paper a different point id after every
            # restart and quietly accumulate duplicates -- the exact opposite of
            # persisting the placement.
            id=_inferred_point_id(row["arxivId"]),
            vector=row["position"],
            payload={
                "arxiv_id": row["arxivId"],
                "cluster_id": row["clusterId"],
                "tile_id": None,
                "title": row["title"],
                "inferred": True,
                "inferred_from": row.get("inferredFrom", 0),
            },
        )
        for row in rows
    ]
    client.upsert(collection_name=MAP_COLLECTION, points=points, wait=False)


async def locate(arxiv_ids: list[str], infer: bool = True) -> dict[str, dict]:
    loop = asyncio.get_running_loop()
    known = await loop.run_in_executor(None, _fetch_known, arxiv_ids)
    missing = [aid for aid in arxiv_ids if aid not in known]
    if not missing or not infer:
        return known

    vectors = await qdrant_svc.get_paper_vectors(missing)
    inferred: list[dict] = []
    for aid in missing:
        vector = vectors.get(aid)
        if not vector:
            continue
        row = await _infer_one(aid, vector)
        if row:
            known[aid] = row
            inferred.append(row)
    if inferred:
        await loop.run_in_executor(None, _persist_inferred, inferred)
    return known


async def collection_stats() -> dict:
    client = _map_client()
    loop = asyncio.get_running_loop()
    info = await loop.run_in_executor(None, client.get_collection, MAP_COLLECTION)
    return {"collection": MAP_COLLECTION, "points": info.points_count, "status": str(info.status)}
