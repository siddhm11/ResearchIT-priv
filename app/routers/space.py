"""
JSON API for researchit-space (the 3D map client).

Endpoints:
  GET /api/space/search?q=&limit=       -- hybrid search, ranked arxiv_ids + titles
  GET /api/space/similarity?a=&b=       -- true BGE-M3 cosine between two papers
  GET /api/space/neighbors?arxiv_id=    -- semantic nearest neighbours of one paper

Why this exists as a separate router:

Every other route in this app returns server-rendered HTML for htmx. The map
client runs on a Cloudflare Worker — a JS edge runtime with no PyTorch — so it
cannot embed a query itself and cannot parse our templates. It needs JSON.

Nothing here re-implements retrieval. `hybrid_search_svc.search` already does
rewrite -> encode -> dense+sparse fanout -> RRF -> title boost, and
`qdrant_svc.get_paper_vectors` already fetches (and caches) stored vectors by
arxiv_id. These are thin, honest wrappers over both. Qdrant credentials stay
here and are never handed to the edge app.
"""
import time

from fastapi import APIRouter, Header, HTTPException, Query

from app import config, hybrid_search_svc, local_meta, qdrant_svc

router = APIRouter(prefix="/api/space")

# Cap what a single anonymous call can pull. The map client's own batch
# coordinate lookup tops out at 100 ids, so there is no use for more.
_MAX_LIMIT = 100


def _authorize(authorization: str | None) -> None:
    """
    Bearer check, but only when a token is configured.

    Left open when SPACE_SERVICE_TOKEN is unset so this router behaves like the
    rest of the app today; setting the variable turns enforcement on without
    needing a code change.
    """
    expected = getattr(config, "SPACE_SERVICE_TOKEN", "")
    if not expected:
        return
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="invalid service token")


def _titles_for(arxiv_ids: list[str]) -> dict[str, str]:
    """Best-effort titles from the local sidecar; absent ids simply stay absent."""
    if not arxiv_ids:
        return {}
    try:
        rows = local_meta.fetch_rows(arxiv_ids)
    except Exception as exc:  # sidecar is optional, never fatal
        print(f"[space] title lookup failed ({exc})")
        return {}
    return {
        row["arxiv_id"]: (row.get("title") or "").strip()
        for row in rows
        if row.get("arxiv_id")
    }


@router.get("/search")
async def space_search(
    q: str = Query("", description="natural-language query"),
    limit: int = Query(24, ge=1, le=_MAX_LIMIT),
    authorization: str | None = Header(default=None),
):
    """
    Ranked arxiv_ids for a query.

    Deliberately returns ids and titles only, not coordinates: this service has
    no idea where the map puts a paper. The client joins these ids against its
    own /api/space/lookup/batch to get positions and cluster ids. Keeping the
    join on the client is what lets the map swap its spatial dataset without
    this endpoint changing at all.
    """
    _authorize(authorization)
    query = (q or "").strip()
    if not query:
        return {"query": "", "results": [], "tookMs": 0, "meta": {}}

    started = time.perf_counter()
    try:
        ids, meta = await hybrid_search_svc.search(query, limit=limit, return_meta=True)
    except Exception as exc:
        # search() documents that it never raises, but a JSON contract should
        # not depend on that promise holding forever.
        print(f"[space] search failed ({exc})")
        raise HTTPException(status_code=502, detail="search unavailable") from exc

    titles = _titles_for(list(ids))
    return {
        "query": query,
        "tookMs": round((time.perf_counter() - started) * 1000),
        "results": [
            {"arxivId": aid, "title": titles.get(aid, ""), "rank": rank}
            for rank, aid in enumerate(ids, start=1)
        ],
        "meta": {
            "rewrittenQuery": (meta or {}).get("rewritten_query"),
            "backends": qdrant_svc._active_backends(),
        },
    }


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@router.get("/similarity")
async def space_similarity(
    a: str = Query(..., description="first arxiv id"),
    b: str = Query(..., description="second arxiv id"),
    authorization: str | None = Header(default=None),
):
    """
    True semantic similarity: cosine between the two stored BGE-M3 vectors.

    This is the authoritative number. The map client also knows the euclidean
    distance between the same two papers in its 3D projection, but that is a
    lossy UMAP artifact and must never be presented as the similarity — the two
    genuinely disagree, and where they disagree is the interesting part.

    Vectors are stored float16 with binary quantization; the retrieve path
    rescores from disk, which is why get_paper_vectors caches aggressively.
    """
    _authorize(authorization)
    left, right = a.strip(), b.strip()
    if not left or not right:
        raise HTTPException(status_code=400, detail="both a and b are required")

    try:
        vectors = await qdrant_svc.get_paper_vectors([left, right])
    except Exception as exc:
        print(f"[space] vector fetch failed ({exc})")
        raise HTTPException(status_code=502, detail="vector store unavailable") from exc

    va, vb = vectors.get(left), vectors.get(right)
    titles = _titles_for([left, right])
    both = va is not None and vb is not None
    return {
        "a": {"arxivId": left, "title": titles.get(left, ""), "found": va is not None},
        "b": {"arxivId": right, "title": titles.get(right, ""), "found": vb is not None},
        # null rather than 0.0 when either vector is missing: absent is not
        # "completely dissimilar", and the UI has to say so.
        "cosine": round(_cosine(va, vb), 6) if both else None,
        "dimensions": len(va) if va else None,
        "model": "BAAI/bge-m3",
    }


@router.get("/neighbors")
async def space_neighbors(
    arxiv_id: str = Query(..., description="anchor paper"),
    limit: int = Query(12, ge=1, le=_MAX_LIMIT),
    authorization: str | None = Header(default=None),
):
    """
    Semantic nearest neighbours of one paper, with real cosine scores.

    Lets the map answer "what is actually near this paper?" from the vector
    store rather than from what happens to be spatially adjacent in the
    projection — which is the same distinction /similarity draws, applied to a
    neighbourhood instead of a pair.
    """
    _authorize(authorization)
    anchor = arxiv_id.strip()
    if not anchor:
        raise HTTPException(status_code=400, detail="arxiv_id is required")

    try:
        vectors = await qdrant_svc.get_paper_vectors([anchor])
        vector = vectors.get(anchor)
        if vector is None:
            return {"arxivId": anchor, "found": False, "results": []}
        hits = await qdrant_svc.search_by_vector_with_scores(
            vector, limit=limit + 1, exclude_ids={anchor},
        )
    except Exception as exc:
        print(f"[space] neighbour search failed ({exc})")
        raise HTTPException(status_code=502, detail="vector store unavailable") from exc

    hits = hits[:limit]
    titles = _titles_for([h["arxiv_id"] for h in hits])
    return {
        "arxivId": anchor,
        "found": True,
        "results": [
            {
                "arxivId": h["arxiv_id"],
                "title": titles.get(h["arxiv_id"], ""),
                "cosine": round(float(h["score"]), 6),
                "rank": rank,
            }
            for rank, h in enumerate(hits, start=1)
        ],
    }
