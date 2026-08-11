"""
Search router — Phase 3 hybrid semantic search.

GET /search?q=<query>
  – Returns full page on normal request
  – Returns partial <div id="search-results"> on HTMX request (hx-target swap)

Phase 3 replaces the arXiv keyword API with:
  LLM rewrite → BGE-M3 encode → Qdrant dense + Zilliz sparse → RRF → rerank

Phase 3.5: Metadata now fetched from Turso cloud DB (fast, includes citations)
  with arXiv API as fallback for papers not in Turso.
"""
import uuid
from fastapi import APIRouter, Request, Cookie
from fastapi.responses import HTMLResponse
from app import arxiv_svc, db, turso_svc, user_state as us, hybrid_search_svc
from app.config import COOKIE_NAME, ARXIV_MAX_RESULTS
from app.templates_env import templates

router = APIRouter()


@router.get("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    q: str = "",
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    import time
    start_time = time.perf_counter()
    search_meta = {}
    papers = []
    if q.strip():
        # Phase 3: Hybrid semantic search (BGE-M3 + Qdrant + Zilliz + RRF)
        try:
            arxiv_ids, search_meta = await hybrid_search_svc.search(
                q.strip(), limit=ARXIV_MAX_RESULTS, return_meta=True
            )
        except Exception as e:
            print(f"[search] Hybrid search error: {e}")
            arxiv_ids = []

        if arxiv_ids:
            # Phase 3.5: Fetch metadata from Turso DB first (fast, ~50ms)
            t0_meta = time.perf_counter()
            try:
                meta = await turso_svc.fetch_metadata_batch(arxiv_ids)
            except Exception as e:
                print(f"[search] Turso metadata fetch failed: {e}")
                meta = {}

            # Fallback: fetch any missing IDs from arXiv API
            missing = [aid for aid in arxiv_ids if aid not in meta]
            if missing:
                try:
                    arxiv_meta = await arxiv_svc.fetch_metadata_batch(missing)
                    meta.update(arxiv_meta)
                except Exception as e:
                    print(f"[search] arXiv fallback for {len(missing)} IDs failed: {e}")
            
            search_meta["meta_time_ms"] = int((time.perf_counter() - t0_meta) * 1000)

            # Phase 4.3: Cache to SQLite so dismissal category JOINs work
            await db.cache_turso_metadata_batch(list(meta.values()))

            # Preserve ranking order from hybrid search
            papers = [meta[aid] for aid in arxiv_ids if aid in meta]

        if not papers and q.strip():
            # Fallback: arXiv keyword API if hybrid returns nothing
            try:
                papers = await arxiv_svc.search(q.strip())
            except Exception as e:
                print(f"[search] arXiv fallback also failed: {e}")
                papers = []
        
        search_meta["total_time_ms"] = int((time.perf_counter() - start_time) * 1000)

    user_id = user_id or str(uuid.uuid4())
    # Phase 6.5 B1: one query_id per search request for per-feed CTR
    query_id = str(uuid.uuid4())
    state = await us.ensure_loaded(user_id)
    saved_ids = set(state.positive_list)
    dismissed_ids = set(state.negative_list)

    for idx, p in enumerate(papers):
        p["saved"] = p["arxiv_id"] in saved_ids
        p["dismissed"] = p["arxiv_id"] in dismissed_ids
        p["query_id"] = query_id
        p["position"] = idx
        p["propensity"] = 1.0  # search is deterministic
        p["policy_id"] = "search_v1"

    if request.headers.get("HX-Request"):
        resp = templates.TemplateResponse(
            request,
            "partials/search_results.html",
            {"papers": papers, "query": q, "search_meta": search_meta},
        )
    else:
        resp = templates.TemplateResponse(
            request,
            "search.html",
            {
                "papers": papers,
                "query": q,
                "has_recs": state.has_enough_for_recs(),
                "search_meta": search_meta,
            },
        )

    resp.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
    return resp


@router.get("/api/search/summary", response_class=HTMLResponse)
async def search_summary(
    request: Request,
    q: str = "",
    ids: str = "",
):
    """
    Generate an AI Overview for a given search query and set of arXiv IDs.
    Called asynchronously by HTMX after search results load.
    """
    if not q or not ids:
        return ""
        
    arxiv_ids = [aid.strip() for aid in ids.split(",") if aid.strip()]
    if not arxiv_ids:
        return ""

    from app import groq_svc
    
    # Fetch metadata for the top papers (need abstracts for summary)
    try:
        meta = await turso_svc.fetch_metadata_batch(arxiv_ids[:5])
        papers = [meta[aid] for aid in arxiv_ids[:5] if aid in meta]
    except Exception as e:
        print(f"[search] Turso fetch failed for summary: {e}")
        return ""

    if not papers:
        return ""

    # The Turso fetch above is already guarded; this call was not. A Groq
    # rate-limit, timeout or outage therefore escaped as a 500, and the client
    # turns any non-2xx here into an htmx:responseError -> "Something went
    # wrong. Please try again." on an otherwise perfectly good results page.
    #
    # The overview is strictly additive, so every failure mode degrades to
    # "no overview" -- the same contract every other optional service in this
    # app follows (BGE-M3, the cross-encoder, Zilliz, the sidecar).
    try:
        summary_html = await groq_svc.generate_search_summary(q, papers)
    except Exception as e:
        print(f"[search] summary generation failed: {e}")
        return ""

    if not summary_html:
        return ""

    return templates.TemplateResponse(
        request,
        "partials/ai_summary.html",
        {"summary_html": summary_html},
    )
