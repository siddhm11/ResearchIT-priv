"""
Search router — Phase 3 hybrid semantic search.

GET /search?q=<query>
  – Returns full page on normal request
  – Returns partial <div id="search-results"> on HTMX request (hx-target swap)

Phase 3 replaces the arXiv keyword API with:
  LLM rewrite → BGE-M3 encode → Qdrant dense + FTS5 keyword → RRF → rerank

Phase 3.5: Metadata now fetched from Turso cloud DB (fast, includes citations)
  with arXiv API as fallback for papers not in Turso.
"""
import re
import uuid
from fastapi import APIRouter, Request, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from app import analytics, arxiv_svc, card_data, db, features, figure_prewarm, search_cache, turso_svc, user_state as us, hybrid_search_svc
from app.config import COOKIE_NAME, COOKIE_MAX_AGE, ARXIV_MAX_RESULTS
from app.templates_env import templates

router = APIRouter()

# A query that is only an arXiv id, or a link to one paper on arXiv, Hugging
# Face Papers or alphaXiv, opens that paper. It is the first thing a researcher
# tries in a new tool, and hybrid search could not answer it.
_ARXIV_ID = r"(?:\d{4}\.\d{4,5}|[a-z][a-z\-]*(?:\.[a-z]{2})?/\d{7})"
_PAPER_QUERY = re.compile(
    rf"^(?:arxiv:\s*|(?:https?://)?(?:www\.|export\.)?"
    rf"(?:arxiv\.org/(?:abs|pdf|html)|huggingface\.co/papers|alphaxiv\.org/(?:abs|overview))/)"
    rf"?({_ARXIV_ID})(?:v\d+)?(?:\.pdf)?/?(?:[?#]\S*)?$",
    re.IGNORECASE)


# ── More results and filters (audit F18) ─────────────────────────────────────
# A search keeps the fused top 50 (search_meta["more_ids"]). Later pages and the
# year/area filters read that list, so they cost a metadata read from the local
# sidecar and nothing else: no encode, no rewrite, no cross-encoder. Pages after
# the first are in fused order; only the first was cross-encoded.
AREAS = {"cs": "Computer science", "math": "Mathematics", "stat": "Statistics", "physics": "Physics",
         "q-bio": "Quantitative biology", "eess": "Electrical engineering", "econ": "Economics",
         "q-fin": "Quantitative finance"}
_PHYSICS = ("physics", "astro-ph", "cond-mat", "gr-qc", "hep-", "math-ph", "nlin", "nucl-",
            "quant-ph")
YEARS = (2025, 2023, 2020)


def _in_area(paper: dict, area: str) -> bool:
    cats = (paper.get("arxiv_categories") or paper.get("category") or "").replace(",", " ").split()
    if area == "physics":
        return any(c.startswith(_PHYSICS) for c in cats)
    return any(c.split(".")[0] == area for c in cats)


def _keeps(paper: dict, since: int, area: str) -> bool:
    from app.local_meta import pub_year_month
    if since:
        ym = pub_year_month(paper.get("arxiv_id", ""))
        if ym is None or ym[0] < since:
            return False
    return not area or _in_area(paper, area)


async def _pool(papers: list[dict], meta: dict) -> list[dict]:
    """The shown results followed by the rest of the fused top 50, as paper dicts."""
    more = [a for a in (meta.get("more_ids") or []) if a not in {p["arxiv_id"] for p in papers}]
    if not more:
        return papers
    try:
        rows = await turso_svc.fetch_metadata_batch(more)
    except Exception as e:
        print(f"[search] more results unavailable: {e}")
        return papers
    return papers + [dict(rows[a]) for a in more if a in rows]


def paper_for_query(q: str) -> str | None:
    """The arXiv id a query names on its own, version stripped, or None."""
    m = _PAPER_QUERY.match(q.strip())
    return m.group(1) if m else None


@router.get("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    q: str = "",
    page: int = 1,
    since: int = 0,
    area: str = "",
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    import time
    page = max(1, min(page, 5))
    since = since if since in YEARS else 0
    area = area if area in AREAS else ""
    named = paper_for_query(q)
    if named:
        target = f"/p/{named}"
        if request.headers.get("HX-Request"):
            # The box submits through htmx; navigate instead of swapping the
            # paper page into the results list.
            return Response(status_code=200, headers={"HX-Redirect": target})
        return RedirectResponse(target, status_code=303)
    start_time = time.perf_counter()
    search_meta = {}
    papers = []
    # A search shown in the last few minutes is served from memory: going back
    # to it, or running it again, costs no encode, no Groq call and no retrieval.
    cached = search_cache.get(q) if q.strip() else None
    if cached is not None:
        papers, search_meta = cached.copies()
        search_meta["cached"] = True
        search_meta["total_time_ms"] = int((time.perf_counter() - start_time) * 1000)
    elif q.strip():
        # Hybrid semantic search (BGE-M3 + Qdrant dense + FTS5 keyword + RRF)
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
                if papers:
                    search_meta["retrieval_mode"] = "arxiv_keyword"
            except Exception as e:
                print(f"[search] arXiv fallback also failed: {e}")
                papers = []
        
        search_meta["total_time_ms"] = int((time.perf_counter() - start_time) * 1000)
        cached = search_cache.put(q, papers, search_meta)

    # Phase 6.5 B1: one query_id per search request for per-feed CTR. A reader
    # coming back to the same cached results keeps the query_id they were shown
    # under, and the visit is not logged or counted as another search.
    repeat_qid = search_cache.repeat_query_id(user_id, q, cached)
    request.state.repeat_search = repeat_qid is not None
    user_id = user_id or str(uuid.uuid4())
    query_id = repeat_qid or str(uuid.uuid4())
    search_cache.remember(user_id, q, cached, query_id)
    # Labels come from the durable decision log, not the hot deques: those keep
    # only the newest 20 saves / 50 dismissals for retrieval, so an older saved
    # paper rendered with a live Save button and invited a duplicate save.
    state = await us.ensure_loaded(user_id)
    feedback = await db.get_current_feedback(user_id)
    saved_ids = {r["paper_id"] for r in feedback if r["event_type"] == "save"}
    dismissed_ids = {r["paper_id"] for r in feedback if r["event_type"] == "not_interested"}

    # Pages and filters read the kept top 50; the first unfiltered page is the
    # search itself.
    per_page = ARXIV_MAX_RESULTS
    if page > 1 or since or area:
        pool = await _pool(papers, search_meta)
        if since or area:
            pool = [p for p in pool if _keeps(p, since, area)]
        start = (page - 1) * per_page
        papers, has_more = pool[start:start + per_page], len(pool) > start + per_page
    else:
        start, has_more = 0, bool(search_meta.get("more_ids"))

    for idx, p in enumerate(papers, start=start):
        p["saved"] = p["arxiv_id"] in saved_ids
        p["dismissed"] = p["arxiv_id"] in dismissed_ids
        p["query_id"] = query_id
        p["position"] = idx
        p["propensity"] = 1.0  # search is deterministic
        p["policy_id"] = "search_v2_independent_lexical"
    # Papers the temporary trending step-up moved into view (SEARCH_TRENDING=on)
    # say so, with the strongest count behind it. Shadow mode tags nothing.
    for p in papers:
        moved = (search_meta.get("trending") or {}).get(p.get("arxiv_id"))
        p["trending"] = {"fact": (moved.get("facts") or [""])[0]} if moved else None
    # The paper the search named, put first by app/named_paper.py, says why.
    named = (search_meta.get("named_paper") or {}).get("arxiv_id")
    for p in papers:
        p["named_paper"] = bool(named) and p.get("arxiv_id") == named

    # A small figure beside each result that already has one saved (switch
    # SEARCH_FIGURES; local reads only, order and instrumentation untouched),
    # and the top few queued so the warm-up reads them for next time.
    if features.visible("SEARCH_FIGURES", request):
        papers = await card_data.enrich(papers, request)
    figure_prewarm.enqueue([p["arxiv_id"] for p in papers[:figure_prewarm.SEARCH_TOP] if p.get("arxiv_id")])
    if q.strip() and repeat_qid is None and page == 1:
        await analytics.log_search(q, query_id, len(papers), search_meta.get("retrieval_mode") or "hybrid",
                                   search_meta.get("rewritten_query"), search_meta.get("total_time_ms") or 0,
                                   features.has_preview(request),
                                   # A cached answer carries the stage times of the search it copies.
                                   None if search_meta.get("cached") else search_meta)

    # htmx restoring a page it has no snapshot of asks with HX-Request too, and
    # swaps the answer in as the whole body, so it gets the whole page.
    paging = {"page": page, "since": since, "area": area, "has_more": has_more, "start": start,
              "years": YEARS, "areas": AREAS}
    if request.headers.get("HX-Request") and not request.headers.get("HX-History-Restore-Request"):
        resp = templates.TemplateResponse(
            request,
            "partials/search_more.html" if page > 1 else "partials/search_results.html",
            {"papers": papers, "query": q, "search_meta": search_meta, "with_title": True, **paging},
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
                **paging,
            },
        )

    resp.set_cookie(COOKIE_NAME, user_id, max_age=COOKIE_MAX_AGE, httponly=True)
    return resp


@router.get("/api/search/summary", response_class=HTMLResponse)
async def search_summary(
    request: Request,
    q: str = "",
    ids: str = "",
    qid: str = "",
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
    # app follows (BGE-M3, the cross-encoder, the sidecar).
    # The same query over the same top papers reuses the overview shown before
    # (app/analytics.py), instead of asking the model again.
    from app import ai_usage
    summary_html = await analytics.cached_overview(q, arxiv_ids)
    ai_usage.count("search_overview", "cache_hit" if summary_html else "cache_miss")
    if not summary_html:
        try:
            summary_html = await groq_svc.generate_search_summary(q, papers)
        except Exception as e:
            print(f"[search] summary generation failed: {e}")
            return ""
        if summary_html:
            await analytics.save_overview(q, arxiv_ids, summary_html)

    if not summary_html:
        return ""

    # Titles the overview names link to their papers. Done here, not before
    # caching, so overviews cached before this existed get links too, and the
    # cache never holds one reader's query_id.
    from app import overview_links
    summary_html = overview_links.link_titles(
        summary_html, papers, qid=qid, positions={aid: i for i, aid in enumerate(arxiv_ids)})

    return templates.TemplateResponse(
        request,
        "partials/ai_summary.html",
        {"summary_html": summary_html},
    )
