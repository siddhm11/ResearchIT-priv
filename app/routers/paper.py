"""
Paper page — the site's owned destination for a single paper.

WHY THIS EXISTS
---------------
Until now no URL on this site was about a paper. Every card sent its title and
its arXiv id straight to arxiv.org, so the product had no page of its own to
share, to link to, or to return to. Three consequences, all of them structural
rather than cosmetic:

  * Nothing is shareable. The Open Graph tags added to base.html describe the
    feed, because the feed was the only thing there was to describe. A pasted
    link could never carry a paper's title.
  * There is nowhere to put comprehension. Plain-language explanation,
    difficulty, "read this first" — every one of them needs a surface, and a
    card in an infinite feed is not it.
  * There is no reason to come back. A link out to arxiv.org is the end of the
    session.

GET /p/{arxiv_id}
  Full metadata, the untruncated-where-possible abstract, save/dismiss, related
  work drawn from the paper's own embedding, and per-paper card tags.
"""
from __future__ import annotations

import asyncio
import time
import uuid

from fastapi import APIRouter, Cookie, Form, Query, Request
from fastapi.responses import HTMLResponse

from app import (abstract_svc, arxiv_svc, briefing_svc, config, db, features, groq_svc,
                 hf_papers_svc, http_client, paper_source_svc, qdrant_svc, readability, runtime_stats,
                 turso_svc, user_state as us)
from app.routers.figures import figure_credit, figure_url
from app.config import COOKIE_NAME
from app.routers.events import _NO_POSITION, _position
from app.templates_env import templates

router = APIRouter()

# How many neighbours to show. Small on purpose: this is "what sits next to
# this paper", not a second feed, and each one costs a metadata row.
_RELATED_LIMIT = 6


def _cluster_of(candidate_source: str) -> int | None:
    """The interest cluster a candidate_source names, or None.

    Tier 1 tags every candidate with "cluster_<idx>"; every other source
    ("exploration", "ewma_longterm", "qdrant_recommend",
    "trending_category_fallback", "paper_page") genuinely has no cluster and
    must stay NULL rather than be coerced to 0 -- 0 is a real cluster index.
    """
    if not candidate_source or not candidate_source.startswith("cluster_"):
        return None
    try:
        return int(candidate_source[len("cluster_"):])
    except ValueError:
        return None


async def _fetch_one(arxiv_id: str) -> dict | None:
    """Metadata for a single paper: sidecar/Turso first, arXiv as fallback.

    A full abstract saved by abstract_svc replaces a 500-character stump. The
    overlay returns a copy, never the dict turso_svc caches for everyone.
    """
    meta = await turso_svc.fetch_metadata_batch([arxiv_id])
    paper = meta.get(arxiv_id)
    if paper:
        return (await abstract_svc.overlay([paper]))[0]
    try:
        fallback = await arxiv_svc.fetch_metadata_batch([arxiv_id])
    except Exception as e:
        print(f"[paper] arXiv fallback failed for {arxiv_id}: {e}")
        return None
    return fallback.get(arxiv_id)


async def _related(arxiv_id: str, seen: set[str]) -> list[dict]:
    """Nearest neighbours of this paper, by its own embedding.

    Best-effort throughout: the page is worth serving without them, so a vector
    store that is down costs the section and nothing else.
    """
    try:
        vectors = await qdrant_svc.get_paper_vectors([arxiv_id])
        vec = vectors.get(arxiv_id)
        if vec is None:          # numpy array — `not vec` would raise
            return []
        hits = await qdrant_svc.search_by_vector(
            query_vector=vec.tolist(),
            limit=_RELATED_LIMIT + 1,
            exclude_ids=seen | {arxiv_id},
        )
    except Exception as e:
        print(f"[paper] related lookup failed for {arxiv_id}: {e}")
        return []

    ids = [h for h in hits if h != arxiv_id][:_RELATED_LIMIT]
    if not ids:
        return []
    meta = await turso_svc.fetch_metadata_batch(ids)
    return [meta[i] for i in ids if i in meta]


@router.get("/p/{arxiv_id}", response_class=HTMLResponse)
async def paper_page(
    arxiv_id: str,
    request: Request,
    qid: str = Query(default=""),
    pos: int = Query(default=_NO_POSITION),
    src: str = Query(default=""),
    sf: str = Query(default=""),
    prop: float = Query(default=0.0),
    pol: str = Query(default=""),
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    t_start = time.perf_counter()
    returning_reader = user_id is not None
    user_id = user_id or str(uuid.uuid4())
    state = await us.ensure_loaded(user_id)

    t_meta = time.perf_counter()
    paper = await _fetch_one(arxiv_id)
    meta_ms = (time.perf_counter() - t_meta) * 1000
    if paper is None:
        # Rendered through the normal layout by the 404 handler in main.py,
        # rather than as a bare JSON detail.
        return templates.TemplateResponse(
            request, "not_found.html", {}, status_code=404)

    # A cut abstract is queued for its full text, which the next view shows.
    # Off the request path: request() only schedules the fetch.
    if abstract_svc.is_truncated(paper.get("abstract")):
        abstract_svc.request(arxiv_id)
    # Likewise the arXiv HTML page (figures, sections, paragraphs), read once
    # in the background when PAPER_SOURCES is shadow or on. Never awaited here.
    paper_source_svc.schedule(arxiv_id)

    paper = {
        **paper,
        "saved": arxiv_id in state.positives,
        "dismissed": arxiv_id in state.negatives,
        # This page is not a ranked surface, so it carries no ranking
        # instrumentation. Leaving the §3.11 fields empty is deliberate: a
        # save from here genuinely has no query_id, propensity or policy, and
        # inventing one would corrupt the very analysis those fields exist for.
        "candidate_source": "paper_page",
    }

    # The click-through. app/db.py has declared `click` as an event_type since
    # the schema was written and nothing ever wrote one, so the system knew
    # which papers were saved and which were dismissed but not which were
    # OPENED — the difference between "scrolled past" and "read", and the
    # densest engagement signal a feed produces.
    #
    # Only logged when the visit carries a query_id, i.e. it came from a ranked
    # surface. A bare /p/{id} visit — a shared link, a bookmark, a crawler —
    # has no propensity and no policy, and recording it as though it did would
    # corrupt the §3.11 contract rather than honour it.
    if qid:
        try:
            await db.log_interaction(
                user_id=user_id,
                paper_id=arxiv_id,
                event_type="click",
                # The SURFACE, not the candidate source. This used to pass
                # `src`, which is the retrieval origin ("cluster_1"), so the
                # column that is supposed to hold search|recommendation|saved
                # filled up with cluster names instead: 2 rows said
                # "recommendation" where 21 belonged, and a click from a search
                # result -- which carries a query_id but no src -- was recorded
                # as a recommendation. Both made grouping by source wrong.
                source=sf or "recommendation",
                position=_position(pos),
                query_id=qid,
                ranker_version=pol or None,
                candidate_source=src or None,
                # Was hard-coded None while `src` literally spelled the cluster
                # out. Per-cluster CTR is the one measurement that can show
                # whether a minority interest earns its slots, and it was
                # unanswerable because this column was empty on every row.
                cluster_id=_cluster_of(src),
                propensity=prop if prop > 0 else None,
                policy_id=pol or None,
            )
        except Exception as e:   # a lost click must never cost the page
            print(f"[paper] click log failed for {arxiv_id}: {e}")
    elif returning_reader:
        # A direct visit belongs in reading history, but has no ranked exposure
        # to attribute. A separate view event must not inflate feed CTR or
        # become a positive/negative profile update. Cookie-less requests
        # (crawlers, link previews) would only mint a one-row phantom user.
        try:
            await db.log_interaction(user_id=user_id, paper_id=arxiv_id,
                                     event_type="view", source="paper_page")
        except Exception as e:
            print(f"[paper] history log failed for {arxiv_id}: {e}")

    t_related = time.perf_counter()
    related = await _related(arxiv_id, us.all_seen(user_id))
    related_ms = (time.perf_counter() - t_related) * 1000

    context = {
        "paper": paper,
        "related": related,
        "reading": readability.assess(paper.get("abstract") or ""),
        "og_title": paper.get("title") or f"arXiv:{arxiv_id}",
        "og_description": _summary_for_card(paper),
        "og_type": "article",
    }
    template = "paper.html"
    if features.visible("PAPER_PAGE_V2", request):
        template = "paper_v2.html"
        context.update(await _v2_context(paper, related, request, came_from_feed=bool(qid)))
    resp = templates.TemplateResponse(request, template, context)
    resp.set_cookie(COOKIE_NAME, user_id, max_age=365 * 24 * 3600, httponly=True)
    # Recorded so a redesign of this page can be compared against what it
    # replaced: /healthz/runtime reports p50/p95 of the last 200 visits.
    runtime_stats.record("paper_page", {
        "meta_ms": round(meta_ms, 1),
        "related_ms": round(related_ms, 1),
        "total_ms": round((time.perf_counter() - t_start) * 1000, 1),
    })
    return resp


async def _v2_context(paper: dict, related: list[dict], request: Request,
                      *, came_from_feed: bool) -> dict:
    """What the redesigned page adds, from local reads only (never arXiv)."""
    arxiv_id = paper["arxiv_id"]
    source = None
    try:
        source = await paper_source_svc.get(arxiv_id)
    except Exception as e:                      # the page is worth serving without it
        print(f"[paper] saved source unavailable for {arxiv_id}: {e}")
    figures = []
    if source is not None and features.visible("FIGURES", request):
        # The replicated row cuts captions at 300 characters; the local full
        # text, when this container has it, holds them whole.
        full_captions = {}
        try:
            full = await paper_source_svc.fulltext(arxiv_id) if source.ok else None
            full_captions = {f["anchor"]: f.get("caption") or "" for f in (full or {}).get("figures", [])}
        except Exception as e:
            print(f"[paper] full text unavailable for {arxiv_id}: {e}")
        for n, f in enumerate(source.figures):
            url = figure_url(source, n)
            if not url:
                continue
            caption = full_captions.get(f.get("anchor")) or f.get("caption") or ""
            if caption == f.get("caption") and len(caption) >= paper_source_svc.REPLICATED_CAPTION:
                caption = caption.rstrip() + "…"
            figures.append({"n": n, "url": url, "label": f.get("label") or "",
                            "caption": caption, "w": f.get("w"), "h": f.get("h")})
    ask_visible = features.visible("ASK", request)
    briefing_visible = features.visible("BRIEFING", request)
    outline = [("sec-sum", "Summary")]
    if figures:
        outline.append(("sec-fig", "Key figure"))
    if briefing_visible:
        outline += [("sec-things", "Three things to know"), ("sec-how", "How it works"),
                    ("sec-use", "Where you'd use this")]
    if len(figures) > 1:
        outline.append(("sec-figs", "All figures"))
    if related:
        outline.append(("sec-near", "What sits near this"))
    return {
        "paper_source": source,
        "figures": figures,
        "credit": figure_credit(paper, source) if source else "",
        "outline": outline,
        "back_to_feed": came_from_feed,
        "ask_visible": ask_visible,
        "briefing_visible": briefing_visible,
    }


def _summary_for_card(paper: dict) -> str:
    """A one-line description for the link preview.

    Kept short and cut on a word boundary — a social card truncates anyway, and
    cutting mid-word is exactly the defect the abstract work just fixed on the
    card itself.
    """
    text = (paper.get("abstract") or "").strip().replace("\n", " ")
    if not text:
        authors = (paper.get("authors") or "").strip()
        return f"{authors} — on arXiv." if authors else "A paper on arXiv."
    if len(text) <= 200:
        return text
    return text[:200].rsplit(" ", 1)[0] + "…"


@router.get("/api/papers/{arxiv_id}/explain", response_class=HTMLResponse)
async def explain(arxiv_id: str, request: Request, v: int = Query(default=1)):
    """Plain-language explanation of one paper.

    Fetched by htmx AFTER the page paints rather than rendered inline. The
    generation is a network call to Groq with an 8s ceiling, and the paper page
    must not wait on it — a reader who wants the abstract should never pay for
    a summary they did not ask to wait for. On a cache hit this is one indexed
    SQLite read and returns immediately.

    Shared across users and content-addressed, so the corpus warms itself: the
    second reader of a paper pays nothing.
    """
    paper = await _fetch_one(arxiv_id)
    if paper is None:
        return HTMLResponse(content="")

    abstract = paper.get("abstract") or ""
    # Write the explanation from the whole abstract when it can be had within a
    # few seconds, rather than caching one written from a stump for everyone.
    if abstract_svc.is_truncated(abstract):
        full = await abstract_svc.ensure(arxiv_id, timeout=4.0)
        if full and len(full) > len(abstract):
            abstract = full
    key = groq_svc.explain_cache_key(arxiv_id, abstract)

    cached = None
    try:
        cached = await db.get_explanation(key)
    except Exception as e:                      # cache is an optimisation
        print(f"[paper] explanation cache read failed: {e}")

    text = cached
    runtime_stats.count("explain_cache_hit" if cached is not None else "explain_cache_miss")
    if text is None:
        text = await groq_svc.explain_paper(paper.get("title") or "", abstract)
        if text:
            try:
                await db.save_explanation(
                    key, arxiv_id, text, groq_svc.explain_model())
            except Exception as e:
                print(f"[paper] explanation cache write failed: {e}")

    # Nothing rather than an apology. The section simply does not appear when
    # there is no honest summary to give — a truncated abstract, no API key, a
    # timeout. An empty slot is quieter than an error the reader cannot act on.
    if not text:
        return HTMLResponse(content="")

    # v=2 is the redesigned summary box, which supplies its own tab and label.
    partial = "partials/paper/explanation_body.html" if v == 2 else "partials/explanation.html"
    return templates.TemplateResponse(
        request, partial,
        {"explanation": text, "cached": cached is not None},
    )


# ── Code link (the redesigned page's Code chip) ─────────────────────────────

CODE_RECHECK_S = 7 * 24 * 3600


def _checked_recently(checked_at: str) -> bool:
    import calendar
    try:
        stamp = calendar.timegm(time.strptime(checked_at, "%Y-%m-%d %H:%M:%S"))
    except (TypeError, ValueError):
        return False
    return time.time() - stamp < CODE_RECHECK_S


@router.get("/api/papers/{arxiv_id}/code", response_class=HTMLResponse)
async def code_link(arxiv_id: str, request: Request):
    """The paper's GitHub repository as a chip, or nothing.

    Loaded after the page paints. Answers, "no repository" included, are saved
    in paper_sources and rechecked weekly. Only papers already in our metadata
    are looked up: no arXiv fallback, so an invented id costs nothing.
    """
    if not features.visible("PAPER_PAGE_V2", request) or not paper_source_svc.ID_PATTERN.match(arxiv_id):
        return HTMLResponse("")
    source = await paper_source_svc.get(arxiv_id)
    if source is not None and _checked_recently(source.code_checked_at):
        repo, stars = source.code_repo, source.code_stars
    else:
        known = await turso_svc.fetch_metadata_batch([arxiv_id])
        if arxiv_id not in known:
            return HTMLResponse("")
        found = await hf_papers_svc.fetch_paper_code(
            http_client.get_client(), arxiv_id, user_agent=config.OUTBOUND_USER_AGENT)
        if found is None:                       # lookup failed: show what we had, cache nothing
            repo, stars = (source.code_repo, source.code_stars) if source else ("", None)
        else:
            repo, stars = found
            try:
                await paper_source_svc.save_code_link(arxiv_id, repo, stars)
            except Exception as e:
                print(f"[paper] saving the code link of {arxiv_id} failed: {e}")
    return templates.TemplateResponse(request, "partials/paper/code_chip.html",
                                      {"repo": hf_papers_svc.github_repo(repo), "stars": stars})



# ── Briefing (three things, how it works, where you'd use this) ──────────────

BRIEFING_WAIT_S = 12.0
_pending_work: set[asyncio.Task] = set()      # strong references (see events._spawn)


async def _known_paper(arxiv_id: str) -> dict | None:
    """Metadata from the sidecar or Turso only: no arXiv fallback, so an invented
    id can't spend a model call or an outside request."""
    if not paper_source_svc.ID_PATTERN.match(arxiv_id):
        return None
    meta = await turso_svc.fetch_metadata_batch([arxiv_id])
    paper = meta.get(arxiv_id)
    return (await abstract_svc.overlay([paper]))[0] if paper else None


@router.get("/api/papers/{arxiv_id}/briefing", response_class=HTMLResponse)
async def briefing(arxiv_id: str, request: Request):
    """The briefing fragment, or nothing.

    Generated once per paper version and cached. A reader waits at most
    BRIEFING_WAIT_S; past that the page shows nothing this time and the
    generation finishes in the background for the next visit. In shadow mode
    it generates and stores, and always answers empty.
    """
    if not features.active("BRIEFING"):
        return HTMLResponse("")
    paper = await _known_paper(arxiv_id)
    if paper is None:
        return HTMLResponse("")
    work = asyncio.ensure_future(briefing_svc.get_or_generate(
        arxiv_id, paper.get("title") or "", paper.get("abstract") or ""))
    _pending_work.add(work)
    work.add_done_callback(_pending_work.discard)
    try:
        status, result = await asyncio.wait_for(asyncio.shield(work), BRIEFING_WAIT_S)
    except asyncio.TimeoutError:
        return HTMLResponse("")                         # it finishes for the next visit
    if status != "ok" or not features.visible("BRIEFING", request):
        return HTMLResponse("")
    source = await paper_source_svc.get(arxiv_id)
    return templates.TemplateResponse(request, "partials/paper/briefing.html",
                                      {"briefing": result, "paper_source": source})


@router.post("/api/papers/{arxiv_id}/report", response_class=HTMLResponse)
async def report_problem(arxiv_id: str, request: Request, feature: str = Form(default=""),
                         reason: str = Form(default=""),
                         user_id: str | None = Cookie(default=None, alias=COOKIE_NAME)):
    """Record a reader's report on generated text; reviewed by hand."""
    if paper_source_svc.ID_PATTERN.match(arxiv_id) and feature in ("briefing", "ask", "explain"):
        try:
            await briefing_svc.report(arxiv_id, feature, reason, user_id)
        except Exception as e:
            print(f"[paper] saving a report failed: {e}")
    return HTMLResponse('<span class="report-done">Thanks — we\'ll look at it.</span>')
