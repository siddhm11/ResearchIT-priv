"""
Turso (libSQL) metadata service — Phase 3.5.

Replaces arxiv_svc.fetch_metadata_batch() with direct Turso DB lookups.
Uses Turso's HTTP pipeline API — no additional Python dependencies needed
(just httpx, already installed).

The DB contains ~1.6M arXiv papers with metadata + citation counts from
Semantic Scholar, bulk-loaded from Kaggle.

Connection: TURSO_URL + TURSO_DB_TOKEN (env vars)
Table:      papers (arxiv_id UNIQUE INDEX)
"""
from __future__ import annotations

import json
import time
from collections import OrderedDict

import httpx

from app import config


# ── In-process metadata cache ────────────────────────────────────────────────
#
# Recommendations + search both fetch metadata for hundreds of arxiv_ids per
# request, often the same well-known papers across users. Each round-trip is
# 1-3s on a 1.6M-row libSQL DB. An in-process LRU absorbs the repeats.
#
# Trade-offs:
#   - Asyncio is single-threaded, no lock needed.
#   - Paper title/abstract/authors are effectively immutable for our use,
#     so we don't TTL-expire metadata. citation_count drifts but is only
#     used for display ranking; staleness is fine.
#   - 50K capacity at ~1KB per row -> ~50MB RAM ceiling.

_METADATA_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_METADATA_CACHE_MAX = 50_000


def _cache_get(arxiv_id: str) -> dict | None:
    val = _METADATA_CACHE.get(arxiv_id)
    if val is not None:
        # Mark as MRU
        _METADATA_CACHE.move_to_end(arxiv_id)
    return val


def _cache_put(arxiv_id: str, paper: dict) -> None:
    if arxiv_id in _METADATA_CACHE:
        _METADATA_CACHE.move_to_end(arxiv_id)
        _METADATA_CACHE[arxiv_id] = paper
        return
    _METADATA_CACHE[arxiv_id] = paper
    if len(_METADATA_CACHE) > _METADATA_CACHE_MAX:
        # Evict LRU
        _METADATA_CACHE.popitem(last=False)


def metadata_cache_stats() -> dict:
    """For diagnostics: current cache size and max."""
    return {"size": len(_METADATA_CACHE), "max": _METADATA_CACHE_MAX}


# ── In-process trending cache ────────────────────────────────────────────────
#
# Trending is filter-by-LIKE on 1.6M rows -> ~15s cold. Onboarding has a
# small fixed set of category combinations, and citation counts barely
# change minute-to-minute. A short TTL converts the 15s wait into a
# one-time hit per category combo.

_TRENDING_CACHE: dict[tuple, tuple[float, list[dict]]] = {}
_TRENDING_TTL_SECONDS = 60 * 60  # 1 hour


# ── Public API ───────────────────────────────────────────────────────────────

async def fetch_metadata(arxiv_id: str) -> dict | None:
    """Fetch metadata for a single paper from Turso."""
    result = await fetch_metadata_batch([arxiv_id])
    return result.get(arxiv_id)


async def fetch_metadata_batch(arxiv_ids: list[str]) -> dict[str, dict]:
    """
    Fetch metadata for multiple papers from Turso DB.

    Returns {arxiv_id: paper_dict} for all IDs found.
    Paper dict has keys: arxiv_id, title, abstract, authors, category,
    published, year, citation_count, influential_citations.

    First checks the in-process LRU cache; only un-cached IDs hit the network.
    """
    if not arxiv_ids:
        return {}

    # Cache check — pull anything already-known up front.
    output: dict[str, dict] = {}
    misses: list[str] = []
    for aid in arxiv_ids:
        cached = _cache_get(aid)
        if cached is not None:
            output[aid] = cached
        else:
            misses.append(aid)

    if not misses:
        return output

    fetched = await _fetch_metadata_batch_uncached(misses)
    output.update(fetched)
    return output


async def _fetch_metadata_batch_uncached(arxiv_ids: list[str]) -> dict[str, dict]:
    """Network fetch for IDs we don't already have cached."""
    url = config.TURSO_URL
    token = config.TURSO_DB_TOKEN

    if not url or not token:
        print("[turso] TURSO_URL or TURSO_DB_TOKEN not configured, skipping")
        return {}

    # Build parameterised query with placeholders
    placeholders = ", ".join(["?" for _ in arxiv_ids])
    sql = f"SELECT arxiv_id, title, authors, categories, primary_topic, update_date, abstract_preview, citation_count, influential_citations FROM papers WHERE arxiv_id IN ({placeholders})"

    args = [{"type": "text", "value": aid} for aid in arxiv_ids]

    # Turso HTTP pipeline API
    pipeline_url = url.rstrip("/")
    # Convert to HTTP API URL format
    if pipeline_url.startswith("libsql://"):
        pipeline_url = "https://" + pipeline_url[len("libsql://"):]
    elif pipeline_url.startswith("http://"):
        pipeline_url = "https://" + pipeline_url[len("http://"):]
    elif not pipeline_url.startswith("https://"):
        pipeline_url = "https://" + pipeline_url

    payload = {
        "requests": [
            {
                "type": "execute",
                "stmt": {"sql": sql, "args": args},
            },
            {"type": "close"},
        ]
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    t0 = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{pipeline_url}/v2/pipeline",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
    except Exception as e:
        print(f"[turso] HTTP request failed: {e}")
        return {}

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[turso] Fetched metadata for {len(arxiv_ids)} IDs in {elapsed_ms:.0f}ms")

    try:
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return {}

        # First result is our execute response
        execute_result = results[0]
        if execute_result.get("type") == "error":
            print(f"[turso] Query error: {execute_result.get('error')}")
            return {}

        response = execute_result.get("response", {})
        result_data = response.get("result", {})
        cols = [c["name"] for c in result_data.get("cols", [])]
        rows = result_data.get("rows", [])

    except (KeyError, IndexError, TypeError) as e:
        print(f"[turso] Response parsing error: {e}")
        return {}

    # Convert rows to paper dicts matching the expected format
    output: dict[str, dict] = {}
    for row in rows:
        # Each row is a list of {"type": "text"|"integer"|"null", "value": ...}
        values = {}
        for i, col in enumerate(cols):
            cell = row[i]
            if cell.get("type") == "null":
                values[col] = None
            else:
                values[col] = cell.get("value", "")

        paper = _to_paper_dict(values)
        if paper:
            output[paper["arxiv_id"]] = paper
            _cache_put(paper["arxiv_id"], paper)

    return output


def _to_paper_dict(row: dict) -> dict | None:
    """
    Convert a Turso row into the paper dict format expected by templates.

    Template expects:
      arxiv_id, title, abstract, authors (JSON string), category, published, year
    Turso provides:
      arxiv_id, title, authors (comma-sep), categories, primary_topic,
      update_date, abstract_preview, citation_count, influential_citations
    """
    arxiv_id = row.get("arxiv_id")
    if not arxiv_id:
        return None

    # Convert authors from comma-separated to JSON array string
    authors_raw = row.get("authors") or ""
    if authors_raw.startswith("["):
        # Already JSON — leave as is
        authors_json = authors_raw
    else:
        # Comma-separated → JSON array (take first 5)
        author_list = [a.strip() for a in authors_raw.split(",") if a.strip()][:5]
        authors_json = json.dumps(author_list)

    # Use primary_topic as category, fall back to first in categories list
    category = row.get("primary_topic") or ""
    if not category:
        cats = row.get("categories") or ""
        category = cats.split()[0] if cats else ""

    # Extract year from update_date (YYYY-MM-DD format)
    update_date = row.get("update_date") or ""
    year = 0
    if len(update_date) >= 4:
        try:
            year = int(update_date[:4])
        except ValueError:
            pass

    # Citation count (bonus data from Semantic Scholar)
    citation_count = 0
    try:
        citation_count = int(row.get("citation_count") or 0)
    except (ValueError, TypeError):
        pass

    influential = 0
    try:
        influential = int(row.get("influential_citations") or 0)
    except (ValueError, TypeError):
        pass

    return {
        "arxiv_id": arxiv_id,
        "title": (row.get("title") or "").replace("\n", " "),
        "abstract": (row.get("abstract_preview") or "").replace("\n", " "),
        "authors": authors_json,
        "category": category,
        # Raw space-separated arXiv codes ("cs.CL cs.LG").
        #
        # `category` above comes from primary_topic, which stores friendly
        # labels ("AI/ML", "NLP/Computational Linguistics") rather than arXiv
        # codes.  Anything that needs to compare against CATEGORY_GROUPS — which
        # is defined in arXiv codes — must use this field instead, or the
        # comparison silently never matches.
        "arxiv_categories": row.get("categories") or "",
        "published": update_date,
        "year": year,
        "citation_count": citation_count,
        "influential_citations": influential,
    }


# ── Phase 5: Trending papers for cold-start fallback ──────────────────────────

async def fetch_trending_by_categories(
    categories: set[str], limit: int = 10
) -> list[dict]:
    """
    Fetch recently published, high-citation papers from Turso DB
    filtered by arXiv categories. Used as Tier 0 popularity fallback
    for onboarded users with zero saves.

    Cached in-process (1 hour TTL): citation counts barely change
    minute-to-minute, and onboarding has a small fixed set of category
    combinations, so the first cold-start hit pays the ~15s LIKE-scan
    cost once and subsequent users get an instant hit.

    Filter strategy:
      Turso's `primary_topic` column stores friendly labels like
      "AI/ML" / "Computer Vision" — NOT arxiv codes — and the mapping
      from arxiv code to friendly label is not 1:1 (e.g. Vaswani's
      cs.CL paper is labeled "AI/ML" while BERT's cs.CL paper is
      labeled "NLP/Computational Linguistics"). The `categories`
      column, however, contains the real space-separated arxiv codes
      ("cs.CL cs.LG"). So we filter via LIKE on `categories`.

      Performance: LIKE '%cs.XX%' with leading wildcard skips the index,
      but Turso's `citation_count > 0` filter + ORDER BY citation_count
      narrows the scan, and trending is not a hot path.
    """
    if not categories:
        return []

    cache_key = (tuple(sorted(categories)), limit)
    cached = _TRENDING_CACHE.get(cache_key)
    if cached is not None and (time.time() - cached[0]) < _TRENDING_TTL_SECONDS:
        return cached[1]

    url = config.TURSO_URL
    token = config.TURSO_DB_TOKEN
    if not url or not token:
        return []

    cat_list = list(categories)
    # categories column is space-separated arxiv codes; arxiv codes
    # don't share substrings (no code is a substring of another), so
    # plain LIKE '%code%' is safe.
    like_clauses = " OR ".join(["categories LIKE ?" for _ in cat_list])
    sql = f"""SELECT arxiv_id, title, authors, categories, primary_topic,
                     update_date, abstract_preview, citation_count, influential_citations
              FROM papers
              WHERE ({like_clauses})
                AND citation_count > 0
              ORDER BY citation_count DESC, update_date DESC
              LIMIT ?"""

    args = [{"type": "text", "value": f"%{c}%"} for c in cat_list]
    args.append({"type": "integer", "value": str(limit)})

    pipeline_url = url.rstrip("/")
    if pipeline_url.startswith("libsql://"):
        pipeline_url = "https://" + pipeline_url[len("libsql://"):]
    elif pipeline_url.startswith("http://"):
        pipeline_url = "https://" + pipeline_url[len("http://"):]
    elif not pipeline_url.startswith("https://"):
        pipeline_url = "https://" + pipeline_url

    payload = {
        "requests": [
            {"type": "execute", "stmt": {"sql": sql, "args": args}},
            {"type": "close"},
        ]
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # Use a longer timeout than metadata fetch — full table scan
    # for citation-sorted trending against 1.6M rows can spike to
    # 15-25s on the first cold hit. Once cached, warm reads are 0ms.
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{pipeline_url}/v2/pipeline",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        # Surface response body on HTTP errors — Turso's empty-string
        # exceptions were the symptom that hid this bug for months.
        body = ""
        try:
            body = e.response.text[:500]
        except Exception:
            pass
        print(f"[turso] trending HTTP error {e.response.status_code}: {body}")
        return []
    except Exception as e:
        print(f"[turso] trending request failed: {type(e).__name__}: {e!r}")
        return []

    try:
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return []

        execute_result = results[0]
        if execute_result.get("type") == "error":
            print(f"[turso] trending query error: {execute_result.get('error')}")
            return []

        response = execute_result.get("response", {})
        result_data = response.get("result", {})
        cols = [c["name"] for c in result_data.get("cols", [])]
        rows = result_data.get("rows", [])
    except (KeyError, IndexError, TypeError) as e:
        print(f"[turso] trending parse error: {type(e).__name__}: {e!r}")
        return []

    papers = []
    for row in rows:
        values = {}
        for i, col in enumerate(cols):
            cell = row[i]
            if cell.get("type") == "null":
                values[col] = None
            else:
                values[col] = cell.get("value", "")
        paper = _to_paper_dict(values)
        if paper:
            papers.append(paper)

    print(f"[turso] trending: {len(papers)} papers in {len(categories)} categories")
    if papers:
        _TRENDING_CACHE[cache_key] = (time.time(), papers)
        # Also seed metadata cache — these papers are likely to be
        # fetched again as part of recommendations / display.
        for p in papers:
            _cache_put(p["arxiv_id"], p)
    return papers
