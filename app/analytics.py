"""
What readers do, for the owner's dashboards: a log of searches, the AI
overviews shown for them, and one row per reader per day. No prompt, answer or
reader is attached to a query.

search_log        one row per search: the query (whitespace collapsed, at most
                  200 characters), its query_id (the results' clicks and saves
                  carry it in interactions.query_id), the number of results,
                  the retrieval mode, the AI rewrite when it changed the query,
                  the time taken, and owner = 1 when the searcher holds the
                  preview token. Not stored against the reader.
search_overviews  the AI overview shown for a query and its top papers. Also
                  the overview cache: the same query over the same papers is
                  answered from here for OVERVIEW_TTL_S instead of a new call.

reader_days       one row per reader (the anonymous cookie; there are no
                  accounts) per UTC day: pages, searches, feed pages, papers
                  opened, saves, passes and questions asked, and owner = 1 for
                  a browser holding the preview token, so the owner's own use
                  can be left out. Counted by observe(), called from the HTTP
                  middleware for every request, so no route has to remember.

Query text and overviews are kept RETENTION_DAYS (90): a daily housekeeping
run clears them here and in Turso, and the counts stay. All four tables are
replicated (app/turso_sync.py).
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import time

import aiosqlite

from app import ai_usage, config, errors

RETENTION_DAYS = 90
OVERVIEW_TTL_S = 7 * 24 * 3600
HOUSEKEEP_EVERY_S = 24 * 3600
OVERVIEW_PROMPT_VERSION = "o1"      # bump when the overview prompt changes

# Clear what people typed once it is RETENTION_DAYS old; run locally and in Turso.
_EXPIRE = [
    "UPDATE search_log SET query = NULL, rewritten = NULL "
    "WHERE query IS NOT NULL AND created_at < datetime('now', '-{d} day')",
    "DELETE FROM search_overviews WHERE created_at < datetime('now', '-{d} day')",
    "UPDATE llm_calls SET subject = NULL WHERE subject IS NOT NULL "
    "AND feature IN ('search_rewrite', 'search_overview') AND created_at < datetime('now', '-{d} day')",
]


# ── searches ─────────────────────────────────────────────────────────────────

async def log_search(query: str, query_id: str, n_results: int, mode: str, rewritten: str | None,
                     latency_ms: int, owner: bool) -> None:
    """One search. Never raises: a search must not fail because it couldn't be counted."""
    q = ai_usage.query_subject(query)
    if not q:
        return
    r = ai_usage.query_subject(rewritten) if rewritten and rewritten.strip() != query.strip() else None
    try:
        async with aiosqlite.connect(config.DB_PATH) as conn:
            await conn.execute(
                "INSERT INTO search_log (query, query_id, n_results, mode, rewritten, latency_ms, owner) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)", (q, query_id, n_results, mode, r, latency_ms, 1 if owner else 0))
            await conn.commit()
    except Exception as e:
        errors.report("analytics", "logging a search failed", e)


def overview_key(query: str, arxiv_ids: list[str]) -> str:
    h = hashlib.sha256()
    for part in [(ai_usage.query_subject(query) or "").lower(), OVERVIEW_PROMPT_VERSION, *arxiv_ids[:5]]:
        h.update(part.encode())
        h.update(b"\x00")
    return h.hexdigest()


async def cached_overview(query: str, arxiv_ids: list[str]) -> str | None:
    """The overview shown for this query over these papers in the last OVERVIEW_TTL_S, or None."""
    try:
        async with aiosqlite.connect(config.DB_PATH) as conn:
            cur = await conn.execute(
                "SELECT overview FROM search_overviews WHERE cache_key = ? AND overview IS NOT NULL "
                "AND created_at >= datetime('now', ?)",
                (overview_key(query, arxiv_ids), f"-{OVERVIEW_TTL_S} second"))
            row = await cur.fetchone()
        return row[0] if row else None
    except Exception as e:
        errors.report("analytics", "reading a cached overview failed", e)
        return None


async def save_overview(query: str, arxiv_ids: list[str], overview_html: str) -> None:
    try:
        async with aiosqlite.connect(config.DB_PATH) as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO search_overviews (cache_key, query, paper_ids, overview) VALUES (?, ?, ?, ?)",
                (overview_key(query, arxiv_ids), ai_usage.query_subject(query), ",".join(arxiv_ids[:5]),
                 overview_html))
            await conn.commit()
    except Exception as e:
        errors.report("analytics", "saving an overview failed", e)


# ── housekeeping ─────────────────────────────────────────────────────────────

async def housekeep() -> None:
    """Clear query text and overviews older than RETENTION_DAYS, here and in Turso."""
    sql = [s.format(d=RETENTION_DAYS) for s in _EXPIRE]
    try:
        async with aiosqlite.connect(config.DB_PATH) as conn:
            for s in sql:
                await conn.execute(s)
            await conn.commit()
    except Exception as e:
        errors.report("analytics", "clearing old search text failed", e)
    from app import turso_sync
    if not turso_sync.enabled():
        return
    try:
        # turso_sync only pushes new rows; text cleared from old ones must be
        # cleared there too, or a restore would bring it back.
        await turso_sync._execute([{"sql": s} for s in sql], timeout=60)
    except Exception as e:
        errors.report("analytics", "clearing old search text in Turso failed", e)


# ── readers ──────────────────────────────────────────────────────────────────

READER_METRICS = ("pages", "searches", "feed_pages", "opens", "saves", "passes", "asks")
FLUSH_S = 60
_PAPER_ACTION = re.compile(r"^/api/papers/[^/]+/(save|not-interested|ask)$")
_NOT_PAGES = ("/api/", "/static/", "/fig/", "/healthz", "/admin", "/favicon")
_readers: dict[tuple[str, str], dict] = {}


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def count_reader(user_id: str | None, metric: str, owner: bool = False, n: int = 1) -> None:
    """Add to today's count for this reader. Never blocks, never raises."""
    if not user_id or metric not in READER_METRICS or n <= 0:
        return
    row = _readers.setdefault((_today(), user_id[:64]), {"owner": False})
    row["owner"] = row["owner"] or bool(owner)
    row[metric] = row.get(metric, 0) + n


def observe(request, status: int, content_type: str) -> None:
    """Count one finished request against its reader, by what the request was."""
    if status >= 400:
        return
    user_id = request.cookies.get(config.COOKIE_NAME)
    if not user_id:
        return                                   # a first visit; the cookie is being set now
    from app import features
    owner = features.has_preview(request)
    path, method = request.url.path, request.method
    partial = request.headers.get("HX-Request") == "true"
    if method == "GET":
        if path == "/search" and request.query_params.get("q", "").strip():
            count_reader(user_id, "searches", owner)
        elif path == "/api/recommendations":
            count_reader(user_id, "feed_pages", owner)
        elif path.startswith("/p/") and not partial:
            count_reader(user_id, "opens", owner)
        if not partial and content_type.startswith("text/html") and not path.startswith(_NOT_PAGES):
            count_reader(user_id, "pages", owner)
    elif method == "POST":
        m = _PAPER_ACTION.match(path)
        if m:
            count_reader(user_id, {"save": "saves", "not-interested": "passes", "ask": "asks"}[m.group(1)], owner)


async def flush_readers() -> None:
    """Write pending reader counts. On failure they are kept for the next flush."""
    if not _readers:
        return
    items = list(_readers.items())
    _readers.clear()
    try:
        async with aiosqlite.connect(config.DB_PATH) as conn:
            for (day, user_id), add in items:
                cur = await conn.execute(
                    f"SELECT owner, {', '.join(READER_METRICS)} FROM reader_days WHERE day = ? AND user_id = ?",
                    (day, user_id))
                row = await cur.fetchone() or (0,) + (0,) * len(READER_METRICS)
                values = [(row[i + 1] or 0) + add.get(m, 0) for i, m in enumerate(READER_METRICS)]
                # Whole-row replace, never UPDATE: a new id is what makes turso_sync push it.
                await conn.execute(
                    f"INSERT OR REPLACE INTO reader_days (day, user_id, owner, {', '.join(READER_METRICS)}) "
                    f"VALUES (?, ?, ?, {', '.join('?' * len(READER_METRICS))})",
                    (day, user_id, 1 if (row[0] or add["owner"]) else 0, *values))
            await conn.commit()
    except Exception as e:
        for key, add in items:
            row = _readers.setdefault(key, {"owner": False})
            row["owner"] = row["owner"] or add["owner"]
            for m in READER_METRICS:
                if add.get(m):
                    row[m] = row.get(m, 0) + add[m]
        errors.report("analytics", "writing reader counts failed", e)


# ── running ──────────────────────────────────────────────────────────────────

_tasks: list[asyncio.Task] = []


async def _housekeeping_loop() -> None:
    while True:
        await housekeep()
        await asyncio.sleep(HOUSEKEEP_EVERY_S)


async def _flush_loop() -> None:
    while True:
        await asyncio.sleep(FLUSH_S)
        await flush_readers()


async def start() -> None:
    if not _tasks:
        _tasks.append(asyncio.create_task(_housekeeping_loop(), name="analytics-housekeeping"))
        _tasks.append(asyncio.create_task(_flush_loop(), name="analytics-readers"))


async def stop() -> None:
    while _tasks:
        task = _tasks.pop()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    await flush_readers()
