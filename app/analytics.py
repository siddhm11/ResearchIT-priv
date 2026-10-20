"""
What readers do, for the owner's dashboards: a log of searches, the AI
overviews shown for them, and (app/analytics.py, next part) one row per reader
per day. No prompt, answer or reader is attached to a query.

search_log        one row per search: the query (whitespace collapsed, at most
                  200 characters), its query_id (the results' clicks and saves
                  carry it in interactions.query_id), the number of results,
                  the retrieval mode, the AI rewrite when it changed the query,
                  the time taken, and owner = 1 when the searcher holds the
                  preview token. Not stored against the reader.
search_overviews  the AI overview shown for a query and its top papers. Also
                  the overview cache: the same query over the same papers is
                  answered from here for OVERVIEW_TTL_S instead of a new call.

Query text and overviews are kept RETENTION_DAYS (90): a daily housekeeping
run clears them here and in Turso, and the counts stay. All three tables are
replicated (app/turso_sync.py).
"""
from __future__ import annotations

import asyncio
import hashlib
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


_tasks: list[asyncio.Task] = []


async def _housekeeping_loop() -> None:
    while True:
        await housekeep()
        await asyncio.sleep(HOUSEKEEP_EVERY_S)


async def start() -> None:
    if not _tasks:
        _tasks.append(asyncio.create_task(_housekeeping_loop(), name="analytics-housekeeping"))


async def stop() -> None:
    while _tasks:
        task = _tasks.pop()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
