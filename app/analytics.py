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
            if not getattr(getattr(request, "state", None), "repeat_search", False):   # already counted
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


# ── reports for the owner's dashboards ───────────────────────────────────────

def _days(n: int) -> list[str]:
    now = time.time()
    return [time.strftime("%Y-%m-%d", time.gmtime(now - 86400 * i)) for i in range(n - 1, -1, -1)]


async def _all(conn, sql: str, *args) -> list:
    return await (await conn.execute(sql, args)).fetchall()


# A reader's first day: the earliest of their first recorded action and their
# first counted day, so readers from before reader_days began aren't "new".
_FIRST_SEEN = """
    SELECT user_id, MIN(day) AS first_day FROM (
        SELECT user_id, substr(timestamp, 1, 10) AS day FROM interactions
        UNION ALL SELECT user_id, day FROM reader_days) GROUP BY user_id"""
_OWNERS = "SELECT DISTINCT user_id FROM reader_days WHERE owner = 1"


async def readers_report(days: int = 30, include_owner: bool = False) -> dict:
    """Readers per day, what they did, and the long view from interactions."""
    owner_rd = "" if include_owner else "AND owner = 0"
    owner_ix = "" if include_owner else f"AND user_id NOT IN ({_OWNERS})"
    since = f"-{days - 1} day"
    async with aiosqlite.connect(config.DB_PATH) as conn:
        per_day = await _all(conn, f"""
            SELECT r.day, COUNT(*), SUM(f.first_day = r.day), SUM(r.pages), SUM(r.searches), SUM(r.feed_pages),
                   SUM(r.opens), SUM(r.saves), SUM(r.passes), SUM(r.asks)
            FROM reader_days r JOIN ({_FIRST_SEEN}) f ON f.user_id = r.user_id
            WHERE r.day >= date('now', ?) {owner_rd} GROUP BY r.day""", since)
        windows = {}
        for label, back in (("today", 0), ("week", 6), ("month", days - 1)):
            windows[label] = (await _all(conn, f"SELECT COUNT(DISTINCT user_id) FROM reader_days "
                                               f"WHERE day >= date('now', ?) {owner_rd}", f"-{back} day"))[0][0]
        did = (await _all(conn, f"""
            SELECT COUNT(*), SUM(s > 0), SUM(o > 0), SUM(sv > 0), SUM(a > 0), SUM(d > 1) FROM (
                SELECT user_id, SUM(searches) s, SUM(opens) o, SUM(saves) sv, SUM(asks) a, COUNT(*) d
                FROM reader_days WHERE day >= date('now', ?) {owner_rd} GROUP BY user_id)""", since))[0]
        tracked_since = (await _all(conn, "SELECT MIN(day) FROM reader_days"))[0][0]
        months = await _all(conn, f"""
            SELECT substr(timestamp, 1, 7), COUNT(DISTINCT user_id), COUNT(*), SUM(event_type = 'save')
            FROM interactions WHERE 1 = 1 {owner_ix} GROUP BY 1 ORDER BY 1""")
        depth = await _all(conn, f"""
            SELECT CASE WHEN d = 1 THEN '1 day' WHEN d <= 3 THEN '2-3 days' WHEN d <= 7 THEN '4-7 days'
                        ELSE '8+ days' END, COUNT(*) FROM (
                SELECT user_id, COUNT(DISTINCT substr(timestamp, 1, 10)) d FROM interactions
                WHERE 1 = 1 {owner_ix} GROUP BY user_id) GROUP BY 1""")
        sources = await _all(conn, f"""
            SELECT COALESCE(source, '?'), SUM(event_type = 'save'), SUM(event_type IN ('click', 'view')),
                   SUM(event_type = 'not_interested') FROM interactions WHERE 1 = 1 {owner_ix}
            GROUP BY 1 ORDER BY 2 DESC LIMIT 8""")
        first_action = (await _all(conn, f"SELECT MIN(timestamp), COUNT(DISTINCT user_id) FROM interactions "
                                         f"WHERE 1 = 1 {owner_ix}"))[0]
    by_day = {r[0]: r for r in per_day}
    rows = []
    for d in _days(days):
        r = by_day.get(d)
        rows.append({"day": d, "readers": r[1] if r else 0, "new": r[2] if r else 0,
                     "returning": (r[1] - r[2]) if r else 0,
                     **{k: (r[i] or 0) if r else 0 for i, k in enumerate(
                         ("pages", "searches", "feed_pages", "opens", "saves", "passes", "asks"), start=3)}})
    order = {"1 day": 0, "2-3 days": 1, "4-7 days": 2, "8+ days": 3}
    return {
        "include_owner": include_owner, "tracked_since": tracked_since, "windows": windows,
        "did": {"readers": did[0] or 0, "searched": did[1] or 0, "opened": did[2] or 0, "saved": did[3] or 0,
                "asked": did[4] or 0, "came_back": did[5] or 0},
        "days": rows,
        "months": [{"month": m, "readers": r, "actions": a, "saves": s or 0} for m, r, a, s in months],
        "depth": sorted(({"bucket": b, "readers": n} for b, n in depth), key=lambda x: order.get(x["bucket"], 9)),
        "sources": [{"source": s, "saves": sv or 0, "opens": o or 0, "passes": p or 0} for s, sv, o, p in sources],
        "since": first_action[0], "all_readers": first_action[1] or 0,
    }


async def searches_report(days: int = 30, include_owner: bool = False) -> dict:
    """Searches per day, what came of them, top and failing queries, and the latest."""
    owner = "" if include_owner else "AND s.owner = 0"
    since = f"-{days - 1} day"
    # A search "worked" when one of its results was opened or saved: the
    # results carry the search's query_id into interactions.
    outcome = """(SELECT MAX(event_type IN ('click', 'view')) FROM interactions i WHERE i.query_id = s.query_id) AS opened,
                 (SELECT MAX(event_type = 'save') FROM interactions i WHERE i.query_id = s.query_id) AS saved"""
    async with aiosqlite.connect(config.DB_PATH) as conn:
        per_day = await _all(conn, f"""
            SELECT substr(s.created_at, 1, 10), COUNT(*), SUM(s.n_results = 0),
                   SUM(s.mode IN ('keyword', 'arxiv_keyword')), SUM(s.rewritten IS NOT NULL),
                   CAST(AVG(s.latency_ms) AS INTEGER)
            FROM search_log s WHERE s.created_at >= date('now', ?) {owner} GROUP BY 1""", since)
        totals = (await _all(conn, f"""
            SELECT COUNT(*), SUM(n_results = 0), SUM(opened > 0), SUM(saved > 0), SUM(rewritten IS NOT NULL)
            FROM (SELECT s.*, {outcome} FROM search_log s WHERE s.created_at >= date('now', ?) {owner})""",
                              since))[0]
        top = await _all(conn, f"""
            SELECT LOWER(s.query), COUNT(*), CAST(AVG(s.n_results) AS INTEGER), SUM(opened > 0), SUM(saved > 0)
            FROM (SELECT s.*, {outcome} FROM search_log s WHERE s.query IS NOT NULL
                  AND s.created_at >= date('now', ?) {owner}) s
            GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 25""", since)
        nothing = await _all(conn, f"""
            SELECT LOWER(s.query), COUNT(*), MAX(s.created_at) FROM search_log s WHERE s.query IS NOT NULL
            AND s.n_results = 0 AND s.created_at >= date('now', ?) {owner} GROUP BY 1 ORDER BY 3 DESC LIMIT 15""",
                             since)
        latest = await _all(conn, f"""
            SELECT s.created_at, s.query, s.n_results, s.mode, s.rewritten, s.latency_ms, s.owner, {outcome},
                   (SELECT o.overview FROM search_overviews o WHERE LOWER(o.query) = LOWER(s.query)
                    ORDER BY o.id DESC LIMIT 1)
            FROM search_log s WHERE s.query IS NOT NULL {owner} ORDER BY s.id DESC LIMIT 40""")
    by_day = {r[0]: r for r in per_day}
    rows = [{"day": d, "searches": by_day[d][1] if d in by_day else 0,
             "no_results": by_day[d][2] or 0 if d in by_day else 0,
             "keyword_only": by_day[d][3] or 0 if d in by_day else 0,
             "rewritten": by_day[d][4] or 0 if d in by_day else 0,
             "avg_ms": by_day[d][5] if d in by_day else None} for d in _days(days)]
    n = totals[0] or 0
    return {
        "include_owner": include_owner, "days": rows,
        "totals": {"searches": n, "no_results": totals[1] or 0, "opened": totals[2] or 0, "saved": totals[3] or 0,
                   "rewritten": totals[4] or 0},
        "top": [{"query": q, "times": c, "avg_results": r, "opened": o or 0, "saved": s or 0}
                for q, c, r, o, s in top],
        "nothing": [{"query": q, "times": c, "last": t} for q, c, t in nothing],
        "latest": [{"at": a, "query": q, "results": r, "mode": m, "rewritten": rw, "ms": ms, "owner": bool(ow),
                    "opened": bool(op), "saved": bool(sv), "overview": ov}
                   for a, q, r, m, rw, ms, ow, op, sv, ov in latest],
        "retention_days": RETENTION_DAYS,
    }


async def ai_history(limit: int = 60) -> dict:
    """Every AI call grouped by the paper or search it was for, with what was written."""
    async with aiosqlite.connect(config.DB_PATH) as conn:
        papers = await _all(conn, f"""
            SELECT subject, COUNT(*), SUM(feature = 'explain'), SUM(feature = 'briefing'), SUM(feature = 'ask'),
                   SUM(status != 'ok'), COALESCE(SUM(tokens_in), 0) + COALESCE(SUM(tokens_out), 0), MAX(created_at)
            FROM llm_calls WHERE subject IS NOT NULL AND feature IN ('explain', 'briefing', 'ask')
            GROUP BY subject ORDER BY MAX(id) DESC LIMIT {int(limit)}""")
        ids = [p[0] for p in papers]
        marks = ",".join("?" * len(ids)) or "''"
        explanations = dict(await _all(conn, f"""
            SELECT arxiv_id, explanation FROM paper_explanations WHERE id IN (
                SELECT MAX(id) FROM paper_explanations WHERE arxiv_id IN ({marks}) GROUP BY arxiv_id)""", *ids))
        briefings = dict(await _all(conn, f"""
            SELECT arxiv_id, status FROM paper_briefings WHERE id IN (
                SELECT MAX(id) FROM paper_briefings WHERE arxiv_id IN ({marks}) GROUP BY arxiv_id)""", *ids))
        written = dict(await _all(conn, f"""
            SELECT arxiv_id, COUNT(*) FROM paper_briefings WHERE arxiv_id IN ({marks}) GROUP BY arxiv_id""", *ids))
        titles = dict(await _all(conn, f"SELECT arxiv_id, title FROM paper_metadata WHERE arxiv_id IN ({marks})",
                                 *ids))
        queries = await _all(conn, f"""
            SELECT subject, SUM(feature = 'search_rewrite'), SUM(feature = 'search_overview'), SUM(status != 'ok'),
                   COALESCE(SUM(tokens_in), 0) + COALESCE(SUM(tokens_out), 0), MAX(created_at)
            FROM llm_calls WHERE subject IS NOT NULL AND feature IN ('search_rewrite', 'search_overview')
            GROUP BY subject ORDER BY MAX(id) DESC LIMIT {int(limit)}""")
        rewrites = dict(await _all(conn, """
            SELECT LOWER(query), rewritten FROM search_log WHERE rewritten IS NOT NULL AND id IN (
                SELECT MAX(id) FROM search_log WHERE rewritten IS NOT NULL GROUP BY LOWER(query))"""))
        overviews = dict(await _all(conn, """
            SELECT LOWER(query), overview FROM search_overviews WHERE overview IS NOT NULL AND id IN (
                SELECT MAX(id) FROM search_overviews GROUP BY LOWER(query))"""))
    missing = [i for i in ids if i not in titles]
    if missing:
        try:
            from app import local_meta
            found = await asyncio.to_thread(local_meta.fetch_rows, missing)
            titles.update({r["arxiv_id"]: r.get("title") or "" for r in found or []})
        except Exception:
            pass
    return {
        "papers": [{"arxiv_id": a, "title": titles.get(a) or "", "calls": n, "explain": e or 0, "briefing": b or 0,
                    "ask": k or 0, "failed": f or 0, "tokens": t, "last": last,
                    "plain": explanations.get(a), "briefing_status": briefings.get(a),
                    "briefings_written": written.get(a, 0)}
                   for a, n, e, b, k, f, t, last in papers],
        "queries": [{"query": q, "rewrites": r or 0, "overviews": o or 0, "failed": f or 0, "tokens": t, "last": last,
                     "rewritten": rewrites.get(q.lower()), "overview": overviews.get(q.lower())}
                    for q, r, o, f, t, last in queries],
        "retention_days": RETENTION_DAYS,
    }
