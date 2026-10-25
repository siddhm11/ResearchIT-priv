"""
Read papers and prepare a figure before anyone opens them (switch FIGURE_PREWARM).

Two sources of papers, both read the same way (paper_source_svc.ensure, then
the first figure that may be shown):

  shown     papers people were just shown: every feed page and the top
            SEARCH_TOP search results call enqueue(). Feed cards and search
            thumbnails only use what is already saved, so a paper read now
            shows its figure the next time it appears.
  recent    every few hours, papers Turso dates in the last two days, so new
            papers have a figure the first time anyone sees them. It runs
            INSIDE the Space on purpose: turso_sync only pushes after boot, so
            rows a GitHub Action wrote to Turso would never reach a running
            Space (trap T2 in docs/redesign/03-start-here-agent-handoff.md).

Polite by construction, because nobody is waiting for any of it: every
request takes paper_source_svc's background lane, one request per 15 s at
most (arXiv's robots.txt Crawl-delay), on top of the pacer readers share.
Work stops while the breaker is open and at DAILY_MAX papers a day. A file
lock keeps it to one runner per container. Owned by the app lifespan, like
DiscoveryRuntime.
"""
from __future__ import annotations

import asyncio
import fcntl
import os
import time
from collections import deque
from pathlib import Path

from app import config, errors, features, licences, paper_source_svc

SWITCH = "FIGURE_PREWARM"
INTERVAL_S = int(os.getenv("FIGURE_PREWARM_INTERVAL_S", "21600"))
# At two requests a paper and one per 15 s, 300 papers take about 2.5 hours.
BATCH = int(os.getenv("FIGURE_PREWARM_BATCH", "300"))
DAILY_MAX = int(os.getenv("FIGURE_PREWARM_DAILY", "800"))
QUEUE_MAX = 300
SEARCH_TOP = 5
FIRST_RUN_DELAY_S = int(os.getenv("FIGURE_PREWARM_FIRST_DELAY_S", "120"))
LOCK_PATH = Path(os.getenv("FIGURE_PREWARM_LOCK", "/tmp/researchit-figure-prewarm.lock"))

_state: dict = {"scheduler": "off", "last_run": None, "next_run": None,
                "done": 0, "skipped": 0, "failed": 0, "stopped_by_breaker": False}
_task: asyncio.Task | None = None
_drain_task: asyncio.Task | None = None
_lock_handle = None
_queue: deque[str] = deque()
_queued: set[str] = set()
_wake: asyncio.Event | None = None
_day = {"date": "", "read": 0, "shown_done": 0}


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _budget_left() -> bool:
    if _day["date"] != _today():
        _day.update(date=_today(), read=0, shown_done=0)
    return _day["read"] < DAILY_MAX


def enqueue(arxiv_ids: list[str]) -> int:
    """Queue papers people were just shown, for reading in the background.

    Cheap and synchronous, safe to call from a request handler. Papers already
    queued are skipped; past QUEUE_MAX the oldest are dropped, since the newest
    are the likeliest to be shown again. Returns how many were added.
    """
    if features.mode(SWITCH) != "on":
        return 0
    added = 0
    for aid in arxiv_ids:
        if not paper_source_svc.ID_PATTERN.match(aid or "") or aid in _queued:
            continue
        if len(_queue) >= QUEUE_MAX:
            _queued.discard(_queue.popleft())
        _queue.append(aid)
        _queued.add(aid)
        added += 1
    if added and _wake is not None:
        _wake.set()
    return added


async def warm(arxiv_id: str) -> str:
    """Read one paper and prepare its first figure: done, skipped or failed."""
    saved = await paper_source_svc.get(arxiv_id)
    if paper_source_svc.needs_fetch(saved):
        if not _budget_left():
            return "skipped"
        _day["read"] += 1
        source = await paper_source_svc.ensure(arxiv_id, background=True)
    else:
        source = saved
    if source is None or not source.ok:
        return "failed" if source is None else "skipped"
    if features.active("FIGURES"):
        await _thumbnail(source)
    return "done"


# The daily job appends new papers at the end of Turso's papers table, so the
# newest rows by rowid are the new papers. Filtering update_date instead (no
# index) read all 1.97M rows on every run, every 6 hours and at every restart:
# about a quarter of the free plan's 500M monthly reads on its own, and part of
# why Turso blocked all reads on 2026-10-24.
RECENT_ROWS = 5000          # the daily job adds about 1,000 a day


async def recent_ids(days: int = 2) -> list[tuple[str, str]]:
    """[(arxiv_id, licence code)] for the newest papers Turso dates in the last `days` days."""
    if not (config.TURSO_URL and config.TURSO_DB_TOKEN):
        return []
    from app.turso_sync import read      # moves to the mirror while the primary is blocked
    since = f"date('now', '-{int(days)} day')"
    newest = f"p.rowid > (SELECT MAX(rowid) FROM papers) - {RECENT_ROWS}"
    try:
        rows = (await read([{"sql": (
            "SELECT p.arxiv_id, COALESCE(l.licence, '') FROM papers p "
            "LEFT JOIN paper_licences l ON l.arxiv_id = p.arxiv_id "
            f"WHERE {newest} AND p.update_date >= {since}")}]))[0]
    except Exception:
        # paper_licences arrives with the daily harvest; without it, no order.
        rows = (await read([{"sql": f"SELECT arxiv_id, '' FROM papers p WHERE {newest} "
                                        f"AND update_date >= {since}"}]))[0]
    return [(str(a), str(c or "")) for a, c in rows]


async def _thumbnail(source) -> bool:
    from app.routers import figures
    for n in range(len(source.figures)):
        if figures.figure_url(source, n):
            return await figures.prepare(source, n, background=True)
    return False


async def run_once(limit: int | None = None) -> dict:
    """One pass over recent papers. Returns this run's counts."""
    limit = BATCH if limit is None else limit
    counts = {"done": 0, "skipped": 0, "failed": 0, "stopped_by_breaker": False}
    candidates = await recent_ids()
    # Papers whose figures may be shown first; within that, newest ids first.
    candidates.sort(key=lambda r: (not licences.allows_figures(r[1]), r[0]), reverse=False)
    saved = await paper_source_svc.get_many([a for a, _ in candidates])
    todo = [a for a, _ in candidates if a not in saved or saved[a].status == "pending"]
    counts["skipped"] = len(candidates) - len(todo)
    for arxiv_id in todo[:limit]:
        if paper_source_svc.breaker_open():
            counts["stopped_by_breaker"] = True
            break
        if not _budget_left():
            break
        try:
            result = await warm(arxiv_id)
            counts["done" if result == "done" else "failed" if result == "failed" else "skipped"] += 1
        except Exception as e:                       # one paper never stops the run
            counts["failed"] += 1
            errors.report("figure_prewarm", f"warming {arxiv_id} failed", e)
    return counts


async def _drain() -> None:
    """Read queued papers one at a time, for as long as the app runs."""
    while True:
        if not _queue:
            _wake.clear()
            await _wake.wait()
            continue
        if paper_source_svc.breaker_open() or not _budget_left():
            await asyncio.sleep(60)
            continue
        arxiv_id = _queue.popleft()
        _queued.discard(arxiv_id)
        try:
            if await warm(arxiv_id) == "done":
                _day["shown_done"] += 1
        except Exception as e:                       # one paper never stops the queue
            errors.report("figure_prewarm", f"warming {arxiv_id} failed", e)


async def _loop() -> None:
    await asyncio.sleep(FIRST_RUN_DELAY_S)
    while True:
        _state["scheduler"] = "running"
        try:
            counts = await run_once()
            _state.update(counts)
        except Exception as e:
            errors.report("figure_prewarm", "a warm-up run failed", e)
        _state["last_run"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        _state["next_run"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() + INTERVAL_S))
        _state["scheduler"] = "waiting"
        await asyncio.sleep(INTERVAL_S)


def _take_lock() -> bool:
    global _lock_handle
    try:
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        handle = LOCK_PATH.open("a")
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        return False
    _lock_handle = handle
    return True


async def start() -> None:
    global _task, _drain_task, _wake
    if features.mode(SWITCH) != "on" or _task is not None:
        return
    if not _take_lock():
        _state["scheduler"] = "blocked: another runner holds the lock"
        return
    _state["scheduler"] = "starting"
    _wake = asyncio.Event()
    if _queue:
        _wake.set()
    _task = asyncio.create_task(_loop(), name="figure-prewarm")
    _drain_task = asyncio.create_task(_drain(), name="figure-prewarm-shown")


async def stop() -> None:
    global _task, _drain_task, _lock_handle, _wake
    for task in (_task, _drain_task):
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
    _task = _drain_task = _wake = None
    if _lock_handle is not None:
        try:
            fcntl.flock(_lock_handle, fcntl.LOCK_UN)
            _lock_handle.close()
        except OSError:
            pass
        _lock_handle = None
    _state["scheduler"] = "off"


def status() -> dict:
    _budget_left()
    return {"switch": features.mode(SWITCH), "interval_s": INTERVAL_S, "batch": BATCH, **_state,
            "background_every_s": paper_source_svc.BACKGROUND_INTERVAL_S, "queued": len(_queue),
            "read_today": _day["read"], "daily_max": DAILY_MAX, "shown_warmed_today": _day["shown_done"]}
