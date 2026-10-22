"""
Prepare new papers' figures before anyone opens them (switch FIGURE_PREWARM).

Every few hours it asks Turso for papers dated in the last two days, skips the
ones already read, and for the rest runs paper_source_svc.ensure() and makes
the first figure's thumbnail, so new papers show a figure in the feed straight
away. It runs INSIDE the Space on purpose: turso_sync only pushes after boot,
so rows a GitHub Action wrote to Turso would never reach a running Space
(trap T2 in docs/redesign/03-start-here-agent-handoff.md).

Polite by construction: every request goes through paper_source_svc's shared
pacer (1 every 2 s, bursts of 3), a run stops when the breaker opens, and a run
handles at most FIGURE_PREWARM_BATCH papers. A file lock keeps it to one
runner per container. Owned by the app lifespan, like DiscoveryRuntime.
"""
from __future__ import annotations

import asyncio
import fcntl
import os
import time
from pathlib import Path

from app import config, errors, features, licences, paper_source_svc

SWITCH = "FIGURE_PREWARM"
INTERVAL_S = int(os.getenv("FIGURE_PREWARM_INTERVAL_S", "21600"))
BATCH = int(os.getenv("FIGURE_PREWARM_BATCH", "1500"))
FIRST_RUN_DELAY_S = int(os.getenv("FIGURE_PREWARM_FIRST_DELAY_S", "120"))
LOCK_PATH = Path(os.getenv("FIGURE_PREWARM_LOCK", "/tmp/researchit-figure-prewarm.lock"))

_state: dict = {"scheduler": "off", "last_run": None, "next_run": None,
                "done": 0, "skipped": 0, "failed": 0, "stopped_by_breaker": False}
_task: asyncio.Task | None = None
_lock_handle = None


async def recent_ids(days: int = 2) -> list[tuple[str, str]]:
    """[(arxiv_id, licence code)] for papers Turso dates in the last `days` days."""
    if not (config.TURSO_URL and config.TURSO_DB_TOKEN):
        return []
    from app.turso_sync import _execute
    since = f"date('now', '-{int(days)} day')"
    try:
        rows = (await _execute([{"sql": (
            "SELECT p.arxiv_id, COALESCE(l.licence, '') FROM papers p "
            "LEFT JOIN paper_licences l ON l.arxiv_id = p.arxiv_id "
            f"WHERE p.update_date >= {since}")}]))[0]
    except Exception:
        # paper_licences arrives with the daily harvest; without it, no order.
        rows = (await _execute([{"sql": f"SELECT arxiv_id, '' FROM papers WHERE update_date >= {since}"}]))[0]
    return [(str(a), str(c or "")) for a, c in rows]


async def _thumbnail(source) -> bool:
    from app.routers import figures
    for n in range(len(source.figures)):
        if figures.figure_url(source, n):
            return await figures.prepare(source, n)
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
        try:
            source = await paper_source_svc.ensure(arxiv_id)
            if source is not None and source.ok and features.active("FIGURES"):
                await _thumbnail(source)
            counts["done" if source is not None else "failed"] += 1
        except Exception as e:                       # one paper never stops the run
            counts["failed"] += 1
            errors.report("figure_prewarm", f"warming {arxiv_id} failed", e)
    return counts


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
    global _task
    if features.mode(SWITCH) != "on" or _task is not None:
        return
    if not _take_lock():
        _state["scheduler"] = "blocked: another runner holds the lock"
        return
    _state["scheduler"] = "starting"
    _task = asyncio.create_task(_loop(), name="figure-prewarm")


async def stop() -> None:
    global _task, _lock_handle
    if _task is not None:
        _task.cancel()
        try:
            await _task
        except (asyncio.CancelledError, Exception):
            pass
        _task = None
    if _lock_handle is not None:
        try:
            fcntl.flock(_lock_handle, fcntl.LOCK_UN)
            _lock_handle.close()
        except OSError:
            pass
        _lock_handle = None
    _state["scheduler"] = "off"


def status() -> dict:
    return {"switch": features.mode(SWITCH), "interval_s": INTERVAL_S, "batch": BATCH, **_state}
