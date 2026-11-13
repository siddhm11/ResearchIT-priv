"""
A daily operations digest that survives restarts (audit F25).

Logs are prints that vanish with the container, and /healthz/runtime forgets
everything on a restart, so an outage overnight left no trace by morning. This
keeps one number per UTC day per metric in `ops_daily`, replicated to Turso:

    error:<module>   errors that module reported (app/errors.py), counted
    boot             times the app started
    <stage>_p50_ms / <stage>_p95_ms
                     search and feed times, the latest since-restart value of
                     the day (a gauge, overwritten, not summed)
    shard_late_<s>   shards that missed the feed's deadline since the restart

Counts and gauges are kept in memory and written once a minute, like
ai_usage.count. Nothing here may raise into a request.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque

import aiosqlite

from app import config, runtime_stats

FLUSH_S = 60
_counts: dict[tuple[str, str], int] = {}
_task: asyncio.Task | None = None
# The last few errors, for /healthz/runtime. Memory only.
recent: deque = deque(maxlen=20)

# (metric prefix, runtime_stats kind, key) sampled into the digest at each flush.
_GAUGES = (("search", "search", "total_ms"), ("feed_build", "feed_tier1", "total_ms"),
           ("feed_first_page", "feed_first_page", "request_ms"))


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def count(metric: str, n: int = 1) -> None:
    """Add n to today's `metric`. Never blocks, never raises."""
    key = (_today(), metric)
    _counts[key] = _counts.get(key, 0) + int(n)


def error(module: str, message: str) -> None:
    """Called by errors.report for every swallowed exception."""
    count(f"error:{module}")
    recent.append({"at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                   "module": module, "message": message[:300]})


def by_module() -> dict[str, int]:
    """Errors per module since the restart, for /healthz/runtime."""
    return {k: v for k, v in runtime_stats.counters().items() if k.startswith("error:")}


def _gauges() -> dict[str, int]:
    out: dict[str, int] = {}
    timings = runtime_stats.timings()
    for prefix, kind, key in _GAUGES:
        stat = timings.get(kind, {}).get(key)
        if stat:
            out[f"{prefix}_p50_ms"], out[f"{prefix}_p95_ms"] = int(stat["p50"]), int(stat["p95"])
    for name, n in runtime_stats.counters().items():
        if name.startswith("shard_late_"):
            out[name] = n
    return out


async def flush() -> None:
    """Write pending counts (added) and gauges (replaced). Failed counts wait for the next flush."""
    items = list(_counts.items())
    _counts.clear()
    gauges = _gauges()
    if not items and not gauges:
        return
    day = _today()
    try:
        async with aiosqlite.connect(config.DB_PATH) as conn:
            for (d, metric), n in items:
                cur = await conn.execute("SELECT n FROM ops_daily WHERE day = ? AND metric = ?", (d, metric))
                row = await cur.fetchone()
                # Whole-row replace, never UPDATE: a new id is what makes turso_sync push it.
                await conn.execute("INSERT OR REPLACE INTO ops_daily (day, metric, n) VALUES (?, ?, ?)",
                                   (d, metric, (row[0] if row else 0) + n))
            for metric, n in gauges.items():
                cur = await conn.execute("SELECT n FROM ops_daily WHERE day = ? AND metric = ?", (day, metric))
                row = await cur.fetchone()
                if row is None or row[0] != n:
                    await conn.execute("INSERT OR REPLACE INTO ops_daily (day, metric, n) VALUES (?, ?, ?)",
                                       (day, metric, n))
            await conn.commit()
    except Exception as e:      # not errors.report: a failing digest must not count itself
        for key, n in items:
            _counts[key] = _counts.get(key, 0) + n
        print(f"[ops_digest] writing the daily digest failed: {e}")


async def last_days(days: int = 7) -> dict[str, dict[str, int]]:
    """{day: {metric: n}} for the owner's dashboard, newest first."""
    async with aiosqlite.connect(config.DB_PATH) as conn:
        rows = await (await conn.execute(
            "SELECT day, metric, n FROM ops_daily WHERE day >= date('now', ?) ORDER BY day DESC",
            (f"-{days - 1} day",))).fetchall()
    out: dict[str, dict[str, int]] = {}
    for day, metric, n in rows:
        out.setdefault(day, {})[metric] = n
    for (day, metric), n in _counts.items():
        out.setdefault(day, {})[metric] = out.get(day, {}).get(metric, 0) + n
    return out


async def _loop() -> None:
    while True:
        await asyncio.sleep(FLUSH_S)
        await flush()


async def start() -> None:
    global _task
    count("boot")
    if _task is None:
        _task = asyncio.create_task(_loop(), name="ops-digest-flush")


async def stop() -> None:
    global _task
    if _task is not None:
        _task.cancel()
        try:
            await _task
        except (asyncio.CancelledError, Exception):
            pass
        _task = None
    await flush()
