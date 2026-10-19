"""
Where the app's AI goes: every model call, kept per call, and daily counts of
cache use and answer quality per feature. /healthz/ai reads both.

Features, one per place the app asks a model:

    search_rewrite   rewrites a conversational search query (groq_svc.rewrite)
    search_overview  the overview above search results (groq_svc.generate_search_summary)
    explain          "In plain terms" on a paper (groq_svc.explain_paper)
    briefing         the paper briefing (briefing_svc, through app/llm.py)
    ask              Ask this paper (ask_svc, through app/llm.py)

Calls go to `llm_calls`, one row each, replicated: tokens in and out from
Groq's usage, the answering model and its place in GROQ_MODELS (0 = primary,
higher = a fallback answered), whether the reply hit the token cap, and the
requests and minute-tokens Groq said were left at that moment. Groq sends no
header for tokens used today, so the day's tokens are the sum of these rows.

Everything else worth counting goes to `ai_daily`, one number per UTC day,
feature and metric: cache_hit and cache_miss, outcomes (declined, busy, ok,
insufficient, unanswerable...), points_dropped and quotes_dropped by the
checks, reported. count() is synchronous and cheap; a loop owned by the app
lifespan flushes the counts every FLUSH_S seconds and on shutdown.
"""
from __future__ import annotations

import asyncio
import time

import aiosqlite

from app import config, errors

FEATURES = ("search_rewrite", "search_overview", "explain", "briefing", "ask")

_COLUMNS = ("feature", "provider", "model", "chars_in", "chars_out", "usd_est", "latency_ms", "status",
            "tokens_in", "tokens_out", "model_rank", "truncated", "requests_left", "tokens_left")


async def record(call) -> None:
    """Write one model call. `call` has the attributes named in _COLUMNS (missing = empty)."""
    from app import groq_svc
    window = groq_svc.windows().get(getattr(call, "model", "") or "", {})
    values = [getattr(call, c, None) for c in _COLUMNS[:12]]
    values[_COLUMNS.index("truncated")] = 1 if getattr(call, "truncated", False) else 0
    values += [window.get("requests_left"), window.get("tokens_left")]
    try:
        async with aiosqlite.connect(config.DB_PATH) as conn:
            await conn.execute(
                f"INSERT INTO llm_calls ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' * len(_COLUMNS))})", values)
            await conn.commit()
    except Exception as e:
        errors.report("ai_usage", "logging a model call failed", e)


class _Row:
    def __init__(self, **kw):
        self.__dict__.update(kw)


async def log_call(feature: str, completion=None, *, latency_ms: int = 0, status: str = "ok",
                   chars_in: int = 0, chars_out: int = 0) -> None:
    """Record a call made through groq_svc directly (search and plain summaries).

    `completion` is the groq_svc.Completion, or None when no model answered.
    """
    c = completion
    await record(_Row(feature=feature, provider="groq", model=c.model if c else "", chars_in=chars_in,
                      chars_out=chars_out if chars_out else (len(c.text) if c else 0), usd_est=0.0,
                      latency_ms=int(latency_ms), status=status,
                      tokens_in=c.tokens_in if c else None, tokens_out=c.tokens_out if c else None,
                      model_rank=c.rank if c else None, truncated=bool(c and c.truncated)))


# ── daily counts ─────────────────────────────────────────────────────────────

FLUSH_S = 60
_pending: dict[tuple[str, str, str], int] = {}
_task: asyncio.Task | None = None


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def count(feature: str, metric: str, n: int = 1) -> None:
    """Add n to today's count of `metric` for `feature`. Never blocks, never raises."""
    if n > 0:
        key = (_today(), feature, metric)
        _pending[key] = _pending.get(key, 0) + int(n)


async def flush() -> None:
    """Write pending counts. On failure they are kept for the next flush."""
    if not _pending:
        return
    items = list(_pending.items())
    _pending.clear()
    try:
        async with aiosqlite.connect(config.DB_PATH) as conn:
            for (day, feature, metric), n in items:
                cur = await conn.execute(
                    "SELECT n FROM ai_daily WHERE day = ? AND feature = ? AND metric = ?", (day, feature, metric))
                row = await cur.fetchone()
                # Whole-row replace, never UPDATE: a new id is what makes turso_sync push it.
                await conn.execute("INSERT OR REPLACE INTO ai_daily (day, feature, metric, n) VALUES (?, ?, ?, ?)",
                                   (day, feature, metric, (row[0] if row else 0) + n))
            await conn.commit()
    except Exception as e:
        for key, n in items:
            _pending[key] = _pending.get(key, 0) + n
        errors.report("ai_usage", "writing daily AI counts failed", e)


async def _loop() -> None:
    while True:
        await asyncio.sleep(FLUSH_S)
        await flush()


async def start() -> None:
    global _task
    if _task is None:
        _task = asyncio.create_task(_loop(), name="ai-usage-flush")


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

