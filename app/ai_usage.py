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
"""
from __future__ import annotations

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
