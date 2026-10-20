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
            "tokens_in", "tokens_out", "model_rank", "truncated", "subject", "requests_left", "tokens_left")
SEARCH_FEATURES = ("search_rewrite", "search_overview")


def query_subject(query: str | None) -> str | None:
    """A search query as stored: whitespace collapsed, at most 200 characters."""
    q = " ".join((query or "").split())
    return q[:200] or None


async def record(call) -> None:
    """Write one model call. `call` has the attributes named in _COLUMNS (missing = empty)."""
    from app import groq_svc
    window = groq_svc.windows().get(getattr(call, "model", "") or "", {})
    values = [getattr(call, c, None) for c in _COLUMNS[:13]]
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
                   chars_in: int = 0, chars_out: int = 0, subject: str | None = None) -> None:
    """Record a call made through groq_svc directly (search and plain summaries).

    `completion` is the groq_svc.Completion, or None when no model answered.
    """
    c = completion
    await record(_Row(feature=feature, provider="groq", model=c.model if c else "", chars_in=chars_in,
                      chars_out=chars_out if chars_out else (len(c.text) if c else 0), usd_est=0.0,
                      latency_ms=int(latency_ms), status=status,
                      tokens_in=c.tokens_in if c else None, tokens_out=c.tokens_out if c else None,
                      model_rank=c.rank if c else None, truncated=bool(c and c.truncated),
                      subject=subject))


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


# ── reading it back ──────────────────────────────────────────────────────────

_STATUSES = ("ok", "failed", "unavailable", "unparsable", "over_budget")


async def summary(days: int = 14) -> dict:
    """Today per feature and per model, Groq's live windows, and daily history.

    Counts only: never a prompt, an answer or a reader. Counts not yet flushed
    are included, so the numbers are live.
    """
    from app import groq_svc, llm
    today = _today()
    async with aiosqlite.connect(config.DB_PATH) as conn:
        by_feature = await (await conn.execute(
            "SELECT feature, COUNT(*), " + ", ".join(f"SUM(status = '{s}')" for s in _STATUSES) + ", "
            "SUM(status = 'ok' AND COALESCE(model_rank, 0) > 0), SUM(COALESCE(truncated, 0)), "
            "COALESCE(SUM(tokens_in), 0), COALESCE(SUM(tokens_out), 0), CAST(AVG(latency_ms) AS INTEGER) "
            "FROM llm_calls WHERE created_at >= date('now') GROUP BY feature")).fetchall()
        by_model = await (await conn.execute(
            "SELECT model, COUNT(*), COALESCE(SUM(tokens_in), 0) + COALESCE(SUM(tokens_out), 0) "
            "FROM llm_calls WHERE created_at >= date('now') AND COALESCE(model, '') != '' GROUP BY model")).fetchall()
        counted = await (await conn.execute(
            "SELECT feature, metric, n FROM ai_daily WHERE day = ?", (today,))).fetchall()
        history = await (await conn.execute(
            "SELECT substr(created_at, 1, 10), feature, COUNT(*), "
            "COALESCE(SUM(tokens_in), 0) + COALESCE(SUM(tokens_out), 0) FROM llm_calls "
            "WHERE created_at >= date('now', ?) GROUP BY 1, 2 ORDER BY 1, 2", (f"-{days - 1} day",))).fetchall()

    features: dict[str, dict] = {f: {"calls": 0, **{s: 0 for s in _STATUSES}, "fallback": 0, "truncated": 0,
                                     "tokens_in": 0, "tokens_out": 0, "avg_ms": None, "counts": {}}
                                 for f in FEATURES}
    for row in by_feature:
        f = features.setdefault(row[0], {"counts": {}})
        n = len(_STATUSES)
        f.update(calls=row[1], **{s: row[2 + i] or 0 for i, s in enumerate(_STATUSES)},
                 fallback=row[2 + n] or 0, truncated=row[3 + n] or 0, tokens_in=row[4 + n],
                 tokens_out=row[5 + n], avg_ms=row[6 + n])
    for feature, metric, n in counted:
        features.setdefault(feature, {"counts": {}})["counts"][metric] = n
    for (day, feature, metric), n in _pending.items():
        if day == today:
            c = features.setdefault(feature, {"counts": {}})["counts"]
            c[metric] = c.get(metric, 0) + n

    windows = groq_svc.windows()
    models = {}
    for m in list(config.GROQ_MODELS) + [r[0] for r in by_model if r[0] not in config.GROQ_MODELS]:
        calls, tokens = next(((r[1], r[2]) for r in by_model if r[0] == m), (0, 0))
        w = windows.get(m)
        if w:
            at = w.pop("at", time.time())
            w["age_s"] = round(time.time() - at)
        models[m] = {"calls": calls, "tokens": tokens,
                     "share_of_daily_tokens": round(tokens / config.GROQ_TOKENS_PER_DAY, 4)
                     if config.GROQ_TOKENS_PER_DAY else None,
                     "window": w}
    return {
        "day": today,
        "limits": {"requests_per_day": config.GROQ_REQUESTS_PER_DAY, "tokens_per_day": config.GROQ_TOKENS_PER_DAY,
                   "tokens_per_minute": config.GROQ_TOKENS_PER_MINUTE, "per": "model"},
        "features": features,
        "models": models,
        "budget": await llm.today(),
        "history": [{"day": d, "feature": f, "calls": c, "tokens": t} for d, f, c, t in history],
    }



# ── the dashboard (/admin/ai) ────────────────────────────────────────────────

LABELS = {"search_rewrite": "Search rewrite", "search_overview": "Search overview", "explain": "In plain terms",
          "briefing": "Briefing", "ask": "Ask this paper"}
# Features whose answers are cached, so a hit rate means something.
CACHED = ("explain", "briefing", "ask")
# What each outcome count is called on the page; anything else shows its own name.
METRIC_LABELS = {"written": "written", "declined": "declined", "trimmed": "trimmed", "skipped": "skipped (no call)",
                 "rewritten": "rewritten", "unchanged": "unchanged", "discarded": "discarded", "shown": "shown",
                 "empty": "empty", "ok": "ok", "insufficient": "declined", "failed": "failed", "busy": "model busy",
                 "points_dropped": "points dropped", "answer": "answered", "unanswerable": "not in paper",
                 "nosupport": "unsupported", "cap": "reader at cap", "error": "error", "quotes_kept": "quotes kept",
                 "quotes_dropped": "quotes dropped", "cites_dropped": "cites dropped", "reported": "reported",
                 "no_text": "full text unavailable"}


def _pct(part: float, whole: float) -> int | None:
    return round(100 * part / whole) if whole else None


def _insights(features: dict, models: dict, history_share: dict) -> list[dict]:
    """Plain sentences about what stands out. level: good | warning | critical | info."""
    out = []
    calls = sum(f["calls"] for f in features.values())
    if history_share:
        top, share = max(history_share.items(), key=lambda kv: kv[1])
        note = " It isn't cached, so a repeated search asks the model again." if top in ("search_overview",
                                                                                         "search_rewrite") else ""
        out.append({"level": "warning" if note and share >= 40 else "info",
                    "text": f"{LABELS.get(top, top)} used {share}% of AI tokens in the last 14 days.{note}"})
    for m, v in models.items():
        share = round(100 * (v["share_of_daily_tokens"] or 0))
        if share >= 80:
            out.append({"level": "critical", "text": f"{m} has used {share}% of today's token limit."})
        elif share >= 50:
            out.append({"level": "warning", "text": f"{m} has used {share}% of today's token limit."})
        w = v["window"] or {}
        if w.get("tokens_limit") and w.get("tokens_left") is not None and (w.get("age_s") or 0) < 300 \
                and w["tokens_left"] < 0.2 * w["tokens_limit"]:
            out.append({"level": "warning", "text": f"{m} was nearly out of tokens for the minute "
                                                    f"({w['tokens_left']:,} of {w['tokens_limit']:,} left)."})
    answered = sum(f["ok"] for f in features.values())
    fallback = sum(f["fallback"] for f in features.values())
    if answered >= 5 and fallback / answered > 0.1:
        out.append({"level": "warning", "text": f"{_pct(fallback, answered)}% of answers came from a backup model: "
                                                "the primary was busy or benched."})
    busy = sum(f["unavailable"] for f in features.values())
    if busy:
        out.append({"level": "warning", "text": f"{busy} call{'s' if busy != 1 else ''} found every model busy today."})
    failed = sum(f["failed"] + f["unparsable"] for f in features.values())
    if failed:
        out.append({"level": "warning", "text": f"{failed} call{'s' if failed != 1 else ''} failed or came back "
                                                "unusable today."})
    for f in CACHED:
        c = features[f]["counts"]
        hits, misses = c.get("cache_hit", 0), c.get("cache_miss", 0)
        if hits + misses >= 5:
            rate = _pct(hits, hits + misses)
            out.append({"level": "good" if rate >= 50 else "info",
                        "text": f"{LABELS[f]} came from the cache {rate}% of the time today."})
    b = features["briefing"]["counts"]
    made = b.get("ok", 0) + b.get("insufficient", 0) + b.get("failed", 0)
    if made >= 3 and b.get("insufficient", 0) / made >= 0.3:
        out.append({"level": "warning", "text": f"{_pct(b['insufficient'], made)}% of new briefings were declined."})
    reports = sum(f["counts"].get("reported", 0) for f in features.values())
    if reports:
        out.append({"level": "warning", "text": f"{reports} reader report{'s' if reports != 1 else ''} today."})
    if calls == 0:
        out.append({"level": "info", "text": "No AI calls yet today."})
    elif not any(i["level"] in ("warning", "critical") for i in out):
        out.append({"level": "good", "text": "Nothing needs attention today."})
    return out


async def dashboard(days: int = 14) -> dict:
    """Everything /admin/ai shows, from summary() and nothing else."""
    s = await summary(days)
    features = s["features"]
    rows = []
    for f in FEATURES:
        v = features[f]
        counts = dict(v["counts"])
        hits, misses = counts.pop("cache_hit", 0), counts.pop("cache_miss", 0)
        tokens = v["tokens_in"] + v["tokens_out"]
        rows.append({
            "feature": f, "label": LABELS[f], "calls": v["calls"], "ok_pct": _pct(v["ok"], v["calls"]),
            "fallback": v["fallback"], "truncated": v["truncated"],
            "failures": v["failed"] + v["unavailable"] + v["unparsable"], "over_budget": v["over_budget"],
            "tokens": tokens, "avg_tokens": round(tokens / v["calls"]) if v["calls"] else None,
            "tokens_in": v["tokens_in"], "tokens_out": v["tokens_out"], "avg_ms": v["avg_ms"],
            "cache_pct": _pct(hits, hits + misses) if f in CACHED else None, "cache_n": hits + misses,
            "outcomes": [(METRIC_LABELS.get(k, k), n) for k, n in sorted(counts.items(), key=lambda kv: -kv[1])],
        })
    limits = s["limits"]
    models = []
    for m, v in s["models"].items():
        w = v["window"] or {}
        used_req = (w["requests_limit"] - w["requests_left"]) if w.get("requests_limit") is not None \
            and w.get("requests_left") is not None else None
        models.append({"model": m, "primary": m == (list(config.GROQ_MODELS) or [""])[0], "calls": v["calls"],
                       "tokens": v["tokens"], "tokens_limit": limits["tokens_per_day"],
                       "tokens_pct": _pct(v["tokens"], limits["tokens_per_day"]),
                       "requests_used": used_req, "requests_limit": w.get("requests_limit"),
                       "minute_left": w.get("tokens_left"), "minute_limit": w.get("tokens_limit"),
                       "seen_s": w.get("age_s")})
    # The 14-day grid, every day present, oldest first.
    today = time.time()
    day_list = [time.strftime("%Y-%m-%d", time.gmtime(today - 86400 * i)) for i in range(days - 1, -1, -1)]
    grid = {d: {f: 0 for f in FEATURES} for d in day_list}
    totals = {f: 0 for f in FEATURES}
    for h in s["history"]:
        if h["day"] in grid and h["feature"] in grid[h["day"]]:
            grid[h["day"]][h["feature"]] += h["tokens"] or 0
            totals[h["feature"]] += h["tokens"] or 0
    all_tokens = sum(totals.values())
    share = {f: _pct(t, all_tokens) for f, t in totals.items() if t} if all_tokens else {}
    calls = sum(r["calls"] for r in rows)
    answered = sum(features[f]["ok"] for f in FEATURES)
    hits = sum(features[f]["counts"].get("cache_hit", 0) for f in CACHED)
    misses = sum(features[f]["counts"].get("cache_miss", 0) for f in CACHED)
    return {
        "day": s["day"], "limits": limits, "budget": s["budget"],
        "kpis": {"calls": calls, "tokens": sum(r["tokens"] for r in rows),
                 "fallback_pct": _pct(sum(r["fallback"] for r in rows), answered),
                 "cache_pct": _pct(hits, hits + misses), "failures": sum(r["failures"] for r in rows)},
        "rows": rows, "models": models,
        "history": [{"day": d, "tokens": grid[d], "total": sum(grid[d].values())} for d in day_list],
        "history_totals": totals, "history_share": share,
        "insights": _insights(features, s["models"], share),
    }
