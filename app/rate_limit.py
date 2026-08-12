"""
In-process per-client rate limiting for the expensive endpoints.

What this protects
------------------
Not CPU, primarily. The binding resource is the Groq free-tier quota: it is a
single shared bucket, /search and /api/search/summary both draw on it, and one
crawler can drain it in minutes. When it is gone, query rewriting and overviews
fail for every real user. Qdrant and Zilliz quotas have the same shape, and the
Space has 2 vCPU to absorb any of it.

So the limits below are deliberately generous -- they are sized to stop
automated traffic, not to shape human behaviour.

Why hand-rolled
---------------
CLAUDE.md rules out Redis at this scale and sanctions in-process state, and
run.py starts a single uvicorn worker with no `workers` argument, so one
process sees every request and these counters are complete rather than
approximate. Adding slowapi would also be a new dependency for ~60 lines.

Identifying the client is the risky part
----------------------------------------
A Space runs behind Hugging Face's proxy, so `request.client.host` is the
proxy, not the visitor. Keyed on that, every visitor in the world shares one
bucket and the limiter locks out the entire user base the moment it trips --
strictly worse than no limiter at all.

Two things contain that risk:

  1. Everything fails OPEN. No X-Forwarded-For, a malformed header, an
     unexpected exception -- the request is allowed. A limiter that cannot
     identify clients must do nothing, not guess.
  2. GET /healthz/client echoes exactly what the limiter derived, so the
     assumption is checkable against the deployed Space with one curl instead
     of being taken on trust.

Verify after deploying, before relying on this:

    curl https://siddhm11-researchit.hf.space/healthz/client

`client_key` should differ between two networks. If it is identical everywhere,
the proxy is not forwarding the client address; set RATE_LIMIT_ENABLED=0 until
that is understood.
"""
from __future__ import annotations

import time
from collections import deque

from app import config

# path prefix -> (max requests, window seconds)
#
# /search            BGE-M3 encode + Qdrant fanout + cross-encoder + Groq rewrite
# /api/search/summary  a Groq completion per call, the most quota-hungry path
# /api/recommendations K medoid queries + rerank; paginates, so it needs headroom
#
# Everything else is unlimited, deliberately. /healthz/* must stay reachable or
# the keepalive workflow starts failing, and "/" is cheap and is what any
# platform-level probe would hit -- throttling it risks the Space's own health.
LIMITS: dict[str, tuple[int, int]] = {
    "/api/recommendations": (60, 60),
    "/api/search/summary": (20, 60),
    "/search": (30, 60),
}

# Bound the memory. Each key holds at most `max requests` timestamps, so the
# worst case is roughly MAX_KEYS * 60 floats -- a few MB at this cap.
MAX_KEYS = 10_000

_hits: dict[tuple[str, str], deque[float]] = {}
_last_sweep = 0.0
_SWEEP_EVERY = 300.0


def client_key(request) -> str | None:
    """Best-effort stable identifier for the caller, or None if unknown.

    Returns the leftmost X-Forwarded-For entry, which is the original client
    where the header is set by a trusted proxy. It is spoofable, but this is
    abuse mitigation rather than access control -- an attacker willing to rotate
    the header is equally able to rotate source addresses.
    """
    try:
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            first = xff.split(",")[0].strip()
            if first:
                return first
        real = request.headers.get("x-real-ip", "").strip()
        if real:
            return real
        client = request.client
        return client.host if client and client.host else None
    except Exception:
        return None


def _rule_for(path: str) -> tuple[str, int, int] | None:
    for prefix, (limit, window) in LIMITS.items():
        if path == prefix or path.startswith(prefix + "/") or path.startswith(prefix + "?"):
            return prefix, limit, window
    return None


def _sweep(now: float) -> None:
    """Drop buckets nothing has touched recently, and hard-cap the dict."""
    global _last_sweep
    if now - _last_sweep < _SWEEP_EVERY:
        return
    _last_sweep = now
    stale = [k for k, dq in _hits.items() if not dq or now - dq[-1] > 3600]
    for k in stale:
        _hits.pop(k, None)
    if len(_hits) > MAX_KEYS:
        # Evict least-recently-active first.
        for k in sorted(_hits, key=lambda k: _hits[k][-1] if _hits[k] else 0.0)[
                :len(_hits) - MAX_KEYS]:
            _hits.pop(k, None)


def check(path: str, key: str | None) -> tuple[bool, int]:
    """Return (allowed, retry_after_seconds).

    Never raises: an unknown key or an unlimited path is always allowed.
    """
    if not config.RATE_LIMIT_ENABLED or key is None:
        return True, 0
    rule = _rule_for(path)
    if rule is None:
        return True, 0
    prefix, limit, window = rule

    now = time.monotonic()
    _sweep(now)

    dq = _hits.setdefault((prefix, key), deque())
    cutoff = now - window
    while dq and dq[0] < cutoff:
        dq.popleft()

    if len(dq) >= limit:
        # Oldest hit in the window governs when a slot frees up.
        return False, max(1, int(window - (now - dq[0])) + 1)

    dq.append(now)
    return True, 0


def stats() -> dict:
    """Snapshot for the health endpoint."""
    return {
        "enabled": config.RATE_LIMIT_ENABLED,
        "limits": {p: {"requests": l, "per_seconds": w} for p, (l, w) in LIMITS.items()},
        "tracked_keys": len(_hits),
    }


def reset() -> None:
    """Clear all buckets. Tests only."""
    _hits.clear()
    global _last_sweep
    _last_sweep = 0.0
