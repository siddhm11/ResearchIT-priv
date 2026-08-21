"""
One pooled httpx.AsyncClient for every outbound call in the app.

Why this exists
---------------
Every outbound call used to build its own client:

    async with httpx.AsyncClient(timeout=10) as client:
        ...

`async with` closes the client at the end of the block, which tears down the
connection pool with it. So each Turso metadata fetch, arXiv fallback,
replication tick and deep health check paid a fresh DNS lookup, TCP handshake
and TLS negotiation — measured at a 308ms median against Turso versus 73ms once
the connection is reused. On the Tier-1 feed path that handshake is paid again
on every request, and it is pure overhead: the destination, the credentials and
the TLS parameters are identical every time.

A module-level client keeps the pool alive across requests, so only the first
call to a host pays setup. It is created lazily on first use — an
`httpx.AsyncClient` binds to the running event loop, so constructing it at
import time would attach it to the wrong loop (or none) under pytest and under
uvicorn's reloader.

Per-call timeouts still work: pass `timeout=` to the individual request, which
overrides the client default for that call only. That matters because the
sensible timeout here ranges from 10s for a metadata read to 30s for the cold
trending scan.
"""
from __future__ import annotations

import asyncio

import httpx

# Sized for a 2-vCPU box talking to a handful of hosts (Turso, arXiv, Qdrant's
# REST fallback). Keepalive slots are what actually buy the saving, so they are
# generous relative to the connection ceiling.
_LIMITS = httpx.Limits(max_keepalive_connections=20, max_connections=40)

# Default only. Callers with a different budget pass timeout= per request.
_DEFAULT_TIMEOUT = 20.0

_client: httpx.AsyncClient | None = None
_loop: asyncio.AbstractEventLoop | None = None


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:          # called outside async context
        return None


def get_client() -> httpx.AsyncClient:
    """The shared client, created on first use.

    Never use this as an async context manager — `async with` would close the
    shared pool for everyone. Just await requests on it directly.

    The client is rebuilt whenever the running event loop changes. An
    `httpx.AsyncClient` holds connections bound to the loop that created them,
    and a pooled connection from a dead loop raises "Event loop is closed" on
    first use rather than failing at creation. That happens routinely under
    pytest, which builds a fresh loop per test, and in any process that runs
    `asyncio.run` more than once. Rebinding is cheap and only costs one
    handshake per loop, so it is strictly better than the alternative of not
    pooling at all.
    """
    global _client, _loop
    loop = _running_loop()
    if _client is None or _client.is_closed or _loop is not loop:
        _client = httpx.AsyncClient(
            timeout=_DEFAULT_TIMEOUT,
            limits=_LIMITS,
            follow_redirects=True,
        )
        _loop = loop
    return _client


async def aclose() -> None:
    """Close the pool. Called from the FastAPI lifespan on shutdown."""
    global _client, _loop
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None
    _loop = None


def stats() -> dict:
    """For diagnostics parity with the other in-process caches."""
    return {
        "created": _client is not None,
        "closed": _client.is_closed if _client is not None else None,
        "max_connections": _LIMITS.max_connections,
        "max_keepalive": _LIMITS.max_keepalive_connections,
    }
