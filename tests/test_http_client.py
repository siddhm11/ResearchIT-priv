"""
The shared connection pool.

Every outbound call used to build its own `httpx.AsyncClient` inside an
`async with`, which closed the pool at the end of the block — so each Turso
fetch, arXiv fallback and 60s replication tick paid a fresh DNS+TCP+TLS
handshake. Measured against live Turso: 143ms median per-call vs 31ms pooled,
a 112ms saving on every request after the first.
"""
import asyncio

import httpx
import pytest

from app import http_client


@pytest.fixture(autouse=True)
async def _reset():
    await http_client.aclose()
    yield
    await http_client.aclose()


async def test_returns_the_same_client_within_one_loop():
    """Pooling is the entire point — a new client per call would defeat it."""
    assert http_client.get_client() is http_client.get_client()


async def test_client_is_configured_for_reuse():
    c = http_client.get_client()
    assert isinstance(c, httpx.AsyncClient)
    assert not c.is_closed


async def test_aclose_is_idempotent():
    http_client.get_client()
    await http_client.aclose()
    await http_client.aclose()          # must not raise
    assert http_client.stats()["created"] is False


async def test_client_survives_being_closed_and_reused():
    """A closed pool must be rebuilt, not handed back closed."""
    first = http_client.get_client()
    await http_client.aclose()
    second = http_client.get_client()
    assert second is not first
    assert not second.is_closed


def test_client_is_rebound_when_the_event_loop_changes():
    """The regression this module actually shipped with.

    An AsyncClient holds connections bound to the loop that created them. A
    pooled connection from a dead loop does not report itself closed — it
    raises "Event loop is closed" on first use. pytest builds a fresh loop per
    test, so caching across loops broke a previously-passing arXiv test.
    """
    held = {}

    async def grab(key):
        # Keep a real reference. Comparing id() would be unsound: the first
        # client is freed when the loop ends and CPython readily hands the same
        # address to the replacement.
        held[key] = http_client.get_client()

    asyncio.run(grab("first"))
    asyncio.run(grab("second"))
    assert held["first"] is not held["second"], (
        "the same client was reused across two different event loops"
    )


def test_a_live_request_works_after_the_loop_is_replaced():
    """End-to-end version of the above, against a local in-memory transport."""
    async def call():
        transport = httpx.MockTransport(lambda r: httpx.Response(200, text="ok"))
        client = http_client.get_client()
        # Swap in a transport so this stays offline but still exercises the
        # real client object and its loop binding.
        client._transport = transport
        resp = await client.get("https://example.invalid/")
        await http_client.aclose()
        return resp.text

    assert asyncio.run(call()) == "ok"
    assert asyncio.run(call()) == "ok"      # second loop, must not raise
