"""
Tests for app/rate_limit.py and its middleware.

The failure this file mostly exists to prevent is not "a crawler got through".
It is the limiter mistaking every visitor for one client and locking out the
whole user base -- which is strictly worse than running with no limiter. So the
fail-open paths get more attention here than the throttling itself.
"""
import pytest
from fastapi.testclient import TestClient

from app import config, rate_limit


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    """Fresh buckets, and the limiter on -- conftest disables it suite-wide."""
    monkeypatch.setattr(config, "RATE_LIMIT_ENABLED", True)
    rate_limit.reset()
    yield
    rate_limit.reset()


class _Req:
    """Minimal stand-in for a Starlette request."""

    class _Client:
        def __init__(self, host):
            self.host = host

    def __init__(self, headers=None, host="10.0.0.1"):
        self.headers = headers or {}
        self.client = self._Client(host) if host else None


# ── Client identification ────────────────────────────────────────────────────

def test_leftmost_forwarded_address_wins():
    """XFF is client, proxy1, proxy2 -- the visitor is leftmost."""
    r = _Req({"x-forwarded-for": "203.0.113.7, 70.41.3.18, 150.172.238.178"})
    assert rate_limit.client_key(r) == "203.0.113.7"


def test_falls_back_through_real_ip_then_socket():
    assert rate_limit.client_key(_Req({"x-real-ip": "198.51.100.9"})) == "198.51.100.9"
    assert rate_limit.client_key(_Req({}, host="192.0.2.5")) == "192.0.2.5"


def test_unidentifiable_client_returns_none_rather_than_a_shared_bucket():
    """No headers and no socket peer must not collapse to a constant key."""
    assert rate_limit.client_key(_Req({}, host=None)) is None
    assert rate_limit.client_key(_Req({"x-forwarded-for": "   "}, host=None)) is None


def test_client_key_never_raises():
    class Hostile:
        @property
        def headers(self):
            raise RuntimeError("boom")

    assert rate_limit.client_key(Hostile()) is None


# ── The fail-open guarantees ─────────────────────────────────────────────────

def test_unknown_client_is_always_allowed():
    """A None key must never be throttled -- that is the lock-everyone-out bug."""
    for _ in range(500):
        allowed, _ = rate_limit.check("/search", None)
        assert allowed


def test_disabling_the_limiter_allows_everything(monkeypatch):
    monkeypatch.setattr(config, "RATE_LIMIT_ENABLED", False)
    for _ in range(500):
        assert rate_limit.check("/search", "1.2.3.4")[0]


def test_unlisted_paths_are_never_limited():
    """/healthz must stay reachable or the keepalive workflow starts failing,
    and "/" is what a platform probe would hit."""
    for path in ("/healthz/deep", "/healthz/reranker", "/", "/saved", "/static/app.js"):
        for _ in range(200):
            assert rate_limit.check(path, "1.2.3.4")[0], path


# ── Throttling ───────────────────────────────────────────────────────────────

def test_limit_trips_only_after_the_budget_is_spent():
    limit, _ = rate_limit.LIMITS["/search"]
    for i in range(limit):
        assert rate_limit.check("/search", "9.9.9.9")[0], f"blocked early at {i}"

    allowed, retry_after = rate_limit.check("/search", "9.9.9.9")
    assert not allowed
    assert 0 < retry_after <= 61


def test_clients_do_not_share_a_budget():
    limit, _ = rate_limit.LIMITS["/search"]
    for _ in range(limit):
        rate_limit.check("/search", "1.1.1.1")
    assert not rate_limit.check("/search", "1.1.1.1")[0]
    # A different visitor is unaffected.
    assert rate_limit.check("/search", "2.2.2.2")[0]


def test_paths_have_independent_budgets():
    limit, _ = rate_limit.LIMITS["/api/search/summary"]
    for _ in range(limit):
        rate_limit.check("/api/search/summary", "3.3.3.3")
    assert not rate_limit.check("/api/search/summary", "3.3.3.3")[0]
    assert rate_limit.check("/search", "3.3.3.3")[0]


def test_window_slides_so_a_client_recovers(monkeypatch):
    """Budget frees up as old hits age out, rather than needing a hard reset."""
    clock = {"t": 1000.0}
    monkeypatch.setattr(rate_limit.time, "monotonic", lambda: clock["t"])

    limit, window = rate_limit.LIMITS["/search"]
    for _ in range(limit):
        rate_limit.check("/search", "4.4.4.4")
    assert not rate_limit.check("/search", "4.4.4.4")[0]

    clock["t"] += window + 1
    assert rate_limit.check("/search", "4.4.4.4")[0]


def test_bucket_memory_is_bounded(monkeypatch):
    """Distinct keys must not accumulate forever -- 2 vCPU, 16GB, long uptime."""
    clock = {"t": 0.0}
    monkeypatch.setattr(rate_limit.time, "monotonic", lambda: clock["t"])
    for i in range(300):
        clock["t"] += 1.0
        rate_limit.check("/search", f"10.0.0.{i}")
    # Age everything out, then trigger a sweep.
    clock["t"] += 7200.0
    rate_limit.check("/search", "172.16.0.1")
    assert len(rate_limit._hits) < 50


# ── Middleware wiring ────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    import asyncio

    import app.db as db_mod

    db_path = str(tmp_path / "rl.db")
    monkeypatch.setattr(config, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    asyncio.run(db_mod.init_db())

    from app.main import app
    with TestClient(app) as c:
        yield c


def test_middleware_returns_429_with_retry_after(client, monkeypatch):
    monkeypatch.setattr(rate_limit, "LIMITS", {"/api/search/summary": (2, 60)})
    headers = {"x-forwarded-for": "203.0.113.42"}

    for _ in range(2):
        assert client.get("/api/search/summary", headers=headers).status_code == 200

    resp = client.get("/api/search/summary", headers=headers)
    assert resp.status_code == 429
    assert int(resp.headers["retry-after"]) > 0


def test_middleware_fault_does_not_break_the_request(client, monkeypatch):
    """The limiter sits in front of everything; a bug in it must not 500."""
    def boom(*a, **kw):
        raise RuntimeError("limiter exploded")

    monkeypatch.setattr(rate_limit, "check", boom)
    assert client.get("/healthz/reranker").status_code == 200


def test_healthz_client_reports_what_the_limiter_sees(client):
    """The endpoint that makes the proxy assumption verifiable after deploy."""
    resp = client.get("/healthz/client",
                      headers={"x-forwarded-for": "203.0.113.99, 10.0.0.1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["client_key"] == "203.0.113.99"
    assert body["sources"]["x_forwarded_for"] == "203.0.113.99, 10.0.0.1"
    assert body["rate_limit"]["enabled"] is True
