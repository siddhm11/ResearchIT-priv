"""Starter lists from the sidecar are computed once, not once an hour.

The live Space spent 5.8 s (p50, 7 Nov) of a 7.3 s first feed in
"exploration": starter_papers for the reader's uncovered interests, one
sidecar trending query per arXiv code. Locally cs.CV alone took 0.77 s, and
the Space's CPU is about 4.5x slower. The sidecar only changes on a deploy,
so its results stay valid for the life of the process, but they sat in a
1-hour cache that, at about one reader a day, was almost always cold.
"""
import time

import pytest

from app import discovery_svc, local_meta, turso_svc
from app.config import CATEGORY_GROUPS, DEFAULT_TRENDING_CATEGORIES, expand_category_groups


@pytest.fixture(autouse=True)
def _sidecar(monkeypatch):
    turso_svc._TRENDING_CACHE.clear()
    calls = []

    def trending(codes, limit, months):
        calls.append(("trending", frozenset(codes), limit))
        return [{"arxiv_id": f"2501.{i:05d}", "title": "t", "categories": " ".join(codes)}
                for i in range(limit)]

    def fresh(codes, limit):
        calls.append(("fresh", frozenset(codes), limit))
        return [{"arxiv_id": f"2510.{i:05d}", "title": "f", "categories": " ".join(codes)}
                for i in range(limit)]

    monkeypatch.setattr(local_meta, "is_available", lambda: True)
    monkeypatch.setattr(local_meta, "fetch_trending", trending)
    monkeypatch.setattr(local_meta, "fetch_fresh", fresh)
    yield calls
    turso_svc._TRENDING_CACHE.clear()


async def test_a_sidecar_starter_list_outlives_the_hourly_ttl(_sidecar, monkeypatch):
    await turso_svc.fetch_trending_by_categories({"cs.CV"}, limit=30)
    later = time.time() + 2 * turso_svc._TRENDING_TTL_SECONDS
    monkeypatch.setattr(turso_svc.time, "time", lambda: later)

    await turso_svc.fetch_trending_by_categories({"cs.CV"}, limit=30)
    await turso_svc.fetch_fresh_by_categories({"cs.CV"}, limit=30)
    await turso_svc.fetch_fresh_by_categories({"cs.CV"}, limit=30)

    assert [c[0] for c in _sidecar] == ["trending", "fresh"]


async def test_turso_starter_lists_still_expire():
    turso_svc._starter_put(("x",), [{"arxiv_id": "1"}])
    stamp, papers, lasting = turso_svc._TRENDING_CACHE[("x",)]
    turso_svc._TRENDING_CACHE[("x",)] = (stamp - 2 * turso_svc._TRENDING_TTL_SECONDS,
                                         papers, lasting)
    assert turso_svc._starter_get(("x",)) is None


async def test_prewarm_fills_exactly_what_the_feed_asks_for(_sidecar):
    await discovery_svc.prewarm(limits=(12, 30))
    _sidecar.clear()

    # An uncovered interest at the exploration pool size, onboarding's
    # suggestions, and a reader who skipped onboarding: all served from cache.
    await discovery_svc.starter_papers({"cv": expand_category_groups(["cv"])}, limit=30)
    await discovery_svc.starter_papers(
        {k: expand_category_groups([k]) for k in list(CATEGORY_GROUPS)[:3]}, limit=12)
    await discovery_svc.starter_papers(set(DEFAULT_TRENDING_CATEGORIES), limit=30)

    assert _sidecar == []


async def test_prewarm_does_nothing_without_the_sidecar(_sidecar, monkeypatch):
    monkeypatch.setattr(local_meta, "is_available", lambda: False)
    await discovery_svc.prewarm(limits=(12,))
    assert _sidecar == []


def test_prewarm_limits_are_the_ones_the_callers_use():
    """The cache key includes the limit, so a limit nobody asks for warms nothing."""
    import inspect
    from app.routers import onboarding, recommendations as R
    assert R._UNCOVERED_POOL in discovery_svc.PREWARM_LIMITS
    assert R._TRENDING_POOL in discovery_svc.PREWARM_LIMITS
    assert R._TRENDING_POOL_SIDECAR in discovery_svc.PREWARM_LIMITS
    assert "starter_papers(categories, limit=12)" in inspect.getsource(onboarding)
    assert 12 in discovery_svc.PREWARM_LIMITS
