"""
Tests for cold-start feed churn (Tier 0).

The behaviour under test, measured on the deployed pipeline before the fix:
a user with zero interactions who refreshed the feed was served byte-identical
papers in identical order, indefinitely, each logged with propensity=1.0.

Two causes, both covered here:
  * `seen` only tracks saves and dismissals, so a reader who refreshes without
    clicking anything is remembered as having done nothing. Papers that were
    merely SHOWN now go to db.feed_impressions.
  * Tier 0 sorted deterministically with exploration disabled. Slots are now
    filled epsilon-greedily, which also makes the logged propensity real.
"""
import pytest

import app.db as db_mod
from app.routers import recommendations as recs


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    import asyncio

    import app.config as cfg

    path = str(tmp_path / "churn.db")
    monkeypatch.setattr(cfg, "DB_PATH", path)
    monkeypatch.setattr(db_mod, "DB_PATH", path)
    asyncio.run(db_mod.init_db())
    return path


def _pool(n: int) -> list[str]:
    return [f"25{i:02d}.{i:05d}" for i in range(n)]


# ── Impression storage ───────────────────────────────────────────────────────

async def test_impressions_are_per_user_and_deduped(db_path):
    await db_mod.record_impressions("u1", ["a", "b"])
    await db_mod.record_impressions("u1", ["b", "c"])   # b re-shown
    await db_mod.record_impressions("u2", ["z"])

    assert await db_mod.get_impressed_ids("u1") == {"a", "b", "c"}
    assert await db_mod.get_impressed_ids("u2") == {"z"}
    assert await db_mod.get_impressed_ids("nobody") == set()


async def test_recording_nothing_is_harmless(db_path):
    assert await db_mod.record_impressions("u1", []) == 0
    assert await db_mod.record_impressions("", ["a"]) == 0
    assert await db_mod.get_impressed_ids("u1") == set()


async def test_forget_oldest_keeps_the_most_recent(db_path):
    for pid in ["a", "b", "c", "d", "e"]:
        await db_mod.record_impressions("u1", [pid])

    await db_mod.forget_oldest_impressions("u1", keep=2)
    assert len(await db_mod.get_impressed_ids("u1")) == 2


# ── Ordering policy ──────────────────────────────────────────────────────────

async def test_refreshing_advances_instead_of_repeating(db_path):
    """The headline behaviour: successive refreshes serve different papers."""
    pool = _pool(40)
    seen_across_refreshes = []

    for _ in range(3):
        ordered, _ = await recs._cold_start_order("u1", pool)
        page = ordered[:recs._PAGE_SIZE]
        seen_across_refreshes.append(page)
        await db_mod.record_impressions("u1", page)

    first, second, third = seen_across_refreshes
    assert first != second and second != third
    # With a pool this deep, pages should not overlap at all yet.
    assert not (set(first) & set(second))
    assert not (set(first) & set(third))


async def test_shown_papers_are_excluded_while_the_pool_lasts(db_path):
    pool = _pool(40)
    await db_mod.record_impressions("u1", pool[:20])

    ordered, _ = await recs._cold_start_order("u1", pool)
    assert not (set(ordered) & set(pool[:20]))


async def test_two_users_with_the_same_pool_get_different_feeds(db_path):
    """Cold start must not hand every user in a category the same ten papers."""
    pool = _pool(60)
    orders = []
    for u in [f"user{i}" for i in range(8)]:
        ordered, _ = await recs._cold_start_order(u, pool)
        orders.append(tuple(ordered[:recs._PAGE_SIZE]))

    assert len(set(orders)) > 1, "every user received an identical feed"


async def test_exhausted_pool_recycles_rather_than_going_empty(db_path):
    """An empty feed is a worse failure than a repeat."""
    pool = _pool(12)
    await db_mod.record_impressions("u1", pool)   # everything already shown

    ordered, props = await recs._cold_start_order("u1", pool)
    assert len(ordered) >= recs._PAGE_SIZE
    assert set(ordered) <= set(pool)
    assert props


async def test_ordering_is_a_permutation_with_no_duplicates(db_path):
    pool = _pool(25)
    ordered, props = await recs._cold_start_order("u1", pool)
    assert len(ordered) == len(set(ordered))
    assert set(ordered) <= set(pool)
    assert set(props) == set(ordered)


async def test_empty_pool_is_handled(db_path):
    assert await recs._cold_start_order("u1", []) == ([], {})


# ── Propensity ───────────────────────────────────────────────────────────────

async def test_propensities_are_no_longer_degenerate(db_path):
    """propensity=1.0 everywhere makes IPS/SNIPS impossible later (§3.11)."""
    pool = _pool(30)
    _, props = await recs._cold_start_order("u1", pool)

    assert props, "no propensities recorded"
    assert all(0.0 < p <= 1.0 for p in props.values())
    assert not all(p == 1.0 for p in props.values()), "policy is still deterministic"


async def test_greedy_and_explore_slots_get_distinguishable_propensities(db_path):
    """A greedy pick carries the (1-eps) mass; an explore pick only eps/n."""
    pool = _pool(50)
    _, props = await recs._cold_start_order("u1", pool)
    values = sorted(props.values())
    assert values[0] < values[-1], "all slots scored identically"
    # Nothing can exceed the greedy share plus one uniform draw.
    assert values[-1] <= (1.0 - recs._COLD_START_EPSILON) + recs._COLD_START_EPSILON + 1e-9


async def test_a_database_failure_still_returns_a_feed(db_path, monkeypatch):
    """Churn is a nicety; serving papers is not. A DB fault must not empty the feed."""
    async def boom(*a, **kw):
        raise RuntimeError("db down")

    monkeypatch.setattr(db_mod, "get_impressed_ids", boom)
    ordered, props = await recs._cold_start_order("u1", _pool(20))
    assert len(ordered) == 20
    assert props


# ── Pool sizing ──────────────────────────────────────────────────────────────

def test_pool_deepens_only_when_the_sidecar_can_afford_it(monkeypatch):
    """Depth buys refresh runway, but the Turso fallback times out on deep limits."""
    from app import local_meta

    monkeypatch.setattr(local_meta, "is_available", lambda: True)
    assert recs._trending_pool_size() == recs._TRENDING_POOL_SIDECAR

    monkeypatch.setattr(local_meta, "is_available", lambda: False)
    assert recs._trending_pool_size() == recs._TRENDING_POOL
