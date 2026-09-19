"""A reader who skips onboarding must still get a feed.

Tier 0 was gated on the reader having picked categories, with no else branch,
so skipping onboarding fell through the whole cascade to "Nothing here yet" --
permanently, unless they independently found search and saved something. 68 of
the first 120 users saved exactly once, so the empty-state path is not a corner
case, and CLAUDE.md §3.6 recorded this fallback as already done.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app import db, turso_svc
from app.config import DEFAULT_TRENDING_CATEGORIES
from app.routers import recommendations as R


class _NoSaves:
    positive_list: list[str] = []
    negative_list: list[str] = []

    def has_enough_for_recs(self) -> bool:
        return False


@pytest.fixture
def feed_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "cold.db"))
    return db.DB_PATH


def _trending_spy(n=12):
    seen: dict = {}

    async def fake(categories, limit=10):
        seen["categories"] = set(categories)
        seen["limit"] = limit
        return [{"arxiv_id": f"2401.{i:05d}"} for i in range(n)]

    return fake, seen


async def _build(user_id="u-skip", categories=None):
    await db.init_db()
    if categories:
        await db.save_onboarding_categories(user_id, list(categories))
    fake, seen = _trending_spy()
    with patch.object(turso_svc, "fetch_trending_by_categories", side_effect=fake):
        entry = await R._build_feed(user_id, _NoSaves(), "q-cold")
    return entry, seen


async def test_skipping_onboarding_still_produces_a_feed(feed_db):
    """The regression. This used to return None."""
    entry, _ = await _build()

    assert entry is not None, "a reader who skipped onboarding got nothing"
    assert entry["ranked"], "the fallback produced an empty ranking"
    assert entry["trending"] is True


async def test_the_default_feed_spans_more_than_machine_learning(feed_db):
    """Someone who told us nothing should not be shown a CS-only site."""
    _, seen = await _build()

    assert seen["categories"] == set(DEFAULT_TRENDING_CATEGORIES)
    prefixes = {c.split(".")[0] for c in seen["categories"]}
    assert len(prefixes) >= 4, f"too narrow a default: {prefixes}"


async def test_the_two_cold_start_feeds_are_distinguishable_in_the_log(feed_db):
    """"Your categories" and "we had nothing to go on" are different feeds and
    must not share a source tag, or the exposure log cannot tell them apart."""
    skipped, _ = await _build(user_id="u-skip")
    picked, _ = await _build(user_id="u-picked", categories=["ml"])

    def source_of(entry):
        return next(iter(entry["tags"].values()))["candidate_source"]

    assert source_of(skipped) == "trending_default_fallback"
    assert source_of(picked) == "trending_category_fallback"


async def test_a_reader_who_picked_categories_still_gets_only_those(feed_db):
    """The fallback must not widen a deliberate choice."""
    _, seen = await _build(user_id="u-picked", categories=["ml"])

    assert seen["categories"] == {"cs.LG", "stat.ML"}   # the "ml" group


async def test_both_cold_start_feeds_report_tier_zero(feed_db):
    """_serving_tier drives the progress UI and the exposure log's tier column;
    a source tag it does not know silently reports Tier 1."""
    skipped, _ = await _build(user_id="u-skip")
    picked, _ = await _build(user_id="u-picked", categories=["ml"])

    assert R._serving_tier(skipped) == 0
    assert R._serving_tier(picked) == 0


async def test_the_fallback_still_carries_real_propensities(feed_db):
    """Tier 0 fills slots epsilon-greedily, and §3.4b forbids logging a
    degenerate 1.0 on a tier that has randomness."""
    entry, _ = await _build()

    props = [t["propensity"] for t in entry["tags"].values()]
    assert props and all(0.0 < p <= 1.0 for p in props)
    assert any(p < 1.0 for p in props), "no exploration propensity recorded"
