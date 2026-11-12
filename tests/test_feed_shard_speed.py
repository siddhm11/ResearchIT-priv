"""The feed's Qdrant searches: seen papers filtered inside Qdrant, the feed's own
oversampling, and a shard deadline (doc 06, 2026-11-13).

Measured cold against the live shards, a 3-interest reader's searches took
9.1 s and a 6-interest reader's 13.8 s, because each query asked every shard
for twice its limit and rescored 8x that many stored vectors from disk.
"""
import asyncio
import time
from types import SimpleNamespace

import pytest

from app import config, qdrant_svc, runtime_stats
from app.routers import recommendations as recs


class FakeClient:
    """Records each query_points call; returns papers p0..p{n-1}, best first."""

    def __init__(self, n=30):
        self.calls, self.n = [], n

    def query_points(self, **kw):
        self.calls.append(kw)
        pts = [SimpleNamespace(payload={"arxiv_id": f"p{i}"}, score=1 - i / 100, vector=None)
               for i in range(self.n)]
        return SimpleNamespace(points=pts[: kw["limit"]])


@pytest.fixture
def one_shard(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(qdrant_svc, "_active_backends", lambda: ["a"])
    monkeypatch.setattr(qdrant_svc, "_client", lambda: client)
    return client


def _excluded(call) -> set[str]:
    f = call["query_filter"]
    return set(f.must_not[0].match.any) if f else set()


async def test_seen_papers_are_left_out_inside_qdrant(one_shard):
    got = await qdrant_svc.search_by_vector_with_scores([0.1] * 4, limit=5, exclude_ids={"p0", "p2"})
    call = one_shard.calls[0]
    assert _excluded(call) == {"p0", "p2"}
    assert call["limit"] == 5                       # not twice the limit any more
    # A shard that ignored the filter still cannot hand back a seen paper.
    assert [g["arxiv_id"] for g in got] == ["p1", "p3", "p4"]


async def test_a_huge_seen_set_is_filtered_afterwards(one_shard, monkeypatch):
    monkeypatch.setattr(qdrant_svc, "_INLINE_EXCLUDE_MAX", 2)
    got = await qdrant_svc.search_by_vector_with_scores([0.1] * 4, limit=5, exclude_ids={"p0", "p1", "p2"})
    call = one_shard.calls[0]
    assert call["query_filter"] is None and call["limit"] == 10
    assert [g["arxiv_id"] for g in got] == ["p3", "p4", "p5", "p6", "p7"]


async def test_the_feed_has_its_own_oversampling(one_shard):
    await qdrant_svc.search_by_vector_with_scores([0.1] * 4, limit=5)
    await qdrant_svc.search_by_vector_with_scores([0.1] * 4, limit=5, oversampling=2.0)
    search, feed = (c["search_params"].quantization for c in one_shard.calls)
    assert search.oversampling == config.SEARCH_OVERSAMPLING and search.rescore
    assert feed.oversampling == 2.0 and feed.rescore


async def test_recommend_filters_seen_papers_inside_qdrant(one_shard, monkeypatch):
    async def vectors(ids, remember=True):
        return {a: [0.1] * 4 for a in ids}
    monkeypatch.setattr(qdrant_svc, "get_paper_vectors", vectors)
    got = await qdrant_svc.recommend(["s1"], [], {"p0"}, limit=3, oversampling=2.0)
    call = one_shard.calls[0]
    assert _excluded(call) == {"p0"} and call["limit"] == 3
    assert call["search_params"].quantization.oversampling == 2.0
    assert got == ["p1", "p2"]


def _shards(monkeypatch, waits: dict[str, float]):
    """Three shards; each answers [its name] after its wait."""
    monkeypatch.setattr(qdrant_svc, "_active_backends", lambda: list(waits))

    async def ask():
        name = qdrant_svc._BACKEND.get()
        await asyncio.sleep(waits[name])
        return [name]
    return ask


async def test_a_slow_shard_does_not_hold_the_feed(monkeypatch):
    ask = _shards(monkeypatch, {"a": 0.0, "b": 0.01, "recent": 2.0})
    before = runtime_stats.counters().get("shard_late_recent", 0)
    t0 = time.monotonic()
    got = await qdrant_svc._fanout(ask, "t", deadline_s=0.2)
    assert time.monotonic() - t0 < 1.0
    assert got == [["a"], ["b"], []]
    assert runtime_stats.counters()["shard_late_recent"] == before + 1


async def test_when_every_shard_is_late_the_first_answer_is_kept(monkeypatch):
    ask = _shards(monkeypatch, {"a": 0.4, "b": 0.15, "recent": 0.4})
    got = await qdrant_svc._fanout(ask, "t", deadline_s=0.05)
    assert got == [[], ["b"], []]


async def test_without_a_deadline_every_shard_is_waited_for(monkeypatch):
    ask = _shards(monkeypatch, {"a": 0.0, "b": 0.0, "recent": 0.1})
    assert await qdrant_svc._fanout(ask, "t") == [["a"], ["b"], ["recent"]]


async def test_each_shard_time_is_recorded(monkeypatch):
    ask = _shards(monkeypatch, {"a": 0.0, "b": 0.0, "recent": 0.0})
    await qdrant_svc._fanout(ask, "timed")
    assert {"a_ms", "b_ms", "recent_ms"} <= runtime_stats.timings()["shard:timed"].keys()


async def test_a_failing_shard_still_costs_only_its_share(monkeypatch):
    monkeypatch.setattr(qdrant_svc, "_active_backends", lambda: ["a", "b"])

    async def ask():
        if qdrant_svc._BACKEND.get() == "b":
            raise RuntimeError("down")
        return ["a"]
    assert await qdrant_svc._fanout(ask, "t", deadline_s=1.0) == [["a"], []]


def test_the_feed_asks_with_its_own_settings(monkeypatch):
    monkeypatch.setattr(config, "FEED_OVERSAMPLING", 3.0)
    monkeypatch.setattr(config, "FEED_SHARD_DEADLINE_S", 1.5)
    assert recs._feed_query() == {"oversampling": 3.0, "deadline_s": 1.5}
    monkeypatch.setattr(config, "FEED_SHARD_DEADLINE_S", 0.0)
    assert recs._feed_query()["deadline_s"] is None
