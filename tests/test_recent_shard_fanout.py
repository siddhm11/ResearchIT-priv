"""Fan-out across the main collection and the recent-papers shard.

The failure modes worth testing are the quiet ones: a shard outage that empties
search results, a ContextVar leak that sends both queries to the same cluster,
and a merge that drops or duplicates papers.
"""
import asyncio
import json

import pytest

from app import config, qdrant_svc


# ── merge ────────────────────────────────────────────────────────────────────

def test_merge_orders_by_score_across_shards():
    main = [{"arxiv_id": "0704.0001", "score": 0.9},
            {"arxiv_id": "0704.0002", "score": 0.5}]
    recent = [{"arxiv_id": "2601.0001", "score": 0.7}]
    out = qdrant_svc._merge_scored([main, recent], limit=10)
    assert [r["arxiv_id"] for r in out] == ["0704.0001", "2601.0001", "0704.0002"]


def test_merge_respects_limit():
    main = [{"arxiv_id": f"a{i}", "score": 1.0 - i / 100} for i in range(50)]
    recent = [{"arxiv_id": f"b{i}", "score": 0.99 - i / 100} for i in range(50)]
    out = qdrant_svc._merge_scored([main, recent], limit=10)
    assert len(out) == 10
    assert out == sorted(out, key=lambda r: r["score"], reverse=True)


def test_merge_dedupes_keeping_higher_score():
    """A paper in both shards must appear once, at its better score."""
    out = qdrant_svc._merge_scored(
        [[{"arxiv_id": "x", "score": 0.4}], [{"arxiv_id": "x", "score": 0.8}]],
        limit=10)
    assert out == [{"arxiv_id": "x", "score": 0.8}]


def test_merge_skips_entries_without_arxiv_id():
    out = qdrant_svc._merge_scored(
        [[{"score": 0.9}, {"arxiv_id": "ok", "score": 0.1}]], limit=10)
    assert [r["arxiv_id"] for r in out] == ["ok"]


# ── fan-out behaviour ────────────────────────────────────────────────────────

@pytest.fixture
def fanout_on(monkeypatch):
    monkeypatch.setattr(config, "SEARCH_FANOUT_RECENT", True)
    monkeypatch.setattr(config, "QDRANT_RECENT_URL", "https://recent.example")
    monkeypatch.setattr(config, "QDRANT_RECENT_API_KEY", "k")
    monkeypatch.setattr(config, "QDRANT_RECENT_COLLECTION", "arxiv_recent")


def test_disabled_fanout_queries_only_main(monkeypatch):
    monkeypatch.setattr(config, "SEARCH_FANOUT_RECENT", False)
    seen = []

    async def fake(vec, limit=50):
        seen.append(qdrant_svc._BACKEND.get())
        return [{"arxiv_id": "main", "score": 0.5}]

    monkeypatch.setattr(qdrant_svc, "search_dense", fake)
    out = asyncio.run(qdrant_svc.search_dense_merged([0.1] * 4, limit=5))
    assert seen == ["a"]
    assert [r["arxiv_id"] for r in out] == ["main"]


def test_fanout_queries_both_backends(monkeypatch, fanout_on):
    """Each gather branch must see its own backend.

    A regression here is near-invisible: if the ContextVar leaked, both calls
    would hit the main cluster and search would still return plausible results,
    just without any recent papers.
    """
    seen = []

    async def fake(vec, limit=50):
        b = qdrant_svc._BACKEND.get()
        seen.append(b)
        await asyncio.sleep(0)          # force interleaving of the two tasks
        assert qdrant_svc._BACKEND.get() == b, "backend changed across await"
        return [{"arxiv_id": f"{b}-1", "score": 0.9 if b == "a" else 0.8}]

    monkeypatch.setattr(qdrant_svc, "search_dense", fake)
    out = asyncio.run(qdrant_svc.search_dense_merged([0.1] * 4, limit=5))
    assert sorted(seen) == ["a", "recent"]
    assert [r["arxiv_id"] for r in out] == ["a-1", "recent-1"]


def test_recent_shard_failure_does_not_break_search(monkeypatch, fanout_on):
    async def fake(vec, limit=50):
        if qdrant_svc._BACKEND.get() == "recent":
            raise RuntimeError("shard down")
        return [{"arxiv_id": "main", "score": 0.5}]

    monkeypatch.setattr(qdrant_svc, "search_dense", fake)
    out = asyncio.run(qdrant_svc.search_dense_merged([0.1] * 4, limit=5))
    assert [r["arxiv_id"] for r in out] == ["main"]


def test_main_failure_still_returns_recent(monkeypatch, fanout_on):
    async def fake(vec, limit=50):
        if qdrant_svc._BACKEND.get() == "a":
            raise RuntimeError("main down")
        return [{"arxiv_id": "2601.0001", "score": 0.5}]

    monkeypatch.setattr(qdrant_svc, "search_dense", fake)
    out = asyncio.run(qdrant_svc.search_dense_merged([0.1] * 4, limit=5))
    assert [r["arxiv_id"] for r in out] == ["2601.0001"]


def test_recent_backend_disables_uint8_conversion(fanout_on):
    """The shard stores float16; converting its query would wreck retrieval."""
    with qdrant_svc.use_backend("recent"):
        url, _key, coll, lo, scale = qdrant_svc._params()
        assert coll == "arxiv_recent"
        assert url == "https://recent.example"
        assert (lo, scale) == (0.0, 0.0)
        vec = [0.1, 0.2, 0.3]
        assert qdrant_svc._quantize_query(vec) == vec


def test_fanout_requires_configuration(monkeypatch):
    """Enabling the flag without credentials must not turn fan-out on."""
    monkeypatch.setattr(config, "SEARCH_FANOUT_RECENT", True)
    monkeypatch.setattr(config, "QDRANT_RECENT_URL", "")
    monkeypatch.setattr(config, "QDRANT_RECENT_API_KEY", "")
    assert config.fanout_enabled() is False


# ── author parsing ───────────────────────────────────────────────────────────

def test_authors_split_on_semicolon():
    """Stored rows join names with '; '; splitting on ',' made one blob author."""
    from app import turso_svc
    out = turso_svc._to_paper_dict(
        {"arxiv_id": "2601.0001", "title": "T",
         "authors": "Yao Xiao; Qiqian Fu; Heyi Tao", "categories": "cs.CV"})
    assert json.loads(out["authors"]) == ["Yao Xiao", "Qiqian Fu", "Heyi Tao"]


def test_authors_still_split_on_comma_when_no_semicolon():
    """arxiv_svc's fallback path supplies comma-separated names."""
    from app import turso_svc
    out = turso_svc._to_paper_dict(
        {"arxiv_id": "2601.0002", "title": "T",
         "authors": "Ada Lovelace, Alan Turing", "categories": "cs.LG"})
    assert json.loads(out["authors"]) == ["Ada Lovelace", "Alan Turing"]


def test_authors_capped_at_five():
    from app import turso_svc
    out = turso_svc._to_paper_dict(
        {"arxiv_id": "2601.0003", "title": "T",
         "authors": "; ".join(f"A{i}" for i in range(9)), "categories": "cs.LG"})
    assert len(json.loads(out["authors"])) == 5
