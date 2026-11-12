"""What a reader does, beyond saving, shapes their interests (2026-11-08).

The owner's own feed showed the gap: 7 saves, all on attention, became 2
interests, both attention, while 38 opened papers spanned LLM reasoning,
vision, robotics and economics, and the feed learned nothing from them.
Interests are now built from saves (1.0), questions asked on a paper (0.6)
and papers opened (0.3). Pass now means "not this paper", never "hide this
field": three Passes used to remove a reader's main category from the feed.
"""
import time
from unittest.mock import AsyncMock

import numpy as np
import pytest

from app import db, qdrant_svc, turso_svc, user_state as us
from app.recommend.clustering import _recency_weights, compute_clusters, per_save_clusters
from app.routers import recommendations as recs

READER = "taste-reader"
SAVES = [f"2601.{i:05d}" for i in range(0, 12, 2)]       # six, all topic 0
OPENS = [f"2603.{i:05d}" for i in range(1, 9, 2)]        # four, all topic 1
POOL = [f"2609.{i:05d}" for i in range(300)]
_POOL = set(POOL)


def _topic(aid: str) -> int:
    return int(aid[-5:]) % 2


def _unit(aid: str) -> np.ndarray:
    rng = np.random.default_rng(int(aid.replace(".", "")))
    v = 0.05 * rng.standard_normal(1024).astype(np.float32)
    v[_topic(aid)] += 1.0
    return v / np.linalg.norm(v)


VECTORS = {aid: _unit(aid) for aid in SAVES + OPENS + POOL + [f"2602.{i:05d}" for i in (1, 3, 5)]}
CATEGORY = {0: "cs.LG", 1: "cs.CV"}


def _meta(aid):
    cat = CATEGORY[_topic(aid)]
    return {"arxiv_id": aid, "title": f"Paper {aid}", "abstract": "A study.", "authors": "[]",
            "category": cat, "arxiv_categories": cat, "published": "2026-09-01", "year": 2026}


@pytest.fixture
def world(monkeypatch):
    async def get_paper_vectors(ids):
        return {a: VECTORS[a] for a in ids if a in VECTORS}

    async def search(query_vector, limit, exclude_ids=None, **_):
        q = np.asarray(query_vector, dtype=np.float32)
        pool = [a for a in POOL + OPENS if a not in (exclude_ids or ())]
        hits = sorted(((float(VECTORS[a] @ q), a) for a in pool), reverse=True)[:limit]
        return [{"arxiv_id": a, "score": s} for s, a in hits]

    async def metadata(ids):
        return {a: _meta(a) for a in ids}

    monkeypatch.setattr(qdrant_svc, "get_paper_vectors", get_paper_vectors)
    monkeypatch.setattr(qdrant_svc, "search_by_vector_with_scores", search)
    monkeypatch.setattr(turso_svc, "fetch_metadata_batch", metadata)
    monkeypatch.setattr(recs.discovery_svc, "starter_papers", AsyncMock(return_value=[]))


async def _reader(opens=(), passes=(), saves=SAVES):
    await db.init_db()
    us._cache.clear()
    for pid in saves:
        await db.log_interaction(READER, pid, "save")
    for pid in opens:
        await db.log_interaction(READER, pid, "click", source="recommendation")
    for pid in passes:
        await db.log_interaction(READER, pid, "not_interested")
    await db.cache_turso_metadata_batch([_meta(p) for p in passes])
    return await us.ensure_loaded(READER)


async def _feed(state):
    ranked, *_ = await recs._multi_interest_recommend(READER, state, set(), 60, query_id="q")
    return ranked


# ── The feed ─────────────────────────────────────────────────────────────────

def _topics_of_interests(clusters) -> set[int]:
    return {_topic(c["medoid_paper_id"]) for c in clusters}


async def test_papers_a_reader_opens_become_an_interest(world):
    ranked = await _feed(await _reader(opens=OPENS))
    assert _topics_of_interests(await db.get_user_clusters(READER)) == {0, 1}, \
        "the opened topic should be an interest of its own"
    assert sum(_topic(a) == 1 for a in ranked) >= 10, "the opened topic should be on the page"


async def test_without_opens_the_feed_stays_on_the_saved_topic(world):
    ranked = await _feed(await _reader())
    assert _topics_of_interests(await db.get_user_clusters(READER)) == {0}
    assert sum(_topic(a) == 1 for a in ranked) <= 3


async def test_an_opened_paper_is_not_served_back(world):
    ranked = await _feed(await _reader(opens=OPENS))
    assert not set(OPENS) & set(ranked)


async def test_saves_outweigh_opens(world):
    await _feed(await _reader(opens=OPENS))
    by_topic = {0: 0.0, 1: 0.0}
    for c in await db.get_user_clusters(READER):
        by_topic[_topic(c["medoid_paper_id"])] += c["importance"]
    assert by_topic[0] > by_topic[1] > 0


async def test_three_passes_do_not_remove_the_readers_own_field(world):
    """Three Passes on cs.LG cards used to suppress cs.LG from the multi-interest
    feed for 14 days, though every one of this reader's saves is cs.LG."""
    passes = [a for a in POOL if _topic(a) == 0][:3]
    second_field = [f"2602.{i:05d}" for i in (1, 3, 5)]        # a smaller cs.CV interest
    ranked = await _feed(await _reader(passes=passes, saves=SAVES + second_field))
    assert sum(_topic(a) == 0 for a in ranked) >= 30


# ── The history the clusterer reads ──────────────────────────────────────────

async def test_taste_history_weights_and_exclusions():
    await db.init_db()
    await db.log_interaction("h", "saved", "save")
    await db.log_interaction("h", "saved", "click")
    await db.log_interaction("h", "asked", "click")
    await db.log_interaction("h", "asked", "ask")
    await db.log_interaction("h", "opened", "click")
    await db.log_interaction("h", "visited", "view")
    await db.log_interaction("h", "passed", "click")
    await db.log_interaction("h", "passed", "not_interested")
    await db.log_interaction("h", "unsaved", "save")
    await db.log_interaction("h", "unsaved", "unsave")
    await db.log_interaction("h", "old", "click")
    import aiosqlite
    async with aiosqlite.connect(db.DB_PATH) as conn:
        await conn.execute("UPDATE interactions SET timestamp = datetime('now', '-40 days') "
                           "WHERE paper_id = 'old'")
        await conn.commit()

    weights = {r["paper_id"]: r["weight"] for r in await db.get_taste_history("h")}
    assert weights == {"saved": 1.0, "asked": 0.6, "opened": 0.3, "visited": 0.3}


async def test_taste_history_is_newest_first():
    await db.init_db()
    await db.log_interaction("n", "first", "save")
    time.sleep(1.1)
    await db.log_interaction("n", "second", "click")
    assert [r["paper_id"] for r in await db.get_taste_history("n")] == ["second", "first"]


# ── Importance ───────────────────────────────────────────────────────────────

def test_importance_is_the_signal_weighted_recency_sum():
    ids = [f"p{i}" for i in range(6)]
    embs = np.stack([_unit(f"2601.{i:05d}") for i in range(0, 12, 2)])
    weights = [1.0, 0.3, 0.3, 1.0, 0.6, 0.3]
    expected = float((_recency_weights(6, None) * np.array(weights)).sum())
    clusters = compute_clusters(ids, embs, None, weights)
    assert sum(c.importance for c in clusters) == pytest.approx(expected)
    single = per_save_clusters(ids[:2], embs[:2], None, weights[:2])
    assert sorted(c.importance for c in single) == pytest.approx(
        sorted(_recency_weights(2, None) * np.array(weights[:2])))
