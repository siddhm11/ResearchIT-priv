"""
The multi-interest feed build, timed stage by stage (2026-10-23).

On the Space a first feed page for a reader with saves took 6.9 s, and the
stage timings it recorded covered 3.9 s of that. These tests run the real
Tier 1 build against fake Qdrant and Turso and check that every wait has its
own key in /healthz/runtime, and that a served page records what the reader
waited for.
"""
import re
from unittest.mock import AsyncMock

import httpx
import numpy as np
import pytest

from app import db, discovery_svc, qdrant_svc, runtime_stats, turso_svc, user_state as us
from app.main import app
from app.routers import recommendations as recs

READER = "timed-reader"
TOPICS = 3
SAVES = [f"2601.{i:05d}" for i in range(12)]          # four per topic
POOL = [f"2609.{i:05d}" for i in range(300)]
_POOL = set(POOL)


def _unit(aid: str) -> np.ndarray:
    """A fixed vector near its topic's axis, so the saves form three interests."""
    n = int(aid[-5:])
    rng = np.random.default_rng(int(aid.replace(".", "")))
    v = 0.05 * rng.standard_normal(1024).astype(np.float32)
    v[n % TOPICS] += 1.0
    return v / np.linalg.norm(v)


VECTORS = {aid: _unit(aid) for aid in SAVES + POOL}

STAGES = {"saved_vectors_ms", "clustering_ms", "cluster_search_ms", "ann_retrieval_ms",
          "cand_vectors_ms", "cand_details_ms", "candidate_meta_ms", "features_ms",
          "rerank_ms", "mmr_ms", "exploration_ms", "total_ms"}


def install(monkeypatch, *, vectors_wait: float = 0.0, details_wait: float = 0.0):
    """Fake Qdrant and Turso. The waits apply to the candidate fetches only."""
    import asyncio

    async def get_paper_vectors(ids):
        if _POOL.intersection(ids):
            await asyncio.sleep(vectors_wait)
        return {a: VECTORS[a] for a in ids if a in VECTORS}

    async def search(query_vector, limit, exclude_ids=None):
        q = np.asarray(query_vector, dtype=np.float32)
        hits = sorted(((float(VECTORS[a] @ q), a) for a in POOL if a not in (exclude_ids or ())),
                      reverse=True)[:limit]
        return [{"arxiv_id": a, "score": s} for s, a in hits]

    async def metadata(ids):
        if _POOL.intersection(ids):
            await asyncio.sleep(details_wait)
        return {a: {"arxiv_id": a, "title": f"Paper {a}", "abstract": "A study.", "authors": "[]",
                    "category": "cs.LG", "arxiv_categories": "cs.LG", "published": "2026-09-01",
                    "year": 2026} for a in ids}

    monkeypatch.setattr(qdrant_svc, "get_paper_vectors", get_paper_vectors)
    monkeypatch.setattr(qdrant_svc, "search_by_vector_with_scores", search)
    monkeypatch.setattr(turso_svc, "fetch_metadata_batch", metadata)


async def reader():
    await db.init_db()
    us._cache.clear()
    for pid in SAVES:
        await db.log_interaction(READER, pid, "save")
    return await us.ensure_loaded(READER)


async def test_every_stage_of_the_build_has_a_timing(monkeypatch):
    install(monkeypatch)
    state = await reader()
    ranked, _explore, _tags, _ms, timing = await recs._multi_interest_recommend(
        READER, state, set(), 60, query_id="q")
    assert ranked and STAGES <= timing.keys()
    # The keyed stages run one after another, so together they fit in the total.
    assert timing["total_ms"] >= timing["saved_vectors_ms"] + timing["ann_retrieval_ms"]


async def test_a_served_page_records_what_the_reader_waited_for(monkeypatch):
    install(monkeypatch)
    await reader()
    recs._FEED_CACHE.clear()
    monkeypatch.setattr(discovery_svc, "starter_papers", AsyncMock(return_value=[]))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test",
                                 cookies={"arxiv_user_id": READER}) as c:
        first = await c.get("/api/recommendations")
        qid = re.search(r"query_id=([0-9a-f-]{36})", first.text).group(1)
        await c.get(f"/api/recommendations?page=2&query_id={qid}")
    stats = runtime_stats.timings()
    assert {"page_ms", "log_ms", "cards_ms", "request_ms"} <= stats["feed_first_page"].keys()
    assert "request_ms" in stats["feed_next_page"] and "total_ms" in stats["feed_tier1"]
