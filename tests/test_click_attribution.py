"""A click has to record WHERE it happened and WHICH interest it came from.

Both were wrong. The click logger passed the candidate source into the `source`
column, so a column documented as search|recommendation|saved filled up with
cluster names, and a click on a search result -- which carries a query_id but
no candidate source -- was filed as a recommendation. Meanwhile `cluster_id`
was hard-coded None while the value it wanted was spelled out in the string
next to it, which left per-cluster engagement unmeasurable.
"""
from __future__ import annotations

from unittest.mock import patch

import aiosqlite
import pytest
from fastapi.testclient import TestClient

from app import db, turso_svc, qdrant_svc
from app.main import app

PAPER = {
    "arxiv_id": "1706.03762",
    "title": "Attention Is All You Need",
    "abstract": "The dominant sequence transduction models are based on "
                "complex recurrent or convolutional neural networks. " * 2,
    "authors": '["A. Vaswani"]',
    "category": "cs.CL",
    "published": "2017-06-12",
    "year": 2017,
    "citation_count": 100000,
}


@pytest.fixture
def click_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "clicks.db"))
    return db.DB_PATH


def _visit(query: str):
    async def fake_meta(ids):
        return {i: PAPER for i in ids if i == PAPER["arxiv_id"]}

    async def fake_search(**kwargs):
        return []

    with patch.object(turso_svc, "fetch_metadata_batch", side_effect=fake_meta), \
         patch.object(qdrant_svc, "search_by_vector", side_effect=fake_search), \
         patch.object(qdrant_svc, "get_paper_vectors", return_value={}):
        with TestClient(app) as c:
            return c.get(f"/p/{PAPER['arxiv_id']}?{query}")


async def _clicks(path):
    async with aiosqlite.connect(path) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(
            "SELECT * FROM interactions WHERE event_type = 'click'")
        return [dict(r) for r in await cur.fetchall()]


async def test_a_click_from_the_feed_is_recorded_as_a_recommendation(click_db):
    r = _visit("qid=q1&sf=recommendation&src=cluster_1&pos=3&prop=0.25&pol=v9.1")
    assert r.status_code == 200

    rows = await _clicks(click_db)
    assert len(rows) == 1
    assert rows[0]["source"] == "recommendation"      # was "cluster_1"
    assert rows[0]["candidate_source"] == "cluster_1"  # the origin keeps its own column


async def test_a_click_from_search_is_not_recorded_as_a_recommendation(click_db):
    """Search sets a query_id but no candidate source, so the old
    `src or "recommendation"` mislabelled every one of these."""
    r = _visit("qid=q2&sf=search&pos=0")
    assert r.status_code == 200

    rows = await _clicks(click_db)
    assert len(rows) == 1
    assert rows[0]["source"] == "search"


async def test_the_cluster_lands_in_the_cluster_column(click_db):
    _visit("qid=q3&sf=recommendation&src=cluster_2&pos=1&prop=1.0&pol=v9.1")

    rows = await _clicks(click_db)
    assert rows[0]["cluster_id"] == 2


async def test_an_exploration_pick_has_no_cluster(click_db):
    """It is served as a step OUTSIDE the reader's interests; filing it under
    cluster 0 would credit an interest that had nothing to do with it."""
    _visit("qid=q4&sf=recommendation&src=exploration&pos=10&prop=0.05&pol=v9.1")

    rows = await _clicks(click_db)
    assert rows[0]["cluster_id"] is None
    assert rows[0]["candidate_source"] == "exploration"


async def test_a_bare_visit_still_logs_nothing(click_db):
    """A shared link or a bookmark has no propensity and no policy. Recording
    it as though it did corrupts the §3.11 contract rather than honouring it."""
    r = _visit("")
    assert r.status_code == 200
    assert await _clicks(click_db) == []


async def test_the_surface_falls_back_rather_than_guessing_a_cluster(click_db):
    """Older links in the wild carry qid but no sf. They must not resurrect
    the bug by putting src back into source."""
    _visit("qid=q5&src=cluster_3&pos=2")

    rows = await _clicks(click_db)
    assert rows[0]["source"] == "recommendation"
    assert rows[0]["cluster_id"] == 3
