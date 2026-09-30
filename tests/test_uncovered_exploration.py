"""Exploration draws from ticked interests that no cluster covers."""
from unittest.mock import AsyncMock

from app import db, discovery_svc, turso_svc
from app.routers import recommendations as recs


def _setup(monkeypatch, groups, medoid_meta):
    monkeypatch.setattr(db, "get_user_category_groups", AsyncMock(return_value=groups))
    monkeypatch.setattr(turso_svc, "fetch_metadata_batch", AsyncMock(return_value=medoid_meta))
    starter = AsyncMock(return_value=[{"arxiv_id": "cv1"}, {"arxiv_id": "seen1"}, {"arxiv_id": "cv2"}])
    monkeypatch.setattr(discovery_svc, "starter_papers", starter)
    return starter


async def test_uncovered_interest_supplies_exploration(monkeypatch):
    starter = _setup(
        monkeypatch,
        {"nlp": {"cs.CL", "cs.IR"}, "cv": {"cs.CV"}, "robotics": {"cs.RO"}},
        {"m1": {"arxiv_categories": "cs.SE cs.CL"}, "m2": {"arxiv_categories": "cs.RO"}},
    )
    got = await recs._uncovered_interest_papers("u", ["m1", "m2"], exclude={"seen1"})
    assert got == ["cv1", "cv2"]
    starter.assert_awaited_once_with({"cv": {"cs.CV"}}, limit=recs._UNCOVERED_POOL)


async def test_every_interest_covered_keeps_the_default_pool(monkeypatch):
    starter = _setup(monkeypatch, {"robotics": {"cs.RO"}},
                     {"m1": {"arxiv_categories": "cs.RO cs.CV"}})
    assert await recs._uncovered_interest_papers("u", ["m1"], exclude=set()) == []
    starter.assert_not_awaited()


async def test_unknown_medoid_categories_do_not_flag_everything(monkeypatch):
    """A medoid missing from metadata would make every interest look uncovered."""
    starter = _setup(monkeypatch, {"cv": {"cs.CV"}}, {"m1": {"arxiv_categories": "cs.RO"}})
    assert await recs._uncovered_interest_papers("u", ["m1", "m2"], exclude=set()) == []
    starter.assert_not_awaited()


async def test_no_ticked_interests_means_no_change(monkeypatch):
    starter = _setup(monkeypatch, {}, {})
    assert await recs._uncovered_interest_papers("u", ["m1"], exclude=set()) == []
    starter.assert_not_awaited()
