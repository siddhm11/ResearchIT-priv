"""
An outage must not be reported as an empty library.

Every Qdrant helper swallows failures into empty lists — `_fanout` appends `[]`
for each dead shard rather than raising — so an all-shard outage walks the whole
tier cascade returning nothing and arrives at the same place as a brand-new
user. Verified before the fix: a user with 20 saves was served "Nothing here yet
— Save 1 paper and this feed starts building itself", which reads as though
their library had been lost.

It has not been. Saves live in SQLite and replicate to Turso, entirely
independent of the vector store.
"""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import qdrant_svc
from app.main import app


def _dead_qdrant():
    """Patch every retrieval entry point to return what a dead shard returns."""
    return (
        patch.object(qdrant_svc, "get_paper_vectors", return_value={}),
        patch.object(qdrant_svc, "search_by_vector", return_value=[]),
        patch.object(qdrant_svc, "search_by_vector_with_scores", return_value=[]),
        patch.object(qdrant_svc, "recommend", return_value=[]),
    )


def test_user_with_saves_is_told_it_is_an_outage():
    with TestClient(app) as c:
        c.cookies.set("researchit_uid", "outage-user-with-library")
        # Build a library first.
        for i in range(6):
            c.post(f"/api/papers/2401.{i:05d}/save",
                   data={"source": "search", "position": 0})

        patches = _dead_qdrant()
        for p in patches:
            p.start()
        try:
            r = c.get("/api/recommendations")
        finally:
            for p in patches:
                p.stop()

    assert "Save 1 paper" not in r.text, (
        "a user with a library was told to start one")
    assert "Nothing here yet" not in r.text
    assert "library is safe" in r.text, "no reassurance that nothing was lost"
    assert r.status_code == 503, (
        "an outage should not be reported as a successful empty feed")


def test_genuinely_new_user_still_gets_the_onboarding_prompt():
    """The fix must not swallow the real cold-start case."""
    with TestClient(app) as c:
        c.cookies.set("researchit_uid", "outage-user-no-library")
        patches = _dead_qdrant()
        for p in patches:
            p.start()
        try:
            r = c.get("/api/recommendations")
        finally:
            for p in patches:
                p.stop()

    assert "library is safe" not in r.text, (
        "a user with no saves was shown an outage message")
    assert r.status_code == 200
