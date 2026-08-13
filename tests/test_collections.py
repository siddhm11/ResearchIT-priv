"""
Tests for curated reading collections.

No network: the vector store and metadata layer are stubbed, so what is under
test is the loading, the seeding contract and the honesty of the follow
response -- not Qdrant.
"""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.db as db_mod
from app import collections_svc


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A collections directory with two known collections."""
    d = tmp_path / "collections"
    d.mkdir()
    (d / "alpha.json").write_text(json.dumps({
        "slug": "alpha", "title": "Alpha Track", "blurb": "First.",
        "anchors": [{"id": f"2401.{i:05d}", "note": f"note {i}"} for i in range(6)],
    }))
    (d / "beta.json").write_text(json.dumps({
        "slug": "beta", "title": "Beta Track", "blurb": "Second.",
        "anchors": [{"id": "1706.03762", "note": "the transformer"}],
    }))
    monkeypatch.setattr(collections_svc, "COLLECTIONS_DIR", str(d))
    collections_svc.load_all.cache_clear()
    yield d
    collections_svc.load_all.cache_clear()


# ── Loading ──────────────────────────────────────────────────────────────────

def test_loads_and_sorts_by_title(repo):
    cols = collections_svc.load_all()
    assert [c["slug"] for c in cols] == ["alpha", "beta"]
    assert cols[0]["count"] == 6


def test_lookup_by_slug(repo):
    assert collections_svc.get("alpha")["title"] == "Alpha Track"
    assert collections_svc.get("nope") is None
    assert collections_svc.anchor_ids("beta") == ["1706.03762"]


def test_a_malformed_file_does_not_break_the_index(repo):
    """One bad edit must not take the page down for every other collection."""
    (repo / "broken.json").write_text("{ this is not json")
    (repo / "empty.json").write_text(json.dumps({"slug": "empty", "anchors": []}))
    collections_svc.load_all.cache_clear()

    slugs = [c["slug"] for c in collections_svc.load_all()]
    assert slugs == ["alpha", "beta"]


def test_missing_directory_is_survivable(monkeypatch, tmp_path):
    monkeypatch.setattr(collections_svc, "COLLECTIONS_DIR", str(tmp_path / "nope"))
    collections_svc.load_all.cache_clear()
    assert collections_svc.load_all() == []
    collections_svc.load_all.cache_clear()


# ── The shipped content ──────────────────────────────────────────────────────

def test_shipped_collections_are_wellformed():
    """Guards the real data/collections/*.json, not a fixture."""
    collections_svc.load_all.cache_clear()
    cols = collections_svc.load_all()
    assert len(cols) >= 8, "expected the eight curated tracks"

    for c in cols:
        # Semantic Scholar's stated cold-start budget is 5 positives; every
        # track must clear it in a single click, and clear Tier 1's >=5 too.
        assert c["count"] >= 6, f"{c['slug']} has too few anchors to seed a profile"
        assert c["blurb"], f"{c['slug']} has no blurb"
        for a in c["anchors"]:
            assert a["note"], f"{c['slug']}/{a['id']} has no curator note"
            # arXiv ids are strings and keep their leading zeros (CLAUDE.md 3.9)
            assert isinstance(a["id"], str) and "." in a["id"]

    ids = [a["id"] for c in cols for a in c["anchors"]]
    assert len(ids) == len(set(ids)), "an anchor appears in two collections"


# ── Routes ───────────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch, repo):
    import asyncio

    import app.config as cfg

    path = str(tmp_path / "coll.db")
    monkeypatch.setattr(cfg, "DB_PATH", path)
    monkeypatch.setattr(db_mod, "DB_PATH", path)

    import app.user_state as us
    us._cache.clear()

    import app.turso_svc as turso

    async def fake_meta(ids):
        return {i: {"arxiv_id": i, "title": f"Paper {i}", "abstract": "abs.",
                    "authors": "[]", "category": "cs.LG", "published": "2024-01-01",
                    "year": 2024, "citation_count": 3} for i in ids}

    monkeypatch.setattr(turso, "fetch_metadata_batch", fake_meta)

    async def no_extend(slug, limit=12, exclude=None):
        return []

    monkeypatch.setattr(collections_svc, "extend", no_extend)

    asyncio.run(db_mod.init_db())
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_index_lists_collections(client):
    r = client.get("/collections")
    assert r.status_code == 200
    assert "Alpha Track" in r.text and "Beta Track" in r.text


def test_detail_renders_anchors_and_notes(client):
    r = client.get("/collections/alpha")
    assert r.status_code == 200
    assert "Alpha Track" in r.text
    assert "note 0" in r.text, "curator note missing from the page"


def test_unknown_slug_is_404_not_500(client):
    assert client.get("/collections/does-not-exist").status_code == 404


# ── Following ────────────────────────────────────────────────────────────────

def test_following_seeds_the_profile(client, monkeypatch):
    """The point of the feature: one click puts a user past the Tier 1 threshold."""
    import app.qdrant_svc as qs

    async def fake_vectors(ids):
        rng = np.random.default_rng(0)
        return {i: rng.random(1024, dtype=np.float32).tolist() for i in ids}

    monkeypatch.setattr(qs, "get_paper_vectors", fake_vectors)

    r = client.post("/api/collections/alpha/follow")
    assert r.status_code == 200
    assert "Following" in r.text
    assert "6 papers added" in r.text

    import asyncio
    import sqlite3

    uid = client.cookies.get("arxiv_user_id")
    rows = asyncio.run(db_mod.get_user_interactions(uid, event_types=["save"]))
    assert len(rows) == 6

    # Instrumentation must survive the seeding path (CLAUDE.md §3.11).
    # get_user_interactions only selects three columns, so read the table.
    conn = sqlite3.connect(db_mod.DB_PATH)
    tagged = conn.execute(
        "SELECT candidate_source, policy_id, propensity, source FROM interactions "
        "WHERE user_id = ? AND event_type = 'save'", (uid,)).fetchall()
    conn.close()
    assert len(tagged) == 6
    assert all(t == ("collection:alpha", "collection_seed", 1.0, "collection")
               for t in tagged)


def test_following_is_idempotent(client, monkeypatch):
    import app.qdrant_svc as qs

    async def fake_vectors(ids):
        return {i: np.zeros(1024, dtype=np.float32).tolist() for i in ids}

    monkeypatch.setattr(qs, "get_paper_vectors", fake_vectors)

    client.post("/api/collections/alpha/follow")
    client.post("/api/collections/alpha/follow")

    import asyncio
    rows = asyncio.run(db_mod.get_user_interactions(
        client.cookies.get("arxiv_user_id"), event_types=["save"]))
    assert len(rows) == 6, "a second follow re-saved the same anchors"


def test_follow_survives_a_vector_store_outage(client, monkeypatch):
    """Papers must still reach the library when Qdrant is unavailable.

    The profile will be under-trained, which the server logs -- but the follow
    must not 500.
    """
    import app.qdrant_svc as qs

    async def boom(ids):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(qs, "get_paper_vectors", boom)

    r = client.post("/api/collections/alpha/follow")
    assert r.status_code == 200


def test_unfollow_does_not_unsave(client, monkeypatch):
    """The EWMA has no exact inverse, so unfollowing cannot rewind the profile.

    Anything else would leave a permanent smudge while claiming to undo.
    """
    import app.qdrant_svc as qs

    async def fake_vectors(ids):
        return {i: np.zeros(1024, dtype=np.float32).tolist() for i in ids}

    monkeypatch.setattr(qs, "get_paper_vectors", fake_vectors)

    client.post("/api/collections/alpha/follow")
    r = client.post("/api/collections/alpha/unfollow")
    assert r.status_code == 200
    assert "Follow" in r.text

    import asyncio
    uid = client.cookies.get("arxiv_user_id")
    assert await_len(asyncio, uid) == 6
    assert asyncio.run(db_mod.get_followed_slugs(uid)) == set()


def await_len(asyncio_mod, uid):
    return len(asyncio_mod.run(
        db_mod.get_user_interactions(uid, event_types=["save"])))


def test_unknown_slug_follow_is_not_an_error(client):
    assert client.post("/api/collections/ghost/follow").status_code == 200


# ── Medoid extension ─────────────────────────────────────────────────────────

async def test_extend_reads_the_shape_search_by_vector_actually_returns(repo, monkeypatch):
    """qdrant_svc.search_by_vector returns a list of arxiv_id STRINGS.

    Treating them as dicts extracted None from every hit, so extend() returned
    [] while the curated half of the page rendered perfectly -- it shipped and
    was only caught by checking the deployed page. This pins the contract.
    """
    import app.qdrant_svc as qs

    async def fake_vectors(ids):
        rng = np.random.default_rng(2)
        return {i: rng.random(1024, dtype=np.float32).tolist() for i in ids}

    seen = {}

    async def fake_search(vec, limit=20, exclude_ids=None):
        seen["limit"] = limit
        seen["exclude"] = set(exclude_ids or ())
        # The real function's return type: bare id strings. The last one is an
        # anchor of this collection, so it must be filtered out even though
        # exclude_ids was already passed down -- belt and braces.
        return ["9999.00001", "9999.00002", "2401.00000"]

    monkeypatch.setattr(qs, "get_paper_vectors", fake_vectors)
    monkeypatch.setattr(qs, "search_by_vector", fake_search)

    out = await collections_svc.extend("alpha", limit=5)

    assert out == ["9999.00001", "9999.00002"], \
        "extension dropped every hit -- the result shape is being misread"
    # Anchors must be excluded, and the exclusion pushed down to the query.
    assert seen["exclude"] >= set(collections_svc.anchor_ids("alpha"))


async def test_extend_returns_nothing_when_anchors_have_no_vectors(repo, monkeypatch):
    """Too few resolvable anchors means no medoid, so no extension section."""
    import app.qdrant_svc as qs

    async def no_vectors(ids):
        return {}

    monkeypatch.setattr(qs, "get_paper_vectors", no_vectors)
    assert await collections_svc.extend("alpha") == []


async def test_extend_survives_a_search_failure(repo, monkeypatch):
    """The curated list is the part with editorial value; it must not be lost."""
    import app.qdrant_svc as qs

    async def fake_vectors(ids):
        rng = np.random.default_rng(3)
        return {i: rng.random(1024, dtype=np.float32).tolist() for i in ids}

    async def boom(vec, limit=20, exclude_ids=None):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(qs, "get_paper_vectors", fake_vectors)
    monkeypatch.setattr(qs, "search_by_vector", boom)
    assert await collections_svc.extend("alpha") == []
