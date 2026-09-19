"""The exposure log — the denominator every Phase 7 metric needs.

`feed_impressions` answers "has this been on screen", and its
(user_id, paper_id) key is right for that. It cannot answer what evaluation
asks: a paper shown on five feeds is one row there, it carries no query_id,
no position and no propensity, so click-through rate had a numerator and no
denominator and every estimator in §3.11 had nothing to pair with.

`feed_exposures` is the log half. These tests pin the properties that make it
joinable, because a logging table that is subtly wrong is worse than none --
you only discover it when you try to analyse, by which point the data is gone.
"""
from __future__ import annotations

import pytest

from app import db
from app.routers.paper import _cluster_of


@pytest.fixture
async def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
    await db.init_db()
    return db.DB_PATH


def _exposure(**over):
    row = {
        "user_id": "u1", "paper_id": "2401.00001",
        "query_id": "q-abc", "position": 0, "tier": 1,
        "candidate_source": "cluster_2", "cluster_id": 2,
        "propensity": 1.0, "policy_id": "v9.1",
    }
    row.update(over)
    return row


async def _rows(path, sql="SELECT * FROM feed_exposures ORDER BY id"):
    import aiosqlite
    async with aiosqlite.connect(path) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute(sql)
        return [dict(r) for r in await cur.fetchall()]


# ── The property the impressions table could not provide ─────────────────────

async def test_showing_the_same_paper_twice_writes_two_rows(fresh_db):
    """This is the whole point. feed_impressions upserts; a log must append.

    Without it there is no exposure COUNT, so no rate has a denominator.
    """
    await db.record_exposures([_exposure(query_id="q1")])
    await db.record_exposures([_exposure(query_id="q2")])

    rows = await _rows(fresh_db)
    assert len(rows) == 2
    assert {r["query_id"] for r in rows} == {"q1", "q2"}


async def test_impressions_still_collapse_repeats(fresh_db):
    """The suppression set must keep its old behaviour, or churn breaks.

    These two tables answer different questions and the fix must not have
    merged them.
    """
    await db.record_impressions("u1", ["2401.00001"])
    await db.record_impressions("u1", ["2401.00001"])

    rows = await _rows(fresh_db, "SELECT * FROM feed_impressions")
    assert len(rows) == 1


async def test_exposures_join_to_interactions_on_query_id(fresh_db):
    """Per-feed CTR, end to end: three shown, one clicked."""
    shown = [_exposure(paper_id=f"2401.0000{i}", position=i) for i in range(3)]
    await db.record_exposures(shown)
    await db.log_interaction(
        user_id="u1", paper_id="2401.00001", event_type="click",
        source="recommendation", position=1, query_id="q-abc",
        candidate_source="cluster_2", cluster_id=2,
        propensity=1.0, policy_id="v9.1",
    )

    import aiosqlite
    async with aiosqlite.connect(fresh_db) as conn:
        cur = await conn.execute(
            """SELECT COUNT(DISTINCT e.paper_id),
                      COUNT(DISTINCT i.paper_id)
                 FROM feed_exposures e
                 LEFT JOIN interactions i
                        ON i.query_id = e.query_id
                       AND i.paper_id = e.paper_id
                       AND i.event_type = 'click'
                WHERE e.query_id = 'q-abc'""")
        shown_n, clicked_n = await cur.fetchone()

    assert (shown_n, clicked_n) == (3, 1)


async def test_per_cluster_ctr_is_computable(fresh_db):
    """The measurement that decides whether quota is earning its keep.

    Cluster 0 gets 4 exposures and 0 clicks, cluster 1 gets 2 and 1. If either
    side of this loses its cluster identity the query returns nothing useful,
    which is the state this replaces.
    """
    await db.record_exposures(
        [_exposure(paper_id=f"a{i}", candidate_source="cluster_0", cluster_id=0)
         for i in range(4)]
        + [_exposure(paper_id=f"b{i}", candidate_source="cluster_1", cluster_id=1)
           for i in range(2)])
    await db.log_interaction(
        user_id="u1", paper_id="b0", event_type="click", source="recommendation",
        query_id="q-abc", candidate_source="cluster_1", cluster_id=1,
        propensity=1.0, policy_id="v9.1")

    import aiosqlite
    async with aiosqlite.connect(fresh_db) as conn:
        cur = await conn.execute(
            """SELECT e.cluster_id, COUNT(*),
                      (SELECT COUNT(*) FROM interactions i
                        WHERE i.cluster_id = e.cluster_id
                          AND i.event_type = 'click')
                 FROM feed_exposures e GROUP BY e.cluster_id ORDER BY e.cluster_id""")
        got = await cur.fetchall()

    assert got == [(0, 4, 0), (1, 2, 1)]


# ── Field fidelity ───────────────────────────────────────────────────────────

async def test_every_phase_311_field_survives_the_write(fresh_db):
    await db.record_exposures([_exposure(position=7, propensity=0.25, tier=0)])
    row = (await _rows(fresh_db))[0]

    assert row["query_id"] == "q-abc"
    assert row["position"] == 7
    assert row["propensity"] == pytest.approx(0.25)
    assert row["policy_id"] == "v9.1"
    assert row["tier"] == 0
    assert row["cluster_id"] == 2


async def test_empty_string_cluster_becomes_null_not_zero(fresh_db):
    """The serving layer uses "" for "no cluster" because that renders as
    nothing. Coercing it to 0 would file exploration picks under cluster 0,
    which is a real cluster."""
    await db.record_exposures([_exposure(cluster_id="", candidate_source="exploration")])
    assert (await _rows(fresh_db))[0]["cluster_id"] is None


async def test_cluster_id_arriving_as_a_string_is_kept(fresh_db):
    """It round-trips through hx-vals as text."""
    await db.record_exposures([_exposure(cluster_id="3")])
    assert (await _rows(fresh_db))[0]["cluster_id"] == 3


async def test_rows_without_a_user_or_paper_are_dropped_not_written(fresh_db):
    await db.record_exposures([
        _exposure(user_id=""), _exposure(paper_id=""), _exposure(),
    ])
    assert len(await _rows(fresh_db)) == 1


async def test_recording_nothing_is_not_an_error(fresh_db):
    assert await db.record_exposures([]) == 0


async def test_exposures_are_pruned_by_age(fresh_db):
    import aiosqlite
    await db.record_exposures([_exposure()])
    async with aiosqlite.connect(fresh_db) as conn:
        await conn.execute(
            "UPDATE feed_exposures SET shown_at = datetime('now','-400 days')")
        await conn.commit()

    assert await db.prune_exposures(retention_days=180) == 1
    assert await _rows(fresh_db) == []


# ── The surface / candidate-source confusion (M2) ────────────────────────────

@pytest.mark.parametrize("src,expected", [
    ("cluster_0", 0),
    ("cluster_1", 1),
    ("cluster_12", 12),
    ("exploration", None),
    ("ewma_longterm", None),
    ("qdrant_recommend", None),
    ("trending_category_fallback", None),
    ("trending_default_fallback", None),
    ("paper_page", None),
    ("", None),
    ("cluster_", None),
    ("cluster_x", None),
])
def test_cluster_is_parsed_only_from_a_cluster_source(src, expected):
    """None, never 0, for sources that have no cluster -- 0 is a real index."""
    assert _cluster_of(src) == expected
