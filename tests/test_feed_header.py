"""
Tests for the feed's self-description: which tier served it, what interests it
spans, and how far the user is from the next tier.

These encode a product rule as much as a code path. The pipeline is gated —
1 save reaches the similarity tier, 3 the EWMA tier, 5 the clustered
multi-interest tier — and until now none of that reached the screen, so a user
on three saves silently got the weaker path and read it as a bad feed.
"""
from app.recommend.clustering import MIN_PAPERS_FOR_CLUSTERING
from app.routers.recommendations import (
    _MIN_EWMA_INTERACTIONS,
    _feed_interests,
    _profile_strength,
    _serving_tier,
)


def entry(tags: dict, ranked: list[str] | None = None) -> dict:
    return {"tags": tags, "ranked": ranked if ranked is not None else list(tags)}


def tag(source: str, *, label: str = "", tone: int = -1) -> dict:
    return {"candidate_source": source, "cluster_label": label, "cluster_tone": tone}


# ── Which tier served this feed ──────────────────────────────────────────────

def test_tier_is_read_from_the_tags_the_pipeline_wrote():
    assert _serving_tier(entry({"a": tag("trending_category_fallback")})) == 0
    assert _serving_tier(entry({"a": tag("qdrant_recommend")})) == 3
    assert _serving_tier(entry({"a": tag("ewma_longterm")})) == 2
    assert _serving_tier(entry({"a": tag("cluster_2", label="Vision", tone=2)})) == 1


def test_short_term_supplements_still_read_as_tier_1():
    # Only Tier 1 emits short_term_supplement at all.
    assert _serving_tier(entry({"a": tag("short_term_supplement")})) == 1


def test_empty_feed_defaults_to_tier_1():
    assert _serving_tier(entry({})) == 1


# ── Progress toward the next tier ────────────────────────────────────────────

def test_tier_2_user_is_told_what_clustering_costs():
    s = _profile_strength(entry({"a": tag("ewma_longterm")}), save_count=3)
    assert s is not None
    assert s["target"] == MIN_PAPERS_FOR_CLUSTERING
    assert s["remaining"] == MIN_PAPERS_FOR_CLUSTERING - 3
    assert "multi-interest" in s["unlocks"]


def test_tier_3_user_is_pointed_at_the_ewma_threshold():
    s = _profile_strength(entry({"a": tag("qdrant_recommend")}), save_count=1)
    assert s is not None
    assert s["target"] == _MIN_EWMA_INTERACTIONS
    assert s["remaining"] == _MIN_EWMA_INTERACTIONS - 1


def test_tier_1_has_nothing_left_to_unlock():
    e = entry({"a": tag("cluster_0", label="Vision", tone=0)})
    assert _profile_strength(e, save_count=12) is None


def test_no_nag_when_the_threshold_is_met_but_the_tier_did_not_engage():
    """
    A user can sit on 6 saves and still be served by Tier 2 — no vectors, an
    empty retrieval, a cluster that failed to build. Telling them to "save 2
    more" when they are already past the threshold is a promise the pipeline
    has just demonstrated it will not keep.
    """
    e = entry({"a": tag("ewma_longterm")})
    assert _profile_strength(e, save_count=MIN_PAPERS_FOR_CLUSTERING + 1) is None


def test_percentage_is_bounded():
    s = _profile_strength(entry({"a": tag("ewma_longterm")}), save_count=4)
    assert 0 <= s["pct"] <= 100


# ── Interest composition ─────────────────────────────────────────────────────

def test_interests_are_counted_over_the_whole_ranked_pool():
    e = entry(
        {
            "p1": tag("cluster_0", label="Computer vision", tone=0),
            "p2": tag("cluster_0", label="Computer vision", tone=0),
            "p3": tag("cluster_1", label="Combinatorics", tone=1),
        },
        ranked=["p1", "p2", "p3"],
    )
    out = _feed_interests(e)
    assert [(i["label"], i["count"]) for i in out] == [
        ("Computer vision", 2), ("Combinatorics", 1),
    ]


def test_interests_are_ordered_largest_first():
    """A cluster's share of the pool IS its importance weight after
    allocate_quotas(), so size order is quota order."""
    e = entry(
        {
            "p1": tag("cluster_1", label="Small", tone=1),
            "p2": tag("cluster_0", label="Big", tone=0),
            "p3": tag("cluster_0", label="Big", tone=0),
        },
        ranked=["p1", "p2", "p3"],
    )
    assert [i["label"] for i in _feed_interests(e)] == ["Big", "Small"]


def test_unlabelled_papers_are_excluded():
    # Exploration picks and short-term supplements carry no cluster label; the
    # strip describes clusters, so they must not appear as a blank chip.
    e = entry(
        {
            "p1": tag("cluster_0", label="Vision", tone=0),
            "p2": tag("exploration"),
            "p3": tag("short_term_supplement"),
        },
        ranked=["p1", "p2", "p3"],
    )
    out = _feed_interests(e)
    assert len(out) == 1 and out[0]["label"] == "Vision"


def test_non_tier_1_feeds_describe_no_interests():
    e = entry({"p1": tag("ewma_longterm")}, ranked=["p1"])
    assert _feed_interests(e) == []


def test_papers_outside_the_ranked_pool_are_not_counted():
    # `tags` covers the exploration pool too; only `ranked` is the feed.
    e = entry(
        {
            "p1": tag("cluster_0", label="Vision", tone=0),
            "p2": tag("cluster_0", label="Vision", tone=0),
        },
        ranked=["p1"],
    )
    assert _feed_interests(e)[0]["count"] == 1


def test_tone_travels_with_the_label():
    e = entry({"p1": tag("cluster_3", label="Robotics", tone=3)}, ranked=["p1"])
    assert _feed_interests(e)[0]["tone"] == 3


# ── Match meter ──────────────────────────────────────────────────────────────

from app.templates_env import _match_pct, _MATCH_CEIL, _MATCH_FLOOR


def test_match_pct_spans_the_band_real_scores_occupy():
    """A bar drawn on the raw 0-1 range would sit between a third and nine
    tenths full for every paper ever retrieved -- visually identical, so
    useless for comparing two cards."""
    assert _match_pct(_MATCH_FLOOR) == 0
    assert _match_pct(_MATCH_CEIL) == 100
    mid = _match_pct((_MATCH_FLOOR + _MATCH_CEIL) / 2)
    assert 45 <= mid <= 55


def test_match_pct_clamps_outside_the_band():
    assert _match_pct(0.99) == 100
    assert _match_pct(0.10) == 0


def test_match_pct_is_zero_for_absent_or_bad_values():
    # 0.0 is what a paper with no score carries; it must render no meter.
    assert _match_pct(0.0) == 0
    assert _match_pct(None) == 0
    assert _match_pct("") == 0
    assert _match_pct(-0.5) == 0
