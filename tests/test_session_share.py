"""The recent-session supplement keeps its share of the feed (audit F4).

Raw interest importances (sums of recency weights, 4.0 and 2.0) used to be
allocated next to the supplement's pool share (0.17): 2 of 60 slots instead of
~10, and its first paper around card 31.
"""
from app.recommend.fusion import allocate_quotas, enforce_quota_on_ranking, group_weights

SIZES = {0: 30, 1: 20, -1: 10}
IMPORTANCE = {0: 4.0, 1: 2.0}


def test_interests_and_the_session_share_one_scale():
    w = group_weights(SIZES, IMPORTANCE)
    assert abs(sum(w.values()) - 1) < 1e-9
    assert abs(w[-1] - 10 / 60) < 1e-9                  # the share of the pool it holds
    assert abs(w[0] / w[1] - 2.0) < 1e-9                # interests still split by importance
    budgets = allocate_quotas([w[0], w[1], w[-1]], total_slots=60, min_slots=1)
    assert budgets == [33, 17, 10]                      # was [39, 19, 2]


def test_the_session_reaches_the_first_page():
    ranked = [f"a{i}" for i in range(30)] + [f"b{i}" for i in range(20)] + [f"s{i}" for i in range(10)]
    cluster_of = {**{f"a{i}": 0 for i in range(30)}, **{f"b{i}": 1 for i in range(20)},
                  **{f"s{i}": -1 for i in range(10)}}
    served = enforce_quota_on_ranking(ranked, cluster_of, IMPORTANCE)
    first = next(i for i, a in enumerate(served) if a.startswith("s"))
    assert first < 10                                   # was 30
    assert sum(a.startswith("s") for a in served[:12]) >= 2
    assert sorted(served) == sorted(ranked)             # still a permutation


def test_without_a_session_nothing_changes_between_interests():
    w = group_weights({0: 30, 1: 20}, IMPORTANCE)
    assert abs(w[0] - 2 / 3) < 1e-9 and abs(w[1] - 1 / 3) < 1e-9
