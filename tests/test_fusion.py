"""
Tests for importance-weighted quota fusion.

Covers:
  - Proportional allocation (dominant cluster gets most slots)
  - Floor guarantee (every cluster gets at least min_slots)
  - Total slots == sum of allocated slots (or >= when floors force it)
  - Remainder distributed correctly
  - Single cluster gets all slots
  - Equal importances → roughly equal allocation
  - Zero importances fall back to equal distribution
  - merge_quota_results deduplication, interleaving and order
  - enforce_quota_on_ranking: quota survives the global re-sort
"""
from app.recommend.fusion import (
    allocate_quotas,
    enforce_quota_on_ranking,
    merge_quota_results,
)


# ── allocate_quotas ───────────────────────────────────────────────────────────

def test_proportional_allocation():
    """Dominant cluster should receive proportionally more slots."""
    importances = [7.0, 3.0]
    slots = allocate_quotas(importances, total_slots=100, min_slots=3)
    assert len(slots) == 2
    assert slots[0] > slots[1], "Dominant cluster (imp=7) should get more slots than minor (imp=3)"


def test_floor_guarantee():
    """Every cluster must receive at least min_slots regardless of importance."""
    # One huge cluster and one tiny one
    importances = [99.0, 1.0]
    slots = allocate_quotas(importances, total_slots=100, min_slots=3)
    assert all(s >= 3 for s in slots), f"Floor violated: {slots}"


def test_total_slots_met():
    """Sum of allocated slots should equal total_slots when no floor pressure."""
    importances = [5.0, 3.0, 2.0]
    total = 100
    slots = allocate_quotas(importances, total_slots=total, min_slots=3)
    assert sum(slots) == total, f"Expected sum={total}, got {sum(slots)} from {slots}"


def test_floor_overrides_total():
    """When many clusters with min_slots exceed total, allocation may go over."""
    # 7 clusters × 3 min_slots = 21 > 20 total
    importances = [1.0] * 7
    slots = allocate_quotas(importances, total_slots=20, min_slots=3)
    assert all(s >= 3 for s in slots), f"Floor violated under pressure: {slots}"
    assert len(slots) == 7


def test_single_cluster_gets_all():
    """A single cluster should receive all slots (or min_slots if larger)."""
    slots = allocate_quotas([5.0], total_slots=50, min_slots=3)
    assert slots == [50]


def test_equal_importances_roughly_equal():
    """Equal importances should produce roughly equal slot counts."""
    importances = [1.0, 1.0, 1.0]
    slots = allocate_quotas(importances, total_slots=99, min_slots=3)
    assert len(slots) == 3
    assert slots == [33, 33, 33], f"Expected equal split [33,33,33], got {slots}"


def test_zero_importances_fallback():
    """All-zero importances should not crash; falls back to equal distribution."""
    importances = [0.0, 0.0, 0.0]
    slots = allocate_quotas(importances, total_slots=30, min_slots=3)
    assert len(slots) == 3
    assert sum(slots) == 30
    assert all(s >= 3 for s in slots)


def test_empty_importances():
    """Empty input returns empty list."""
    assert allocate_quotas([], total_slots=100) == []


def test_remainder_distributed():
    """With 3 equal clusters and 100 slots, remainder 1 goes to someone."""
    importances = [1.0, 1.0, 1.0]
    # 100 / 3 = 33.333 → floor is 33 each, remainder = 1
    slots = allocate_quotas(importances, total_slots=100, min_slots=3)
    assert sum(slots) == 100
    assert sorted(slots) == [33, 33, 34]


def test_two_cluster_sum_correct():
    """70/30 split on 100 slots: sum should be exactly 100."""
    slots = allocate_quotas([70.0, 30.0], total_slots=100, min_slots=3)
    assert sum(slots) == 100
    assert slots[0] >= slots[1]
    assert slots[1] >= 3


def test_doc06_worked_example():
    """
    Doc 06 worked example:
      importances = [0.55, 0.30, 0.15], total=30, min=3
      raw = [16.5, 9.0, 4.5]
      floor = [16, 9, 4]  (sum=29)
      remainder = 1 → largest frac (0.5 at idx 0) gets it
      final = [17, 9, 4]
    """
    slots = allocate_quotas([0.55, 0.30, 0.15], total_slots=30, min_slots=3)
    assert slots == [17, 9, 4], f"Doc 06 example expected [17, 9, 4], got {slots}"
    assert sum(slots) == 30


def test_doc06_tiny_cluster_floor():
    """
    Doc 06 tiny-cluster edge case:
      importances = [0.60, 0.25, 0.10, 0.05], total=30, min=3
      raw = [18.0, 7.5, 3.0, 1.5]
      floor applied: [18, 7, 3, 3]  -- smallest cluster gets 3 not 1
    """
    slots = allocate_quotas([0.60, 0.25, 0.10, 0.05], total_slots=30, min_slots=3)
    # The smallest cluster must get at least min_slots (3), not 1
    assert slots[3] >= 3, f"Floor violated: smallest cluster got {slots[3]}"
    # The dominant cluster still dominates
    assert slots[0] > slots[1] > slots[2]


def test_fractional_priority_deterministic():
    """
    Remainder should go to clusters with the largest fractional parts.
    importances=[10,10,10], total=20, min=3
      raw = [6.667, 6.667, 6.667]
      floor = [6, 6, 6]  (sum=18)
      remainder = 2 → all fractions equal (0.667), first two get +1 (stable sort)
      final = [7, 7, 6]
    """
    slots = allocate_quotas([10.0, 10.0, 10.0], total_slots=20, min_slots=3)
    assert sum(slots) == 20
    # With 2 remainder slots and 3 equal clusters, counts should be [7, 7, 6] in some order
    assert sorted(slots, reverse=True) == [7, 7, 6]


def test_fractional_priority_prefers_larger_frac():
    """
    Cluster with larger fractional part should receive remainder bonus first.
    importances=[2, 3] on 10 slots, min=3:
      raw = [4.0, 6.0]
      floor = [4, 6]  (sum=10, remainder=0)
      final = [4, 6]
    """
    slots = allocate_quotas([2.0, 3.0], total_slots=10, min_slots=3)
    assert slots == [4, 6]


def test_many_clusters_floor_overflow():
    """
    10 clusters, each needs min=3, but total=20 means 10×3=30 > 20.
    Floor guarantee overrides total — sum exceeds total_slots.
    """
    slots = allocate_quotas([1.0] * 10, total_slots=20, min_slots=3)
    assert len(slots) == 10
    assert all(s >= 3 for s in slots)
    # Floor overflow: sum exceeds requested total because min_slots dominates
    assert sum(slots) == 30


def test_zero_importances_respects_floor_edge():
    """
    Zero-importance with total < n × min should still respect floor.
    """
    slots = allocate_quotas([0.0, 0.0, 0.0], total_slots=6, min_slots=3)
    assert all(s >= 3 for s in slots)
    assert len(slots) == 3


def test_dominant_cluster_does_not_starve_minority():
    """
    Critical Doc 06 fairness test:
    User 70% NLP, 30% RL — RL must not get zero slots (the RRF failure mode).
    """
    slots = allocate_quotas([70.0, 30.0], total_slots=30, min_slots=3)
    assert slots[1] >= 3, f"Minority RL cluster starved: got {slots[1]}"
    assert slots[0] > slots[1]  # but dominance is still preserved
    assert sum(slots) == 30


def test_allocation_order_matches_input():
    """Output order must match input order (importance-ranked already by caller)."""
    slots = allocate_quotas([50.0, 25.0, 25.0], total_slots=100, min_slots=3)
    # Cluster 0 is the largest, gets most slots; clusters 1 and 2 tied
    assert slots[0] >= slots[1]
    assert slots[0] >= slots[2]


# ── merge_quota_results ───────────────────────────────────────────────────────

def test_merge_respects_quota():
    """Each cluster contributes at most its quota to the result."""
    cluster_a = ["a1", "a2", "a3", "a4", "a5"]
    cluster_b = ["b1", "b2", "b3"]
    result = merge_quota_results([cluster_a, cluster_b], quotas=[3, 3])
    a_count = sum(1 for r in result if r.startswith("a"))
    b_count = sum(1 for r in result if r.startswith("b"))
    assert a_count <= 3, f"Cluster A exceeded quota: {a_count}"
    assert b_count <= 3, f"Cluster B exceeded quota: {b_count}"


def test_merge_deduplicates():
    """Papers appearing in multiple clusters should appear only once."""
    cluster_a = ["shared", "a1", "a2"]
    cluster_b = ["shared", "b1", "b2"]
    result = merge_quota_results([cluster_a, cluster_b], quotas=[3, 3])
    assert result.count("shared") == 1, "Duplicate 'shared' should appear only once"


def test_merge_interleaves_clusters():
    """Clusters are interleaved, not concatenated.

    This test previously asserted ["a1", "a2", "b1", "b2"] -- i.e. it locked in
    the block ordering that made the multi-interest feed render as a single
    interest. Blocked output puts every minority-cluster paper behind the whole
    dominant block, and since `candidate_position` (reranker feature 1) is the
    index in THIS list, the dominant cluster also collected the entire
    `position_inverse` bonus. The docstring always said round-robin; the code
    did not.
    """
    result = merge_quota_results([["a1", "a2"], ["b1", "b2"]], quotas=[2, 2])
    assert result == ["a1", "b1", "a2", "b2"]


def test_merge_preserves_within_cluster_order():
    """A cluster's own ranking is never reordered by the interleave."""
    result = merge_quota_results([["a1", "a2", "a3"], ["b1", "b2"]], quotas=[3, 2])
    assert [r for r in result if r.startswith("a")] == ["a1", "a2", "a3"]
    assert [r for r in result if r.startswith("b")] == ["b1", "b2"]


def test_merge_spaces_clusters_proportionally():
    """A 3:1 quota split appears roughly 3:1 throughout, not 3 then 1.

    Strict alternation would be just as wrong as concatenation in the other
    direction: it would hand a 25% interest half of the head of the list.
    """
    big = [f"a{i}" for i in range(30)]
    small = [f"b{i}" for i in range(10)]
    result = merge_quota_results([big, small], quotas=[30, 10])

    # Every leading window is roughly 3:1, so the minority interest is present
    # early but never over-served.
    for window in (8, 16, 24):
        head = result[:window]
        n_small = sum(1 for r in head if r.startswith("b"))
        assert 1 <= n_small <= window // 3 + 1, (
            f"first {window}: {n_small} minority papers -- expected ~{window // 4}"
        )


def test_merge_first_page_spans_every_cluster():
    """The property the whole architecture exists for.

    With three interests present, a 10-card first page must show all three.
    Under the old block merge it showed only the dominant one.
    """
    clusters = [[f"c{c}.{i}" for i in range(40)] for c in range(3)]
    result = merge_quota_results(clusters, quotas=[60, 30, 10])
    first_page = {r.split(".")[0] for r in result[:10]}
    assert first_page == {"c0", "c1", "c2"}, f"first page only had {first_page}"


def test_merge_set_is_unchanged_by_interleaving():
    """Interleaving changes ORDER only -- never which papers are in the pool.

    Guards the fix against silently altering pool composition, which would
    change what the reranker gets to consider rather than just its arrangement.
    """
    clusters = [[f"c{c}.{i}" for i in range(50)] for c in range(3)]
    quotas = [30, 20, 10]

    # Reference: the previous block-concatenating behaviour.
    seen, expected = set(), []
    for ids, q in zip(clusters, quotas):
        count = 0
        for aid in ids:
            if count >= q:
                break
            if aid not in seen:
                expected.append(aid)
                seen.add(aid)
                count += 1

    result = merge_quota_results(clusters, quotas)
    assert set(result) == set(expected)
    assert len(result) == len(expected)


def test_merge_empty_cluster():
    """An empty cluster contributes nothing; others still fill their quota."""
    cluster_a = ["a1", "a2", "a3"]
    cluster_b: list[str] = []
    result = merge_quota_results([cluster_a, cluster_b], quotas=[3, 3])
    assert result == ["a1", "a2", "a3"]


def test_merge_empty_input():
    """No clusters → empty result."""
    assert merge_quota_results([], []) == []


# ── enforce_quota_on_ranking ──────────────────────────────────────────────────
#
# The pool-level quota is only half the guarantee. rerank_candidates sorts
# globally by score and MMR selects greedily, and neither knows what a cluster
# is -- so a pool with a correct split can still be SERVED as one interest.
# These cover the stage that defends the split all the way to the screen.

def _blocked(counts: dict[str, int]) -> tuple[list[str], dict[str, int]]:
    """A ranking that is perfectly cluster-sorted -- the worst realistic case."""
    ranked, cluster_of = [], {}
    for ci, (name, n) in enumerate(counts.items()):
        for i in range(n):
            aid = f"{name}.{i}"
            ranked.append(aid)
            cluster_of[aid] = ci
    return ranked, cluster_of


def test_enforce_is_a_permutation():
    """Nothing is dropped, duplicated, or invented."""
    ranked, cluster_of = _blocked({"a": 35, "b": 25})
    out = enforce_quota_on_ranking(ranked, cluster_of, {0: 0.5, 1: 0.5})
    assert sorted(out) == sorted(ranked)
    assert len(out) == len(ranked)


def test_enforce_preserves_within_cluster_order():
    """The ranker's judgment inside a cluster is untouched -- only the
    cross-cluster arrangement is re-derived from importance."""
    ranked, cluster_of = _blocked({"a": 20, "b": 10})
    out = enforce_quota_on_ranking(ranked, cluster_of, {0: 0.7, 1: 0.3})
    for name in ("a", "b"):
        kept = [r for r in out if r.startswith(name)]
        assert kept == [r for r in ranked if r.startswith(name)]


def test_enforce_balances_equally_important_interests():
    """Two equally important interests split the first page evenly.

    Measured before the fix: a 50/50 pool served an 80/20 first page, because
    the long-term EWMA profile leaned one way and dragged every score with it.
    """
    ranked, cluster_of = _blocked({"a": 35, "b": 25})
    out = enforce_quota_on_ranking(ranked, cluster_of, {0: 0.5, 1: 0.5})
    n_b = sum(1 for r in out[:10] if r.startswith("b"))
    assert 4 <= n_b <= 6, f"expected a near-even first page, got {10 - n_b}/{n_b}"


def test_enforce_rescues_a_starved_minority_interest():
    """A blocked ranking is the case that produced a single-interest feed."""
    ranked, cluster_of = _blocked({"a": 50, "b": 8, "c": 2})
    out = enforce_quota_on_ranking(
        ranked, cluster_of, {0: 0.6, 1: 0.3, 2: 0.1}
    )
    assert len({r.split(".")[0] for r in ranked[:10]}) == 1   # before: one interest
    assert len({r.split(".")[0] for r in out[:10]}) == 3      # after: all three


def test_enforce_respects_importance_ordering():
    """More important interests still get more of the page -- balance, not parity."""
    ranked, cluster_of = _blocked({"a": 40, "b": 20})
    out = enforce_quota_on_ranking(ranked, cluster_of, {0: 0.8, 1: 0.2})
    n_a = sum(1 for r in out[:10] if r.startswith("a"))
    assert n_a > 5, f"dominant interest should still lead the page, got {n_a}/10"


def test_enforce_single_cluster_is_a_noop():
    """Nothing to interleave — the ranking passes through untouched."""
    ranked, cluster_of = _blocked({"a": 12})
    assert enforce_quota_on_ranking(ranked, cluster_of, {0: 1.0}) == ranked


def test_enforce_handles_short_term_supplement():
    """Supplement papers (cluster -1) have no importance; they keep the share
    they already hold and are spread rather than clumped at the end."""
    ranked, cluster_of = _blocked({"a": 20, "b": 10})
    for i in range(6):
        aid = f"st.{i}"
        ranked.append(aid)
        cluster_of[aid] = -1
    out = enforce_quota_on_ranking(ranked, cluster_of, {0: 0.7, 1: 0.3})
    assert sorted(out) == sorted(ranked)
    positions = [i for i, r in enumerate(out) if r.startswith("st")]
    assert min(positions) < len(out) // 2, "supplement was clumped at the tail"


def test_enforce_tolerates_unknown_and_missing_clusters():
    """Never raises on ids absent from the map or importances absent for a
    cluster — both are reachable when a recluster races a cached feed."""
    ranked = [f"x{i}" for i in range(10)]
    assert sorted(enforce_quota_on_ranking(ranked, {}, {})) == sorted(ranked)
    cluster_of = {aid: i % 3 for i, aid in enumerate(ranked)}
    out = enforce_quota_on_ranking(ranked, cluster_of, {0: 0.5})
    assert sorted(out) == sorted(ranked)


def test_enforce_trivial_inputs():
    assert enforce_quota_on_ranking([], {}, {}) == []
    assert enforce_quota_on_ranking(["only"], {"only": 0}, {0: 1.0}) == ["only"]
