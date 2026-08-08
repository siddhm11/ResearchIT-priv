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
  - merge_quota_results deduplication and order
"""
from app.recommend.fusion import allocate_quotas, merge_quota_results


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


def test_merge_preserves_order():
    """Cluster A results appear before Cluster B results."""
    cluster_a = ["a1", "a2"]
    cluster_b = ["b1", "b2"]
    result = merge_quota_results([cluster_a, cluster_b], quotas=[2, 2])
    assert result == ["a1", "a2", "b1", "b2"]


def test_merge_empty_cluster():
    """An empty cluster contributes nothing; others still fill their quota."""
    cluster_a = ["a1", "a2", "a3"]
    cluster_b: list[str] = []
    result = merge_quota_results([cluster_a, cluster_b], quotas=[3, 3])
    assert result == ["a1", "a2", "a3"]


def test_merge_empty_input():
    """No clusters → empty result."""
    assert merge_quota_results([], []) == []
