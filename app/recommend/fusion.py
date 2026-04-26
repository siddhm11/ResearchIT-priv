"""
Importance-weighted quota fusion for multi-interest recommendations.

Replaces RRF for the recommendation pipeline (not search).

RRF is correct for search (different retrievers, same query).
For recommendations (different cluster queries, same user), RRF lets
the dominant cluster drown minority interests.  Quota ensures every
interest cluster gets a guaranteed floor of slots.

Reference: doc 06 §3.1 — "importance-weighted quota with a floor"
  w_k = importance_k / sum(importance_k)
  slot_k = max(floor(F * w_k), F_min)   # F = total, F_min = 3
  # distribute remainder by largest fractional part
"""
from __future__ import annotations


def allocate_quotas(
    importances: list[float],
    total_slots: int,
    min_slots: int = 3,
) -> list[int]:
    """
    Allocate recommendation slots proportionally to cluster importances,
    with a guaranteed minimum per cluster.

    Args:
        importances: importance score per cluster, same order as clusters
        total_slots: total candidate slots to distribute (e.g. 100)
        min_slots:   minimum slots guaranteed to every cluster (default 3)

    Returns:
        List of slot counts, same length and order as importances.
        sum(result) >= total_slots (may exceed if floor constraints force it).
    """
    n = len(importances)
    if n == 0:
        return []
    if n == 1:
        return [max(total_slots, min_slots)]

    total_imp = sum(importances)

    if total_imp <= 0:
        # Degenerate: equal distribution with floor guarantee
        per = total_slots // n
        result = [per] * n
        for i in range(total_slots - per * n):
            result[i] += 1
        return [max(r, min_slots) for r in result]

    # Proportional raw allocations
    raw = [imp / total_imp * total_slots for imp in importances]

    # Apply floor: max(floor(raw_i), min_slots)
    floored = [max(int(r), min_slots) for r in raw]

    remainder = total_slots - sum(floored)

    if remainder <= 0:
        # Floor guarantees already account for all slots (or more)
        return floored

    # Distribute remainder slots by largest fractional part of raw allocations
    fracs = sorted(range(n), key=lambda i: raw[i] % 1.0, reverse=True)
    for j in range(remainder):
        floored[fracs[j % n]] += 1

    return floored


def merge_quota_results(
    per_cluster_ids: list[list[str]],
    quotas: list[int],
) -> list[str]:
    """
    Merge per-cluster search results respecting quota allocations.

    Takes up to `quota_k` unique results from each cluster in round-robin
    order across clusters (by importance rank), deduplicating globally.

    Args:
        per_cluster_ids: list of arxiv_id lists, one per cluster (importance order)
        quotas:          slot count for each cluster (same order)

    Returns:
        Merged list of arxiv_ids, deduplicated, quota-bounded per cluster.
    """
    seen: set[str] = set()
    result: list[str] = []

    for cluster_ids, quota in zip(per_cluster_ids, quotas):
        count = 0
        for aid in cluster_ids:
            if count >= quota:
                break
            if aid not in seen:
                result.append(aid)
                seen.add(aid)
                count += 1

    return result
