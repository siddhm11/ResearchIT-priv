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

    Takes up to `quota_k` unique results from each cluster, deduplicating
    globally, and INTERLEAVES the clusters so that each one's picks are spread
    through the merged list in proportion to its quota.

    Why the order matters, and why it is not cosmetic
    -------------------------------------------------
    This used to emit all of cluster 0, then all of cluster 1, and so on. The
    quota was still honoured — minority interests were never starved out of the
    candidate pool — but every one of their papers landed *after* the dominant
    cluster's entire block. Two things downstream then turned that into a
    single-interest feed:

      * `reranker.compute_features` sets feature 1 (`candidate_position`) to the
        index in THIS list, and feature 35 (`position_inverse`) to 1/(pos+1),
        which `heuristic_score` weights at 0.10 — a bonus worth as much as a
        0.25 swing in long-term cosine, but concentrated almost entirely in the
        first ~10 entries. Under block ordering those first entries are, by
        construction, all from the dominant cluster. Fusion order was therefore
        being laundered into a relevance signal: the merge decided position,
        position inflated the score, and the score decided the feed.
      * the served page is a prefix of the final ranking, so anything pushed
        down is not merely lower — it is on another page.

    Interleaving removes the bias at its source rather than compensating for it
    later: `candidate_position` becomes a rank that means the same thing for
    every cluster, so the positional prior stays a within-retriever relevance
    signal (which is what it is for) instead of a cross-cluster one.

    Spacing is proportional, not strict alternation. A cluster holding 75% of
    the quota should appear roughly three slots in four; round-robin one-each
    would instead give a 25% interest half the head of the list, over-serving
    it just as badly as block ordering under-served it. Cluster k's j-th pick
    is scheduled at virtual position (j + 0.5) / quota_k and the slots are
    emitted in ascending order — the standard stride schedule, which spreads
    each cluster evenly and breaks ties toward the larger quota.

    Args:
        per_cluster_ids: list of arxiv_id lists, one per cluster (importance order)
        quotas:          slot count for each cluster (same order)

    Returns:
        Merged list of arxiv_ids, deduplicated, quota-bounded per cluster.
        Same SET as the previous block-ordered merge — only the order differs.
    """
    # Schedule of (virtual_position, cluster_idx), one entry per allocated slot.
    slots: list[tuple[float, int]] = []
    for ci, quota in enumerate(quotas):
        if quota <= 0 or ci >= len(per_cluster_ids):
            continue
        for j in range(quota):
            slots.append(((j + 0.5) / quota, ci))

    # Ascending virtual position; ties go to the cluster with the larger quota
    # (equivalently, the earlier index, since `quotas` is in importance order).
    slots.sort(key=lambda s: (s[0], s[1]))

    seen: set[str] = set()
    result: list[str] = []
    cursors = [0] * len(per_cluster_ids)

    for _, ci in slots:
        ids = per_cluster_ids[ci]
        i = cursors[ci]
        # Advance past anything another cluster already claimed.
        while i < len(ids) and ids[i] in seen:
            i += 1
        if i >= len(ids):
            cursors[ci] = i
            continue        # this cluster is exhausted; its slot goes unused
        aid = ids[i]
        result.append(aid)
        seen.add(aid)
        cursors[ci] = i + 1

    return result


def enforce_quota_on_ranking(
    ranked_ids: list[str],
    cluster_of: dict[str, int],
    importance_of: dict[int, float],
    *,
    min_slots: int = 1,
) -> list[str]:
    """
    Re-arrange an already-ranked feed so its ORDER honours the cluster quotas.

    `allocate_quotas` + `merge_quota_results` bound the composition of the
    *candidate pool*. Nothing downstream then defends that composition: the
    reranker sorts globally by score and MMR selects greedily, so a feed whose
    pool is a correct 50/50 split of two equally important interests can still
    be served 80/20 — measured, not hypothesised — simply because the
    long-term EWMA profile leans toward one of them and drags every score with
    it. Quota that holds in the pool but not on the screen is not a guarantee;
    it is a guarantee's shadow.

    This closes that gap, and it is deliberately the *last* word on ordering:

      * Order WITHIN each cluster is preserved exactly. That is where the
        reranker's and MMR's judgment lives, and none of it is discarded — the
        best cluster-2 paper stays cluster 2's first pick.
      * Order ACROSS clusters is re-derived from importance, which is the one
        thing quota is supposed to own. Papers with no cluster (the short-term
        supplement, cluster -1) are treated as their own group weighted by the
        share of the feed they already hold, so the supplement is spread rather
        than clumped.
      * Nothing is dropped. Slots left unused by an exhausted cluster are
        backfilled from whatever remains, in the original ranked order, so the
        output is always a permutation of the input.

    Args:
        ranked_ids:    the final ordered feed (post rerank + MMR)
        cluster_of:    arxiv_id -> cluster index (-1 for the short-term supplement)
        importance_of: cluster index -> importance weight
        min_slots:     floor per present cluster, in feed slots

    Returns:
        A permutation of `ranked_ids` whose cross-cluster arrangement is
        proportional to importance.
    """
    if len(ranked_ids) < 2:
        return list(ranked_ids)

    # Group in rank order. `keys` is ordered by importance so that quota
    # allocation and tie-breaking both favour the stronger interest.
    groups: dict[int, list[str]] = {}
    for aid in ranked_ids:
        groups.setdefault(cluster_of.get(aid, -1), []).append(aid)

    if len(groups) < 2:
        return list(ranked_ids)

    total = len(ranked_ids)

    def weight(key: int) -> float:
        if key < 0:
            # No importance to read; a supplement earns the share it already
            # holds, so this stage neither promotes nor demotes it.
            return len(groups[key]) / total
        return float(importance_of.get(key, 0.0))

    keys = sorted(groups, key=lambda k: (-weight(k), k))
    quotas = allocate_quotas([weight(k) for k in keys], total, min_slots=min_slots)

    # Deliberately NOT capped to what each group actually holds. Capping looks
    # tidier but spreads a small group too thin: a cluster with 2 papers and a
    # 10%-of-60 allocation, capped to 2, gets its stride recomputed over the
    # whole feed and lands at the 25% and 75% marks — off the first page, which
    # is the one place its presence was supposed to be guaranteed. Left
    # uncapped, its 6 slots start near the top and the 4 it cannot fill are
    # simply skipped by merge_quota_results. Slots freed that way cost nothing:
    # the papers that would have used them are restored by the backfill below.
    merged = merge_quota_results([groups[k] for k in keys], quotas)

    # Backfill so the result is a permutation, not a truncation.
    placed = set(merged)
    merged.extend(aid for aid in ranked_ids if aid not in placed)
    return merged
