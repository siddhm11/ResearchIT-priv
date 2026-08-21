"""
The multi-interest guarantee, tested where it actually has to hold: on the
page the user sees.

`test_fusion.py` covers the fusion primitives in isolation. These tests run the
real Tier-1 stages in sequence -- allocate_quotas -> merge_quota_results ->
rerank_candidates -> mmr_rerank -> enforce_quota_on_ranking -- against a
synthetic user with genuinely distinct interests, and assert on the composition
of the served first page.

That end-to-end framing is the point. The defect these guard against was
invisible to unit tests of any single stage: the quota was allocated correctly,
the merge respected it, the reranker sorted correctly and MMR diversified
correctly, and the feed still arrived as a single interest -- because the merge
emitted cluster blocks and every later stage was blind to what a cluster was.
"""
from collections import Counter

import numpy as np
import pytest

from app.recommend.diversity import mmr_rerank
from app.recommend.fusion import (
    allocate_quotas,
    enforce_quota_on_ranking,
    merge_quota_results,
)
from app.recommend.reranker import rerank_candidates

DIM = 1024
PAGE_SIZE = 10          # _PAGE_SIZE in recommendations.py
FEED_POOL = 60          # _FEED_POOL
OVERSAMPLE = 3          # _OVERSAMPLE


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-10)


def _world(n_clusters: int, seed: int = 7, per_cluster: int = 400):
    """Well-separated interest directions, each with a corpus around it."""
    rng = np.random.default_rng(seed)
    basis = _unit(rng.normal(size=(n_clusters, DIM)))
    q, _ = np.linalg.qr(basis.T)
    medoids = _unit(q.T[:n_clusters])

    ids, embs, owner = [], [], {}
    for ci in range(n_clusters):
        for j in range(per_cluster):
            aid = f"c{ci}.{j:04d}"
            ids.append(aid)
            embs.append(_unit(medoids[ci] + 0.28 * rng.normal(size=DIM)))
            owner[aid] = ci
    return medoids, ids, np.array(embs, dtype=np.float32), owner, rng


def _serve(importances, *, seed=7, lt_bias=0, enforce=True):
    """Run the Tier-1 stages and return (first_page_ids, owner_map)."""
    medoids, ids, embs, owner, rng = _world(len(importances), seed=seed)
    emb_of = dict(zip(ids, embs))

    quotas = allocate_quotas(importances, total_slots=100, min_slots=3)

    per_cluster, score_map, cluster_of = [], {}, {}
    for ci, (medoid, quota) in enumerate(zip(medoids, quotas)):
        sims = embs @ medoid
        order = np.argsort(-sims)[: quota * OVERSAMPLE]
        hits = [ids[i] for i in order]
        per_cluster.append(hits)
        for i in order:
            aid = ids[i]
            cluster_of.setdefault(aid, ci)
            score_map[aid] = max(score_map.get(aid, -9.0), float(sims[i]))

    candidates = merge_quota_results(per_cluster, quotas)

    cand_embs = np.array([emb_of[a] for a in candidates], dtype=np.float32)
    meta = [
        {"arxiv_id": a, "category": "AI/ML", "arxiv_categories": "cs.LG",
         "published": "2024-06-01", "citation_count": 5,
         "influential_citations": 1, "authors": "A B", "title": a}
        for a in candidates
    ]

    # A long-term profile that leans on one interest -- the realistic case, and
    # the one that drags every score toward the dominant cluster.
    lt = _unit(medoids[lt_bias] + 0.15 * rng.normal(size=DIM)).astype(np.float32)

    ranked_ids, scores, ranked_embs = rerank_candidates(
        candidate_ids=candidates,
        candidate_embeddings=cand_embs,
        candidate_metadata=meta,
        long_term_vec=lt,
        qdrant_scores=np.array([score_map[a] for a in candidates], dtype=np.float32),
        cluster_importance=np.array(
            [importances[cluster_of[a]] for a in candidates], dtype=np.float32),
        cluster_medoid=np.stack(
            [medoids[cluster_of[a]] for a in candidates]).astype(np.float32),
        user_total_saves=8,
        user_total_dismissals=2,
    )

    feed = mmr_rerank(lt, ranked_embs, ranked_ids, scores,
                      lambda_param=0.6, top_k=FEED_POOL)
    if enforce:
        feed = enforce_quota_on_ranking(
            feed, cluster_of, dict(enumerate(importances)))
    return feed[:PAGE_SIZE], owner, feed


def _composition(page, owner):
    return Counter(owner[a] for a in page)


# ── The guarantee ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("importances,expect_interests", [
    ([0.75, 0.25], 2),
    ([0.60, 0.30, 0.10], 3),
    ([0.50, 0.50], 2),
    ([0.40, 0.30, 0.20, 0.10], 4),
])
def test_first_page_shows_every_interest(importances, expect_interests):
    """Every interest the user has must appear on the first screen.

    This is the product's whole premise: a feed that surfaces a user's distinct
    research areas without collapsing toward the dominant one.
    """
    page, owner, _ = _serve(importances)
    comp = _composition(page, owner)
    assert len(comp) == expect_interests, (
        f"importances {importances} -> first page held {dict(comp)}; "
        f"expected all {expect_interests} interests represented"
    )


def test_first_page_is_proportional_to_importance():
    """Balance, not parity: a 60/30/10 user gets roughly 6/3/1 cards."""
    page, owner, _ = _serve([0.60, 0.30, 0.10])
    comp = _composition(page, owner)
    assert comp[0] > comp[1] >= comp[2], f"not importance-ordered: {dict(comp)}"
    assert comp[0] <= 8, f"dominant interest took {comp[0]}/10 of the page"


def test_equally_important_interests_split_the_page():
    """Two equal interests split the first page near-evenly.

    Measured before the fix: a correct 50/50 candidate pool was served as a
    9/1 first page, because the long-term EWMA profile leaned one way.
    """
    page, owner, _ = _serve([0.5, 0.5])
    comp = _composition(page, owner)
    assert min(comp.values()) >= 4, f"expected a near-even page, got {dict(comp)}"


def test_single_interest_user_is_unaffected():
    """A user with one interest sees an ordinary relevance-ranked feed."""
    page, owner, feed = _serve([1.0])
    assert len(page) == PAGE_SIZE
    assert set(_composition(page, owner)) == {0}


def test_enforcement_is_what_rescues_the_page():
    """Without the serving-level stage the feed collapses; with it, it does not.

    Pins the specific regression rather than the general property, so that a
    future change which quietly drops the stage fails loudly here.
    """
    importances = [0.6, 0.3, 0.1]
    without, owner, _ = _serve(importances, enforce=False)
    with_, _, _ = _serve(importances, enforce=True)
    assert len(_composition(with_, owner)) > len(_composition(without, owner))


def test_enforcement_never_drops_or_duplicates_papers():
    """The serving stage is a permutation of the ranked feed."""
    _, _, feed = _serve([0.6, 0.3, 0.1])
    assert len(feed) == len(set(feed)), "duplicate papers in the served feed"
    assert len(feed) == FEED_POOL


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_guarantee_holds_across_corpora(seed):
    """Not an artefact of one lucky synthetic corpus."""
    page, owner, _ = _serve([0.6, 0.3, 0.1], seed=seed)
    assert len(_composition(page, owner)) == 3


@pytest.mark.parametrize("lt_bias", [0, 1, 2])
def test_guarantee_holds_whichever_interest_the_profile_favours(lt_bias):
    """The dominant EWMA direction must not decide the page's composition."""
    page, owner, _ = _serve([0.6, 0.3, 0.1], lt_bias=lt_bias)
    assert len(_composition(page, owner)) == 3


# ── MMR must not truncate an interest away ───────────────────────────────────
#
# Doc 06 §3.5: "Quota (3.1) handles cross-cluster diversity. MMR handles
# within-quota redundancy." A single global MMR over the merged pool does
# neither — it is cluster-blind AND it truncates. Measured on a live
# two-interest user: the merge handed it a correct 61/39 pool, it selected 39
# dominant + 1 minority, and dumped the other 60 minority papers into the
# exploration pool. Enforcing quota afterwards cannot repair that; by then
# there is one minority paper left to arrange.

def _skewed_pool(n_major=61, n_minor=39, seed=11):
    """A merged pool whose minority papers all score below its majority."""
    rng = np.random.default_rng(seed)
    med = _unit(rng.normal(size=(2, DIM)))
    q, _ = np.linalg.qr(med.T)
    med = _unit(q.T[:2])

    ids, embs, scores, cluster_of = [], [], [], {}
    for ci, n in enumerate((n_major, n_minor)):
        for j in range(n):
            aid = f"c{ci}.{j}"
            v = _unit(med[ci] + 0.25 * rng.normal(size=DIM))
            ids.append(aid)
            embs.append(v.astype(np.float32))
            # Majority scores strictly above every minority paper — the real
            # case, since the EWMA profile leans toward the dominant interest.
            scores.append((1.0 if ci == 0 else 0.0) + rng.random() * 0.5)
            cluster_of[aid] = ci
    return ids, np.array(embs, dtype=np.float32), scores, cluster_of, med


def test_global_mmr_truncates_the_minority_interest():
    """Pins the defect, so the reason for per-cluster MMR stays legible."""
    ids, embs, scores, cluster_of, med = _skewed_pool()
    selected = mmr_rerank(med[0], embs, ids, scores, lambda_param=0.6, top_k=60)
    minority = sum(1 for a in selected if cluster_of[a] == 1)
    assert minority <= 5, (
        f"fixture no longer reproduces the truncation ({minority} minority "
        "papers survived global MMR)"
    )


def test_per_cluster_mmr_preserves_both_interests():
    """The fix: each interest is diversified against its OWN slot budget."""
    ids, embs, scores, cluster_of, med = _skewed_pool()
    idx_of = {a: i for i, a in enumerate(ids)}

    groups = {}
    for a in ids:
        groups.setdefault(cluster_of[a], []).append(a)

    importances = [0.61, 0.39]
    budgets = allocate_quotas(importances, total_slots=60, min_slots=1)

    per_group = []
    for k, budget in zip(sorted(groups), budgets):
        g = groups[k]
        rows = [idx_of[a] for a in g]
        per_group.append(mmr_rerank(
            med[0], embs[rows], g, [scores[i] for i in rows],
            lambda_param=0.6, top_k=min(budget, len(g)),
        ))

    selected = merge_quota_results(per_group, [len(g) for g in per_group])
    counts = Counter(cluster_of[a] for a in selected)

    assert len(counts) == 2, "an interest was still lost entirely"
    assert counts[1] >= 15, (
        f"minority interest got only {counts[1]} of {len(selected)} slots; "
        f"its 39% entitlement of 60 is ~23"
    )
    # And it is present on the first page, not just somewhere in the pool.
    assert any(cluster_of[a] == 1 for a in selected[:PAGE_SIZE])
