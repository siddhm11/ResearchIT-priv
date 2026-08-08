"""
Recommendation engine evaluation harness.

Bypasses HTTP and calls the same pipeline functions the router uses,
with full DB setup/cleanup per scenario. Each scenario probes a specific
behavior (which tier fired, how many clusters formed, whether suppression
removed disliked categories, etc.) rather than just "did we get results."

Run:  python scripts/eval_recs_quality.py
"""
from __future__ import annotations

import asyncio
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import aiosqlite

# Force UTF-8 stdout so unicode glyphs (>=, ->, etc.) don't crash on Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import qdrant_svc, db, turso_svc, user_state as us
from app.config import REC_LIMIT, DB_PATH
from app.recommend import profiles
from app.recommend.clustering import (
    compute_clusters, MIN_PAPERS_FOR_CLUSTERING,
)
from app.routers.recommendations import (
    _multi_interest_recommend, _ewma_recommend,
)


# ── Curated paper ids (verified-famous papers in each domain) ────────────────

NLP_PAPERS = [
    ("1706.03762", "Attention Is All You Need"),
    ("1810.04805", "BERT"),
    ("2005.14165", "GPT-3"),
    ("1907.11692", "RoBERTa"),
    ("1910.10683", "T5"),
    ("2203.02155", "InstructGPT"),
    ("2201.11903", "CoT Prompting"),
    ("2307.09288", "Llama 2"),
]

CV_PAPERS = [
    ("1512.03385", "ResNet"),
    ("2010.11929", "Vision Transformer"),
    ("1409.1556",  "VGG"),
    ("1505.04597", "U-Net"),
    ("2103.14030", "Swin Transformer"),
    ("2104.14294", "DINO"),
    ("2112.10752", "Latent Diffusion"),
    ("1311.2524",  "R-CNN"),
]

ML_THEORY_PAPERS = [
    # cs.LG / stat.ML — used for negative-suppression test
    ("1607.06450", "Layer Normalization"),
    ("1502.03167", "Batch Normalization"),
    ("1412.6980",  "Adam optimizer"),
    ("1411.1784",  "Conditional GAN"),
]


# ── User setup / teardown helpers ────────────────────────────────────────────

async def setup_user(
    user_id: str,
    save_ids: list[str],
    dismiss_ids: list[str] | None = None,
    onboarding_categories: list[str] | None = None,
) -> object:
    """Build a test user from scratch: saves, dismisses, EWMA, in-memory state."""
    dismiss_ids = dismiss_ids or []

    if onboarding_categories:
        await db.save_onboarding_categories(user_id, onboarding_categories)

    # Pre-fetch all vectors in one batch
    all_ids = save_ids + dismiss_ids
    vecs = await qdrant_svc.get_paper_vectors(all_ids) if all_ids else {}

    # Cache metadata so category suppression / display work
    if all_ids:
        meta = await turso_svc.fetch_metadata_batch(all_ids)
        if meta:
            await db.cache_turso_metadata_batch(list(meta.values()))

    state = await us.ensure_loaded(user_id)

    for pid in save_ids:
        if pid not in vecs:
            print(f"  [setup] WARNING: {pid} not in Qdrant; skipping")
            continue
        state.add_positive(pid)
        emb = np.array(vecs[pid], dtype=np.float32)
        await profiles.update_on_save(user_id, emb)
        await db.log_interaction(user_id, pid, "save")

    for pid in dismiss_ids:
        if pid not in vecs:
            continue
        state.add_negative(pid)
        emb = np.array(vecs[pid], dtype=np.float32)
        await profiles.update_on_dismiss(user_id, emb)
        await db.log_interaction(user_id, pid, "not_interested")

    return state


async def cleanup_user(user_id: str) -> None:
    """Wipe all DB rows + in-memory cache for a test user."""
    async with aiosqlite.connect(DB_PATH) as conn:
        for sql in [
            "DELETE FROM interactions WHERE user_id = ?",
            "DELETE FROM user_profiles WHERE user_id = ?",
            "DELETE FROM user_clusters WHERE user_id = ?",
            "DELETE FROM user_onboarding WHERE user_id = ?",
            "DELETE FROM cluster_snapshots WHERE user_id = ?",
        ]:
            try:
                await conn.execute(sql, (user_id,))
            except Exception:
                pass
        await conn.commit()
    if user_id in us._cache:
        del us._cache[user_id]


# ── Pipeline runner (mirrors get_recommendations() cascade) ──────────────────

async def run_pipeline(user_id: str, state) -> tuple[str, list[str], dict, float]:
    """Returns (tier_label, rec_ids, paper_tags, latency_ms)."""
    seen = us.all_seen(user_id)
    n_saves = len(state.positive_list)

    t0 = time.perf_counter()

    # Tier 0: cold-start (no saves) → trending by category
    if n_saves == 0:
        cat_filter = await db.get_user_category_filter(user_id)
        if cat_filter:
            trending = await turso_svc.fetch_trending_by_categories(
                cat_filter, limit=REC_LIMIT,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            return ("Tier 0 trending",
                    [t["arxiv_id"] for t in trending],
                    {}, elapsed)
        elapsed = (time.perf_counter() - t0) * 1000
        return ("EMPTY (no onboarding)", [], {}, elapsed)

    # Tier 1: ≥5 saves → multi-interest clustering + quota
    if n_saves >= MIN_PAPERS_FOR_CLUSTERING:
        rec_ids, paper_tags = await _multi_interest_recommend(
            user_id, state, seen, REC_LIMIT, query_id="eval-test",
        )
        if rec_ids:
            elapsed = (time.perf_counter() - t0) * 1000
            return ("Tier 1 multi-interest", rec_ids, paper_tags, elapsed)

    # Tier 2: ≥3 saves (EWMA threshold internally) → single-vector search
    rec_ids = await _ewma_recommend(user_id, seen, REC_LIMIT)
    if rec_ids:
        elapsed = (time.perf_counter() - t0) * 1000
        return ("Tier 2 EWMA", rec_ids, {}, elapsed)

    # Tier 3: ≥1 save → Qdrant Recommend with raw IDs
    rec_ids = await qdrant_svc.recommend(
        positive_arxiv_ids=state.positive_list,
        negative_arxiv_ids=state.negative_list,
        seen_arxiv_ids=seen,
        limit=REC_LIMIT,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    if rec_ids:
        return ("Tier 3 Qdrant Recommend", rec_ids, {}, elapsed)
    return ("EMPTY (all tiers exhausted)", [], {}, elapsed)


async def report_results(rec_ids: list[str], paper_tags: dict) -> tuple[Counter, Counter]:
    """Print top-10 with category and cluster origin. Return (cat_counts, source_counts)."""
    if not rec_ids:
        print("    (no results)")
        return Counter(), Counter()

    meta = await turso_svc.fetch_metadata_batch(rec_ids)
    cats: Counter = Counter()
    sources: Counter = Counter()

    for i, aid in enumerate(rec_ids, 1):
        m = meta.get(aid, {})
        title = m.get("title", "(no title)")
        if len(title) > 65:
            title = title[:62] + "..."
        cat = m.get("category", "?")
        cats[cat] += 1
        tag = paper_tags.get(aid, {}) if paper_tags else {}
        source = tag.get("candidate_source", "")
        sources[source] += 1
        src_short = f"  [{source}]" if source else ""
        print(f"    {i:2d}. {aid:13s} {cat:14s}  {title}{src_short}")

    return cats, sources


# ── Scenarios ────────────────────────────────────────────────────────────────

async def scenario_1_cold_with_onboarding():
    """Tier 0: zero saves, NLP categories selected during onboarding."""
    user_id = f"eval-recs-1-{uuid.uuid4().hex[:6]}"
    print("\n" + "=" * 100)
    print("S1  Cold-start with onboarding categories (NLP)")
    print("    Expect: Tier 0 trending; results in NLP-adjacent friendly categories")
    print("=" * 100)
    try:
        await setup_user(user_id, save_ids=[], onboarding_categories=["nlp"])
        state = await us.ensure_loaded(user_id)
        tier, rec_ids, tags, lat = await run_pipeline(user_id, state)
        print(f"    Tier: {tier}  ({lat:.0f} ms)  Returned: {len(rec_ids)}")
        cats, _ = await report_results(rec_ids, tags)
        nlp_count = sum(
            c for k, c in cats.items()
            if k in {"AI/ML", "NLP/Computational Linguistics"} or k.startswith("cs.CL")
        )
        verdict = "PASS" if tier.startswith("Tier 0") and len(rec_ids) >= 5 else \
                  "FAIL  (Tier 0 broken — fetch_trending_by_categories returned 0)"
        print(f"    Categories: {dict(cats)}  -->  NLP count: {nlp_count}/{len(rec_ids)}")
        print(f"    VERDICT: {verdict}")
    finally:
        await cleanup_user(user_id)


async def scenario_2_single_save():
    """Tier 3: 1 save, expect Qdrant Recommend nearest-neighbors."""
    user_id = f"eval-recs-2-{uuid.uuid4().hex[:6]}"
    print("\n" + "=" * 100)
    print("S2  Single save (Vaswani Attention)")
    print("    Expect: Tier 3 Qdrant Recommend; results semantically near saved paper")
    print("=" * 100)
    try:
        await setup_user(user_id, save_ids=["1706.03762"])
        state = await us.ensure_loaded(user_id)
        tier, rec_ids, tags, lat = await run_pipeline(user_id, state)
        print(f"    Tier: {tier}  ({lat:.0f} ms)  Returned: {len(rec_ids)}")
        cats, _ = await report_results(rec_ids, tags)
        ml_count = sum(c for k, c in cats.items() if k in {"AI/ML", "NLP/Computational Linguistics"})
        verdict = "PASS" if tier.startswith("Tier 3") and ml_count >= 6 else "PARTIAL"
        print(f"    Categories: {dict(cats)}  -->  AI/ML + NLP count: {ml_count}/10")
        print(f"    VERDICT: {verdict}")
    finally:
        await cleanup_user(user_id)


async def scenario_3_three_nlp_saves():
    """Tier 2: 3 same-domain saves, expect EWMA single-vector search."""
    user_id = f"eval-recs-3-{uuid.uuid4().hex[:6]}"
    save_ids = [pid for pid, _ in NLP_PAPERS[:3]]
    print("\n" + "=" * 100)
    print("S3  Three NLP saves")
    print(f"    Saved: {save_ids}")
    print("    Expect: Tier 2 EWMA single-vector; results NLP-coherent")
    print("=" * 100)
    try:
        await setup_user(user_id, save_ids=save_ids)
        state = await us.ensure_loaded(user_id)
        tier, rec_ids, tags, lat = await run_pipeline(user_id, state)
        print(f"    Tier: {tier}  ({lat:.0f} ms)  Returned: {len(rec_ids)}")
        cats, _ = await report_results(rec_ids, tags)
        nlp_count = sum(c for k, c in cats.items() if k in {"AI/ML", "NLP/Computational Linguistics"})
        verdict = "PASS" if tier.startswith("Tier 2") and nlp_count >= 7 else "PARTIAL"
        print(f"    Categories: {dict(cats)}  -->  AI/ML + NLP count: {nlp_count}/10")
        print(f"    VERDICT: {verdict}")
    finally:
        await cleanup_user(user_id)


async def scenario_4_five_nlp_saves_single_cluster():
    """Tier 1, single interest: expect K=1 cluster, NLP-only output."""
    user_id = f"eval-recs-4-{uuid.uuid4().hex[:6]}"
    save_ids = [pid for pid, _ in NLP_PAPERS[:5]]
    print("\n" + "=" * 100)
    print("S4  Five NLP saves (single interest)")
    print(f"    Saved: {save_ids}")
    print("    Expect: Tier 1; 1 or few clusters; ML/NLP-coherent output")
    print("=" * 100)
    try:
        await setup_user(user_id, save_ids=save_ids)
        state = await us.ensure_loaded(user_id)
        # Inspect clusters explicitly
        vecs = await qdrant_svc.get_paper_vectors(save_ids)
        embs = np.array([vecs[p] for p in save_ids if p in vecs], dtype=np.float32)
        clusters = compute_clusters([p for p in save_ids if p in vecs], embs)
        print(f"    Clusters formed: K={len(clusters)}")
        for c in clusters:
            print(f"      cluster {c.cluster_idx}: medoid={c.medoid_paper_id}  importance={c.importance:.3f}  size={len(c.paper_ids)}")

        tier, rec_ids, tags, lat = await run_pipeline(user_id, state)
        print(f"    Tier: {tier}  ({lat:.0f} ms)  Returned: {len(rec_ids)}")
        cats, _ = await report_results(rec_ids, tags)
        nlp_count = sum(c for k, c in cats.items() if k in {"AI/ML", "NLP/Computational Linguistics"})
        verdict = "PASS" if tier.startswith("Tier 1") and nlp_count >= 7 else "PARTIAL"
        print(f"    Categories: {dict(cats)}  -->  AI/ML + NLP count: {nlp_count}/10")
        print(f"    VERDICT: {verdict}")
    finally:
        await cleanup_user(user_id)


async def scenario_5_multi_interest_balanced():
    """Tier 1, the headline test for quota fusion."""
    user_id = f"eval-recs-5-{uuid.uuid4().hex[:6]}"
    save_ids = [pid for pid, _ in NLP_PAPERS[:5]] + [pid for pid, _ in CV_PAPERS[:5]]
    print("\n" + "=" * 100)
    print("S5  Multi-interest (5 NLP + 5 CV)  -- THE HEADLINE QUOTA TEST")
    print(f"    Saved: 5x NLP + 5x CV")
    print("    Expect: K>=2 clusters, both interests visible, neither cluster swamps")
    print("=" * 100)
    try:
        await setup_user(user_id, save_ids=save_ids)
        state = await us.ensure_loaded(user_id)
        # Inspect clusters
        vecs = await qdrant_svc.get_paper_vectors(save_ids)
        aligned = [p for p in save_ids if p in vecs]
        embs = np.array([vecs[p] for p in aligned], dtype=np.float32)
        clusters = compute_clusters(aligned, embs)
        print(f"    Clusters formed: K={len(clusters)}")
        for c in clusters:
            print(f"      cluster {c.cluster_idx}: medoid={c.medoid_paper_id}  importance={c.importance:.3f}  size={len(c.paper_ids)}")

        tier, rec_ids, tags, lat = await run_pipeline(user_id, state)
        print(f"    Tier: {tier}  ({lat:.0f} ms)  Returned: {len(rec_ids)}")
        cats, sources = await report_results(rec_ids, tags)
        nlp_count = sum(c for k, c in cats.items() if k in {"AI/ML", "NLP/Computational Linguistics"})
        cv_count  = sum(c for k, c in cats.items() if k == "Computer Vision")
        print(f"    NLP (AI/ML + NLP): {nlp_count}   CV (Computer Vision): {cv_count}")
        print(f"    Cluster origin counts: {dict(sources)}")
        smaller = min(nlp_count, cv_count) if (nlp_count and cv_count) else 0
        verdict = "PASS" if len(clusters) >= 2 and smaller >= 3 else "FAIL"
        print(f"    VERDICT: {verdict}  (floor=3 enforced: {smaller >= 3})")
    finally:
        await cleanup_user(user_id)


async def scenario_6_multi_interest_imbalanced():
    """Tier 1: imbalanced split — does the floor=3 rescue the minority?"""
    user_id = f"eval-recs-6-{uuid.uuid4().hex[:6]}"
    save_ids = [pid for pid, _ in NLP_PAPERS[:8]] + [pid for pid, _ in CV_PAPERS[:2]]
    print("\n" + "=" * 100)
    print("S6  Multi-interest imbalanced (8 NLP + 2 CV)  -- FLOOR TEST")
    print("    Expect: if K>=2, CV gets >=3 slots even though importance is ~80/20")
    print("=" * 100)
    try:
        await setup_user(user_id, save_ids=save_ids)
        state = await us.ensure_loaded(user_id)
        vecs = await qdrant_svc.get_paper_vectors(save_ids)
        aligned = [p for p in save_ids if p in vecs]
        embs = np.array([vecs[p] for p in aligned], dtype=np.float32)
        clusters = compute_clusters(aligned, embs)
        print(f"    Clusters formed: K={len(clusters)}")
        for c in clusters:
            print(f"      cluster {c.cluster_idx}: medoid={c.medoid_paper_id}  importance={c.importance:.3f}  size={len(c.paper_ids)}")

        tier, rec_ids, tags, lat = await run_pipeline(user_id, state)
        print(f"    Tier: {tier}  ({lat:.0f} ms)  Returned: {len(rec_ids)}")
        cats, sources = await report_results(rec_ids, tags)
        nlp_count = sum(c for k, c in cats.items() if k in {"AI/ML", "NLP/Computational Linguistics"})
        cv_count  = sum(c for k, c in cats.items() if k == "Computer Vision")
        print(f"    NLP: {nlp_count}   CV: {cv_count}   Cluster sources: {dict(sources)}")
        if len(clusters) >= 2:
            verdict = "PASS" if cv_count >= 3 else "FAIL  (floor not enforced)"
        else:
            verdict = "AMBIGUOUS  (only 1 cluster formed - floor doesn't apply)"
        print(f"    VERDICT: {verdict}")
    finally:
        await cleanup_user(user_id)


async def scenario_7_category_suppression():
    """Tier 1 with dismissals: 'Computer Vision' should be suppressed."""
    # Save 5 NLP, dismiss 3 CV — non-overlapping friendly categories
    user_id = f"eval-recs-7-{uuid.uuid4().hex[:6]}"
    save_ids = [pid for pid, _ in NLP_PAPERS[:5]]
    dismiss_ids = [pid for pid, _ in CV_PAPERS[:3]]
    print("\n" + "=" * 100)
    print("S7  Category suppression (5 NLP saves + 3 CV dismissals)")
    print("    Expect: 'Computer Vision' suppressed; zero CV papers in output")
    print("=" * 100)
    try:
        await setup_user(user_id, save_ids=save_ids, dismiss_ids=dismiss_ids)
        state = await us.ensure_loaded(user_id)
        suppressed = await db.get_suppressed_categories(user_id)
        print(f"    Suppressed categories detected: {suppressed}")

        tier, rec_ids, tags, lat = await run_pipeline(user_id, state)
        print(f"    Tier: {tier}  ({lat:.0f} ms)  Returned: {len(rec_ids)}")
        cats, _ = await report_results(rec_ids, tags)
        cv_count = cats.get("Computer Vision", 0)
        verdict = "PASS" if cv_count == 0 and "Computer Vision" in suppressed else \
                  "FAIL  (CV leaked through)" if cv_count > 0 else \
                  "PARTIAL  (no CV but suppression set empty)"
        print(f"    CV count in output: {cv_count}    VERDICT: {verdict}")
    finally:
        await cleanup_user(user_id)


async def scenario_8_hungarian_stability():
    """Cluster IDs should remain stable across reclusterings when one new save is added."""
    user_id = f"eval-recs-8-{uuid.uuid4().hex[:6]}"
    save_ids = [pid for pid, _ in NLP_PAPERS[:5]] + [pid for pid, _ in CV_PAPERS[:5]]
    new_save = NLP_PAPERS[5][0]   # 6th NLP paper (added later)
    print("\n" + "=" * 100)
    print("S8  Hungarian cluster-ID stability")
    print("    Run pipeline once -> save 1 more NLP paper -> run again")
    print("    Expect: same cluster_idx assigned to NLP cluster across runs")
    print("=" * 100)
    try:
        await setup_user(user_id, save_ids=save_ids)
        state = await us.ensure_loaded(user_id)

        # First run
        await run_pipeline(user_id, state)
        clusters_v1 = await db.get_user_clusters(user_id)
        v1 = {(c["cluster_idx"], c["medoid_paper_id"]) for c in clusters_v1}
        print(f"    After run 1: {sorted(v1)}")

        # Add one more NLP paper
        more_vecs = await qdrant_svc.get_paper_vectors([new_save])
        if new_save in more_vecs:
            state.add_positive(new_save)
            await profiles.update_on_save(user_id, np.array(more_vecs[new_save], dtype=np.float32))
            await db.log_interaction(user_id, new_save, "save")

        # Second run
        await run_pipeline(user_id, state)
        clusters_v2 = await db.get_user_clusters(user_id)
        v2 = {(c["cluster_idx"], c["medoid_paper_id"]) for c in clusters_v2}
        print(f"    After run 2: {sorted(v2)}")

        # Stability check: every (idx, medoid) in v1 still present in v2 (medoid may change but idx must stay)
        idx_v1 = {c["cluster_idx"]: c["medoid_paper_id"] for c in clusters_v1}
        idx_v2 = {c["cluster_idx"]: c["medoid_paper_id"] for c in clusters_v2}
        # All cluster_idx that existed in v1 should still exist in v2
        stable = all(k in idx_v2 for k in idx_v1)
        print(f"    Cluster IDs in v1: {sorted(idx_v1.keys())}   v2: {sorted(idx_v2.keys())}")
        print(f"    VERDICT: {'PASS  (all v1 cluster_idx preserved)' if stable else 'FAIL  (cluster_idx churned)'}")
    finally:
        await cleanup_user(user_id)


async def scenario_9_latency():
    """Latency sanity: full Tier 1 pipeline on 10 saved papers."""
    user_id = f"eval-recs-9-{uuid.uuid4().hex[:6]}"
    save_ids = [pid for pid, _ in NLP_PAPERS[:5]] + [pid for pid, _ in CV_PAPERS[:5]]
    print("\n" + "=" * 100)
    print("S9  Latency sanity (Tier 1, 10 saved papers)")
    print("    Expect: <30 ms compute (excluding metadata I/O); end-to-end <2s")
    print("=" * 100)
    try:
        await setup_user(user_id, save_ids=save_ids)
        state = await us.ensure_loaded(user_id)
        # Warm: run once to load profiles
        await run_pipeline(user_id, state)
        # Time multiple runs
        runs = []
        for i in range(3):
            tier, _, _, lat = await run_pipeline(user_id, state)
            runs.append(lat)
            print(f"    Run {i+1}: {tier}  {lat:.0f} ms")
        print(f"    Mean: {sum(runs)/len(runs):.0f} ms   Min: {min(runs):.0f} ms   Max: {max(runs):.0f} ms")
        # The 30ms compute target excludes Qdrant + Turso I/O — full e2e includes them
        e2e_pass = max(runs) < 2000
        print(f"    VERDICT: {'PASS (e2e <2s)' if e2e_pass else 'PARTIAL  (over 2s e2e — investigate)'}")
    finally:
        await cleanup_user(user_id)


# ── Pre-flight + main ────────────────────────────────────────────────────────

async def preflight():
    """Verify all curated paper IDs exist in Qdrant before running."""
    all_ids = [p[0] for p in NLP_PAPERS + CV_PAPERS + ML_THEORY_PAPERS]
    vecs = await qdrant_svc.get_paper_vectors(all_ids)
    missing = [pid for pid in all_ids if pid not in vecs]
    if missing:
        print(f"WARNING: {len(missing)} curated IDs not in Qdrant: {missing}")
        print("Some scenarios may produce skewed results.")
    else:
        print(f"Pre-flight: all {len(all_ids)} curated paper IDs present in Qdrant.")


async def wipe_all_eval_users():
    """Belt-and-braces cleanup: remove any eval-recs-* users left from crashes."""
    async with aiosqlite.connect(DB_PATH) as conn:
        for tbl in ["interactions", "user_profiles", "user_clusters",
                    "user_onboarding", "cluster_snapshots"]:
            try:
                await conn.execute(f"DELETE FROM {tbl} WHERE user_id LIKE ?", ("eval-recs-%",))
            except Exception:
                pass
        await conn.commit()


async def main():
    print("=" * 100)
    print("RECOMMENDATION ENGINE EVALUATION")
    print("=" * 100)
    await db.init_db()
    await wipe_all_eval_users()
    await preflight()

    scenarios = [
        scenario_1_cold_with_onboarding,
        scenario_2_single_save,
        scenario_3_three_nlp_saves,
        scenario_4_five_nlp_saves_single_cluster,
        scenario_5_multi_interest_balanced,
        scenario_6_multi_interest_imbalanced,
        scenario_7_category_suppression,
        scenario_8_hungarian_stability,
        scenario_9_latency,
    ]

    for s in scenarios:
        try:
            await s()
        except Exception as e:
            import traceback
            print(f"  SCENARIO ERROR: {e}")
            traceback.print_exc()

    # Final safety wipe in case any cleanup_user calls failed
    await wipe_all_eval_users()
    print("\n" + "=" * 100)
    print("DONE — all eval-recs-* users wiped from DB")
    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())
