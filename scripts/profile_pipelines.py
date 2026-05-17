"""
Stage-by-stage profiler for the search and recommendation pipelines.

Mirrors the production paths (hybrid_search_svc.search and
_multi_interest_recommend) with explicit timers between every stage,
so we can see where the time actually goes.

Run: python scripts/profile_pipelines.py
"""
from __future__ import annotations

import asyncio
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import (
    config, embed_svc, qdrant_svc, zilliz_svc, groq_svc, turso_svc,
    db, user_state as us,
)
from app.recommend import profiles
from app.recommend.clustering import (
    compute_clusters, stabilize_cluster_ids, save_clusters_to_db,
    load_clusters_from_db, MIN_PAPERS_FOR_CLUSTERING, InterestCluster,
)
from app.recommend.fusion import allocate_quotas, merge_quota_results
from app.recommend.reranker import rerank_candidates
from app.recommend.diversity import mmr_rerank, inject_exploration


@contextmanager
def stage(name: str, sink: list):
    t0 = time.perf_counter()
    yield
    sink.append((name, (time.perf_counter() - t0) * 1000))


def print_breakdown(label: str, timings: list[tuple[str, float]]):
    total = sum(t for _, t in timings)
    print(f"\n  --- {label} ---")
    print(f"  {'Stage':<46s} {'ms':>10s}  {'%':>6s}")
    print(f"  {'-'*46} {'-'*10}  {'-'*6}")
    for name, t in timings:
        pct = (100.0 * t / total) if total > 0 else 0.0
        print(f"  {name:<46s} {t:>10.0f}  {pct:>5.1f}%")
    print(f"  {'-'*46} {'-'*10}  {'-'*6}")
    print(f"  {'TOTAL':<46s} {total:>10.0f}  {100.0:>5.1f}%")


# ── Search pipeline profiler ─────────────────────────────────────────────────

async def profile_search(query: str) -> list[tuple[str, float]]:
    """Mirror hybrid_search_svc.search() with stage timers."""
    timings: list[tuple[str, float]] = []
    limit = 10
    fetch_k = limit * config.SEARCH_FETCH_K_MULTIPLIER

    # Stage 1: Groq rewrite
    rewritten = query
    with stage("1. Groq rewrite (LLM)", timings):
        try:
            rewritten = await groq_svc.rewrite(query)
        except Exception:
            rewritten = query

    # Stage 2: BGE-M3 encode (original)
    with stage("2a. BGE-M3 encode (original)", timings):
        d_orig, s_orig = embed_svc.encode_query(query)

    encodings = [(d_orig, s_orig)]

    # Stage 2b: BGE-M3 encode (rewritten, if different)
    if rewritten and rewritten != query:
        with stage("2b. BGE-M3 encode (rewrite)", timings):
            d_rw, s_rw = embed_svc.encode_query(rewritten)
        encodings.append((d_rw, s_rw))
    else:
        timings.append(("2b. BGE-M3 encode (rewrite skipped)", 0.0))

    # Stage 3: Parallel Qdrant + Zilliz searches
    with stage(f"3. Parallel search ({len(encodings)*2} tasks)", timings):
        tasks = []
        for d, s in encodings:
            tasks.append(qdrant_svc.search_dense(d.tolist(), limit=fetch_k))
            tasks.append(zilliz_svc.search_sparse(s, limit=fetch_k))
        raw = await asyncio.gather(*tasks, return_exceptions=True)

    valid_lists = [r for r in raw if not isinstance(r, Exception) and r]

    # Stage 4: RRF fusion
    with stage("4. RRF fusion", timings):
        from app.hybrid_search_svc import _rrf_fuse_multi, _title_match_rerank
        fused = _rrf_fuse_multi(valid_lists, k=config.SEARCH_RRF_K)

    # Stage 5: Title-boost (Turso fetch + scoring)
    with stage("5. Title-match boost (Turso + score)", timings):
        ranked = await _title_match_rerank(fused, query, top_n_for_boost=50)

    return timings


# ── Recommendations Tier 1 pipeline profiler ─────────────────────────────────

async def profile_recs_tier1(user_id: str, save_ids: list[str]) -> list[tuple[str, float]]:
    """Mirror _multi_interest_recommend() with stage timers."""
    timings: list[tuple[str, float]] = []

    state = await us.ensure_loaded(user_id)
    seen = us.all_seen(user_id)
    REC_LIMIT = config.REC_LIMIT
    OVERSAMPLE = 3
    ST_SUPPLEMENT = 20
    positives = state.positive_list

    # 1. Fetch saved-paper vectors from Qdrant
    with stage("1. Fetch saved-paper vectors (Qdrant)", timings):
        vectors = await qdrant_svc.get_paper_vectors(positives)

    aligned_ids = [pid for pid in positives if pid in vectors]
    aligned_embs = np.array([vectors[pid] for pid in aligned_ids], dtype=np.float32)

    # 2. Ward clustering (CPU)
    with stage("2. Ward clustering (CPU)", timings):
        clusters = compute_clusters(aligned_ids, aligned_embs)

    # 3. Hungarian: load + match
    with stage("3. Hungarian load+match (SQLite + numpy)", timings):
        old_clusters_data = await load_clusters_from_db(user_id)
        if old_clusters_data:
            old_clusters = []
            for row in old_clusters_data:
                mpid = row["medoid_paper_id"]
                if mpid in vectors:
                    medoid_emb = np.array(vectors[mpid], dtype=np.float32)
                elif row.get("medoid_embedding_blob") is not None:
                    medoid_emb = np.frombuffer(
                        row["medoid_embedding_blob"], dtype=np.float32
                    ).copy()
                else:
                    continue
                old_clusters.append(InterestCluster(
                    cluster_idx=row["cluster_idx"],
                    medoid_paper_id=mpid,
                    medoid_embedding=medoid_emb,
                    paper_ids=[],
                    importance=row["importance"],
                ))
            if old_clusters:
                clusters = stabilize_cluster_ids(clusters, old_clusters)

    # 4. Save clusters + snapshot (SQLite writes)
    with stage("4. Save clusters + snapshot (SQLite)", timings):
        await save_clusters_to_db(user_id, clusters)
        await db.save_cluster_snapshot(user_id, [
            {
                "cluster_idx": c.cluster_idx,
                "medoid_paper_id": c.medoid_paper_id,
                "importance": c.importance,
                "paper_ids": c.paper_ids,
                "medoid_embedding_blob": c.medoid_embedding.astype(np.float32).tobytes(),
            }
            for c in clusters
        ])

    # 5. Quota allocation (CPU)
    with stage("5. Allocate quotas (CPU)", timings):
        importances = [c.importance for c in clusters]
        quotas = allocate_quotas(importances, total_slots=100, min_slots=3)

    # 6. Load short-term profile
    with stage("6. Load short-term profile (SQLite)", timings):
        st_vec = await profiles.load_profile(user_id, "short_term")

    # 7. Per-cluster parallel ANN searches (no with_vectors — that path
    # is 10x slower on Qdrant Cloud free tier; we cache vectors instead)
    with stage(f"7. Per-cluster ANN searches (gather {len(clusters)})", timings):
        search_tasks = [
            qdrant_svc.search_by_vector_with_scores(
                query_vector=c.medoid_embedding.tolist(),
                limit=quota * OVERSAMPLE,
                exclude_ids=seen,
            )
            for c, quota in zip(clusters, quotas)
        ]
        per_cluster_scored = await asyncio.gather(*search_tasks)

    paper_cluster_map: dict[str, int] = {}
    qdrant_score_map: dict[str, float] = {}
    for cluster, scored in zip(clusters, per_cluster_scored):
        for hit in scored:
            aid = hit["arxiv_id"]
            if aid not in paper_cluster_map:
                paper_cluster_map[aid] = cluster.cluster_idx
            if aid not in qdrant_score_map or hit["score"] > qdrant_score_map[aid]:
                qdrant_score_map[aid] = float(hit["score"])

    per_cluster_ids = [
        [h["arxiv_id"] for h in scored] for scored in per_cluster_scored
    ]
    candidate_ids = merge_quota_results(per_cluster_ids, quotas)

    # 8. Short-term supplement search
    with stage("8. Short-term supplement (Qdrant)", timings):
        if st_vec is not None:
            seen_so_far = seen | set(candidate_ids)
            st_scored = await qdrant_svc.search_by_vector_with_scores(
                query_vector=st_vec.tolist(),
                limit=ST_SUPPLEMENT,
                exclude_ids=seen_so_far,
            )
            for hit in st_scored:
                aid = hit["arxiv_id"]
                if aid not in set(candidate_ids):
                    candidate_ids.append(aid)
                    paper_cluster_map[aid] = -1
                if aid not in qdrant_score_map:
                    qdrant_score_map[aid] = float(hit["score"])

    # 9. Fetch candidate vectors (LRU-cached by arxiv_id in qdrant_svc).
    with stage(f"9. Fetch {len(candidate_ids)} candidate vectors (Qdrant, cached)", timings):
        cand_vectors = await qdrant_svc.get_paper_vectors(candidate_ids)

    # 10. Fetch candidate metadata from Turso (with cache)
    with stage(f"10. Fetch {len(candidate_ids)} candidate metadata (Turso)", timings):
        cand_meta = await turso_svc.fetch_metadata_batch(candidate_ids)

    # 11. Cache metadata to SQLite
    with stage("11. Cache Turso metadata to SQLite", timings):
        await db.cache_turso_metadata_batch(list(cand_meta.values()))

    valid_ids = [cid for cid in candidate_ids if cid in cand_vectors and cid in cand_meta]
    valid_embs = np.array([cand_vectors[cid] for cid in valid_ids], dtype=np.float32)
    valid_meta = [cand_meta[cid] for cid in valid_ids]

    # 12. Load profiles (long-term, negative)
    with stage("12. Load long-term + negative profiles (SQLite)", timings):
        lt_vec = await profiles.load_profile(user_id, "long_term")
        neg_vec = await profiles.load_profile(user_id, "negative")

    # 13. SQLite reads (suppression + onboarding)
    with stage("13. Suppression + onboarding lookup (SQLite)", timings):
        suppressed = await db.get_suppressed_categories(user_id)
        onboarding_categories = await db.get_user_category_filter(user_id)

    # 14. Build feature arrays (CPU)
    with stage("14. Build per-candidate feature arrays (CPU)", timings):
        user_total_saves = len(state.positive_list)
        user_total_dismissals = len(state.negative_list)
        qdrant_scores = np.asarray(
            [qdrant_score_map.get(cid, 0.0) for cid in valid_ids],
            dtype=np.float32,
        )
        per_cand_imp = np.asarray(
            [
                clusters[paper_cluster_map[cid]].importance
                if cid in paper_cluster_map and 0 <= paper_cluster_map[cid] < len(clusters)
                else 0.0
                for cid in valid_ids
            ],
            dtype=np.float32,
        )
        per_cand_med = np.stack(
            [
                np.asarray(clusters[paper_cluster_map[cid]].medoid_embedding, dtype=np.float32)
                if cid in paper_cluster_map and 0 <= paper_cluster_map[cid] < len(clusters)
                else np.zeros(1024, dtype=np.float32)
                for cid in valid_ids
            ],
            axis=0,
        )
        is_suppressed_arr = np.asarray(
            [1.0 if cand_meta.get(cid, {}).get("category", "") in suppressed else 0.0
             for cid in valid_ids],
            dtype=np.float32,
        )
        onb_match_arr = np.asarray(
            [1.0 if cand_meta.get(cid, {}).get("category", "") in onboarding_categories else 0.0
             for cid in valid_ids],
            dtype=np.float32,
        )

    # 15. LightGBM rerank
    with stage("15. LightGBM rerank (CPU)", timings):
        reranked_ids, reranked_scores, reranked_embs = rerank_candidates(
            candidate_ids=valid_ids,
            candidate_embeddings=valid_embs,
            candidate_metadata=valid_meta,
            long_term_vec=lt_vec,
            short_term_vec=st_vec,
            negative_vec=neg_vec,
            qdrant_scores=qdrant_scores,
            cluster_importance=per_cand_imp,
            cluster_medoid=per_cand_med,
            is_suppressed_category=is_suppressed_arr,
            onboarding_category_match=onb_match_arr,
            user_total_saves=user_total_saves,
            user_total_dismissals=user_total_dismissals,
        )

    # 16. MMR
    with stage("16. MMR diversity (CPU)", timings):
        query_vec = lt_vec if lt_vec is not None else aligned_embs.mean(axis=0)
        mmr_selected = mmr_rerank(
            query_embedding=query_vec,
            candidate_embeddings=reranked_embs,
            candidate_ids=reranked_ids,
            scores=reranked_scores,
            lambda_param=0.6,
            top_k=REC_LIMIT,
        )

    # 17. Exploration injection
    with stage("17. Exploration injection (CPU)", timings):
        final = inject_exploration(
            selected_ids=mmr_selected,
            all_candidate_ids=reranked_ids,
            n_explore=2,
        )

    return timings


# ── Setup helper for recs profile ────────────────────────────────────────────

async def setup_recs_user(user_id: str, save_ids: list[str]):
    vecs = await qdrant_svc.get_paper_vectors(save_ids)
    state = await us.ensure_loaded(user_id)
    for pid in save_ids:
        if pid not in vecs:
            continue
        state.add_positive(pid)
        emb = np.array(vecs[pid], dtype=np.float32)
        await profiles.update_on_save(user_id, emb)
        await db.log_interaction(user_id, pid, "save")


async def cleanup_user(user_id: str):
    import aiosqlite
    async with aiosqlite.connect(config.DB_PATH) as conn:
        for tbl in ["interactions", "user_profiles", "user_clusters",
                    "user_onboarding", "cluster_snapshots"]:
            try:
                await conn.execute(f"DELETE FROM {tbl} WHERE user_id = ?", (user_id,))
            except Exception:
                pass
        await conn.commit()
    if user_id in us._cache:
        del us._cache[user_id]


async def main():
    print("=" * 92)
    print("PIPELINE PROFILER")
    print("=" * 92)

    await db.init_db()

    # Warm BGE-M3 + Turso connection so first stage isn't a 15s outlier
    print("\nWarming up BGE-M3 + Turso...")
    embed_svc.encode_query("warmup")
    await turso_svc.fetch_metadata_batch(["1706.03762"])

    # ── Search profiling ────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("SEARCH PIPELINE — three representative queries")
    print("=" * 92)

    queries = [
        ("known-item title", "attention is all you need"),
        ("conceptual rewrite", "when AI makes up fake facts"),
        ("academic, no rewrite", "BGE-M3 multilingual dense retrieval"),
    ]
    for label, q in queries:
        print(f"\n>>> Query [{label}]: {q!r}")
        # Run twice — first cold, second warm — to show cache effect
        for run in (1, 2):
            timings = await profile_search(q)
            print_breakdown(f"Run {run}", timings)

    # ── Recs Tier 1 profiling ───────────────────────────────────────────────
    print("\n\n" + "=" * 92)
    print("RECS TIER 1 PIPELINE — 10 saved papers (5 NLP + 5 CV)")
    print("=" * 92)

    user_id = f"profile-recs-{uuid.uuid4().hex[:6]}"
    save_ids = [
        "1706.03762", "1810.04805", "2005.14165", "1907.11692", "1910.10683",
        "1512.03385", "2010.11929", "1409.1556", "1505.04597", "2103.14030",
    ]
    try:
        await setup_recs_user(user_id, save_ids)

        for run in (1, 2, 3):
            timings = await profile_recs_tier1(user_id, save_ids)
            print_breakdown(f"Run {run}", timings)
    finally:
        await cleanup_user(user_id)


if __name__ == "__main__":
    asyncio.run(main())
