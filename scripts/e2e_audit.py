"""
End-to-end audit of the ResearchIT recommendation pipeline.

Steps:
  1. Smoke test: hybrid search (10 queries, per-layer scores)
  2. User profile pipeline: EWMA update + Ward clustering
  3. Recommendation feed generation with quota fusion
  4. LightGBM reranker pass
  5. Gap analysis

Run:  python scripts/e2e_audit.py
"""
from __future__ import annotations
import asyncio, sys, time, json, struct
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Imports ──────────────────────────────────────────────────────────────────

from app import hybrid_search_svc, turso_svc, embed_svc, qdrant_svc, zilliz_svc, groq_svc, db
from app.recommend import profiles, clustering
from app.recommend.reranker import (
    rerank_candidates, compute_features, heuristic_score,
    is_model_loaded, get_num_trees, FEATURE_NAMES,
)
from app.recommend.diversity import mmr_rerank, inject_exploration

# ── Globals ──────────────────────────────────────────────────────────────────

ERRORS: list[str] = []
WRONG_OUTPUTS: list[str] = []
MISSING: list[str] = []
TEST_USER = "e2e_audit_user_001"

# ── Helpers ──────────────────────────────────────────────────────────────────

def banner(text: str):
    print(f"\n{'='*90}")
    print(f"  {text}")
    print(f"{'='*90}\n")

def check(label: str, condition: bool, detail: str = ""):
    status = "OK" if condition else "FAIL"
    msg = f"  [{status:>4}] {label}"
    if detail:
        msg += f"  --  {detail}"
    print(msg)
    if not condition:
        WRONG_OUTPUTS.append(f"{label}: {detail}")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — SMOKE TEST: HYBRID SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

SEARCH_QUERIES = [
    "vision transformer image classification",
    "reinforcement learning reward shaping",
    "large language model fine-tuning RLHF",
    "graph neural network drug discovery",
    "federated learning differential privacy",
    "attention is all you need",
    "diffusion models image generation",
    "knowledge distillation BERT compression",
    "object detection YOLO real-time",
    "protein structure prediction deep learning",
]


async def step1_search():
    banner("STEP 1: HYBRID SEARCH SMOKE TEST")
    print(f"Running {len(SEARCH_QUERIES)} queries...\n")

    all_latencies = []
    all_results_count = []

    for i, q in enumerate(SEARCH_QUERIES, 1):
        t0 = time.perf_counter()
        try:
            results = await hybrid_search_svc.search(q, limit=10)
            elapsed = (time.perf_counter() - t0) * 1000
        except Exception as e:
            ERRORS.append(f"Step 1: Query {q!r} threw {type(e).__name__}: {e}")
            print(f"  Q{i}: {q!r} -> ERROR: {e}")
            continue

        all_latencies.append(elapsed)
        all_results_count.append(len(results))

        # Fetch metadata for display
        meta = {}
        if results:
            try:
                meta = await turso_svc.fetch_metadata_batch(results)
            except Exception as e:
                ERRORS.append(f"Step 1: Metadata fetch failed for {q!r}: {e}")

        print(f"  Q{i}: {q!r}")
        print(f"       Results: {len(results)}  |  Latency: {elapsed:.0f}ms")

        for rank, aid in enumerate(results[:5], 1):
            m = meta.get(aid, {})
            title = (m.get("title") or "?")[:65]
            cites = m.get("citation_count", 0) or 0
            print(f"       {rank}. [{cites:>6} cites] {aid:14s}  {title}")

        # Relevance check: does the query topic appear in at least 3/5 titles?
        if results and meta:
            q_words = set(q.lower().split())
            relevant = 0
            for aid in results[:5]:
                t = (meta.get(aid, {}).get("title") or "").lower()
                matches = sum(1 for w in q_words if w in t)
                if matches >= 2:
                    relevant += 1
            check(f"Q{i} relevance ({relevant}/5 top results overlap query terms)",
                  relevant >= 2,
                  f"{q!r}")

        print()

    # Summary
    if all_latencies:
        print(f"  --- Search Summary ---")
        print(f"  Queries: {len(all_latencies)}")
        print(f"  Avg latency: {sum(all_latencies)/len(all_latencies):.0f}ms")
        print(f"  p50: {sorted(all_latencies)[len(all_latencies)//2]:.0f}ms")
        print(f"  Max: {max(all_latencies):.0f}ms")
        zero_results = sum(1 for c in all_results_count if c == 0)
        print(f"  Zero-result queries: {zero_results}")
        if zero_results > 0:
            ERRORS.append(f"Step 1: {zero_results} queries returned 0 results")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — USER PROFILE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

# Real paper IDs from known categories:
# CV papers (computer vision)
CV_PAPERS = [
    "1512.03385",   # ResNet
    "2010.11929",   # ViT
    "2105.01601",   # Swin Transformer
    "2106.08254",   # BEiT
    "1409.1556",    # VGGNet
]
# LLM papers (NLP / language models)
LLM_PAPERS = [
    "1706.03762",   # Attention Is All You Need
    "1810.04805",   # BERT
    "2005.14165",   # GPT-3
    "2303.08774",   # GPT-4
    "2302.13971",   # LLaMA
]

ALL_SEED_PAPERS = CV_PAPERS + LLM_PAPERS


async def step2_profiles():
    banner("STEP 2: USER PROFILE PIPELINE")

    # Initialize DB
    await db.init_db()
    print(f"  Test user: {TEST_USER}")
    print(f"  Seed papers: {len(ALL_SEED_PAPERS)} (5 CV + 5 LLM)")

    # Step 2a: Retrieve embeddings for seed papers from Qdrant (batch)
    print(f"\n  Fetching embeddings from Qdrant for {len(ALL_SEED_PAPERS)} papers...")
    embeddings = {}
    try:
        vecs = await qdrant_svc.get_paper_vectors(ALL_SEED_PAPERS)
        for aid, vec in vecs.items():
            embeddings[aid] = np.array(vec, dtype=np.float32)
        missing = [a for a in ALL_SEED_PAPERS if a not in embeddings]
        if missing:
            print(f"    WARN: No vectors for {len(missing)} papers: {missing[:3]}...")
    except Exception as e:
        print(f"    ERROR: get_paper_vectors -> {e}")
        ERRORS.append(f"Step 2: get_paper_vectors failed: {e}")

    print(f"  Retrieved {len(embeddings)}/{len(ALL_SEED_PAPERS)} embeddings")

    if len(embeddings) < 5:
        ERRORS.append(f"Step 2: Only {len(embeddings)} embeddings retrieved, need >= 5")
        print("  ABORT: Not enough embeddings to continue Step 2")
        return None, None

    # Step 2b: EWMA profile updates
    print(f"\n  Running EWMA profile updates (alpha_long={profiles.ALPHA_LONG_TERM}, "
          f"alpha_short={profiles.ALPHA_SHORT_TERM})...")

    for aid in ALL_SEED_PAPERS:
        if aid not in embeddings:
            continue
        try:
            await profiles.update_on_save(TEST_USER, embeddings[aid])
        except Exception as e:
            ERRORS.append(f"Step 2: EWMA update failed for {aid}: {e}")
            print(f"    ERROR: update_on_save({aid}) -> {e}")

    # Load profiles back
    lt_vec = await profiles.load_profile(TEST_USER, "long_term")
    st_vec = await profiles.load_profile(TEST_USER, "short_term")
    lt_count = await profiles.get_interaction_count(TEST_USER, "long_term")
    st_count = await profiles.get_interaction_count(TEST_USER, "short_term")

    check("Long-term profile exists", lt_vec is not None)
    check("Short-term profile exists", st_vec is not None)
    check(f"Long-term interaction count = {lt_count}", lt_count == len(embeddings),
          f"expected {len(embeddings)}")
    check(f"Short-term interaction count = {st_count}", st_count == len(embeddings),
          f"expected {len(embeddings)}")

    if lt_vec is not None:
        lt_norm = float(np.linalg.norm(lt_vec))
        check(f"Long-term vector L2-norm ~= 1.0 (actual: {lt_norm:.4f})",
              abs(lt_norm - 1.0) < 0.01)

    if st_vec is not None:
        st_norm = float(np.linalg.norm(st_vec))
        check(f"Short-term vector L2-norm ~= 1.0 (actual: {st_norm:.4f})",
              abs(st_norm - 1.0) < 0.01)

    # Step 2c: Ward hierarchical clustering
    print(f"\n  Running Ward clustering on {len(embeddings)} paper embeddings...")

    paper_ids = list(embeddings.keys())
    emb_matrix = np.stack([embeddings[aid] for aid in paper_ids])

    try:
        clusters = clustering.compute_clusters(
            paper_ids=paper_ids,
            embeddings=emb_matrix,
        )
    except Exception as e:
        ERRORS.append(f"Step 2: compute_clusters failed: {e}")
        print(f"    ERROR: {e}")
        return lt_vec, st_vec

    print(f"  Clusters found: {len(clusters)}")
    for c in clusters:
        print(f"    Cluster {c.cluster_idx}: medoid={c.medoid_paper_id}, "
              f"papers={len(c.paper_ids)}, importance={c.importance:.3f}")
        for pid in c.paper_ids:
            label = "CV" if pid in CV_PAPERS else "LLM" if pid in LLM_PAPERS else "?"
            print(f"      - {pid} [{label}]")

    check(f"Number of clusters >= 2 (actual: {len(clusters)})",
          len(clusters) >= 2,
          "CV and LLM papers should form distinct clusters")

    # Check cluster purity
    for c in clusters:
        cv_count = sum(1 for p in c.paper_ids if p in CV_PAPERS)
        llm_count = sum(1 for p in c.paper_ids if p in LLM_PAPERS)
        total = len(c.paper_ids)
        purity = max(cv_count, llm_count) / total if total > 0 else 0
        dominant = "CV" if cv_count > llm_count else "LLM"
        check(f"Cluster {c.cluster_idx} purity ({dominant}: {purity:.0%})",
              purity >= 0.6,
              f"{cv_count} CV + {llm_count} LLM papers")

    # Save clusters for Step 3
    try:
        await clustering.save_clusters_to_db(TEST_USER, clusters)
    except Exception as e:
        ERRORS.append(f"Step 2: save_clusters_to_db failed: {e}")

    return lt_vec, st_vec


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — RECOMMENDATION FEED GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

async def step3_recommendation_feed(lt_vec, st_vec):
    banner("STEP 3: RECOMMENDATION FEED GENERATION")

    if lt_vec is None:
        ERRORS.append("Step 3: Skipped — no long-term profile from Step 2")
        print("  SKIPPED: No profile vectors from Step 2")
        return None, None, None

    # Load clusters from DB
    clusters = await clustering.load_clusters_from_db(TEST_USER)
    if not clusters:
        ERRORS.append("Step 3: No clusters found in DB")
        print("  SKIPPED: No clusters in DB")
        return None, None, None

    print(f"  Loaded {len(clusters)} clusters from DB")
    print(f"  Target feed size: 20 papers")

    # Step 3a: Search for candidates per cluster (using medoid embeddings)
    all_candidates: dict[str, dict] = {}  # arxiv_id -> metadata
    all_embeddings: dict[str, np.ndarray] = {}
    cluster_assignments: dict[str, int] = {}  # arxiv_id -> cluster_idx
    seen = set(ALL_SEED_PAPERS)

    t0 = time.perf_counter()

    # Get medoid vectors in batch
    medoid_ids = [c["medoid_paper_id"] for c in clusters]
    medoid_vecs = await qdrant_svc.get_paper_vectors(medoid_ids)

    for c in clusters:
        mid = c["medoid_paper_id"]
        medoid_vec = None

        # Try stored blob first
        if c.get("medoid_embedding_blob"):
            medoid_vec = np.frombuffer(c["medoid_embedding_blob"], dtype=np.float32)

        # Fallback: batch-fetched vector
        if medoid_vec is None and mid in medoid_vecs:
            medoid_vec = np.array(medoid_vecs[mid], dtype=np.float32)

        if medoid_vec is None:
            ERRORS.append(f"Step 3: No medoid vector for cluster {c['cluster_idx']}")
            continue

        # Search Qdrant for similar papers (with scores + vectors)
        try:
            results = await qdrant_svc.search_by_vector_with_scores(
                medoid_vec.tolist(), limit=30, with_vectors=True
            )
        except Exception as e:
            ERRORS.append(f"Step 3: search failed for cluster {c['cluster_idx']}: {e}")
            continue

        # Filter out seen papers
        for r in results:
            aid = r["arxiv_id"]
            if aid in seen:
                continue
            all_candidates[aid] = {"score": r["score"]}
            cluster_assignments[aid] = c["cluster_idx"]
            if "vector" in r:
                all_embeddings[aid] = np.array(r["vector"], dtype=np.float32)
            seen.add(aid)
            if len([a for a in cluster_assignments if cluster_assignments[a] == c["cluster_idx"]]) >= 15:
                break

    elapsed_search = (time.perf_counter() - t0) * 1000
    print(f"  Candidate search: {len(all_candidates)} papers in {elapsed_search:.0f}ms")

    if not all_candidates:
        ERRORS.append("Step 3: Zero candidates retrieved")
        print("  ABORT: No candidates")
        return None, None, None

    # Step 3b: Fetch metadata
    cand_ids = list(all_candidates.keys())
    try:
        meta = await turso_svc.fetch_metadata_batch(cand_ids)
    except Exception as e:
        ERRORS.append(f"Step 3: metadata fetch failed: {e}")
        meta = {}

    # Step 3c: Fetch embeddings for candidates (use what we got from search + batch fetch rest)
    cand_embeddings = dict(all_embeddings)  # Already have some from with_vectors=True
    missing_emb = [aid for aid in cand_ids if aid not in cand_embeddings]
    if missing_emb:
        print(f"  Fetching {len(missing_emb)} missing embeddings from Qdrant...")
        try:
            extra = await qdrant_svc.get_paper_vectors(missing_emb)
            for aid, vec in extra.items():
                cand_embeddings[aid] = np.array(vec, dtype=np.float32)
        except Exception as e:
            print(f"    WARN: batch vector fetch failed: {e}")

    print(f"  Got {len(cand_embeddings)}/{len(cand_ids)} embeddings")

    # Build aligned arrays
    valid_ids = [aid for aid in cand_ids if aid in cand_embeddings and aid in meta]
    if len(valid_ids) < 5:
        ERRORS.append(f"Step 3: Only {len(valid_ids)} valid candidates")
        print(f"  ABORT: Not enough valid candidates")
        return None, None, None

    emb_matrix = np.stack([cand_embeddings[aid] for aid in valid_ids])
    meta_list = [meta[aid] for aid in valid_ids]

    # Step 3d: Print the raw candidate feed
    print(f"\n  Raw candidate feed ({len(valid_ids)} papers):")
    cluster_counts: dict[int, int] = {}
    for i, aid in enumerate(valid_ids[:20]):
        m = meta.get(aid, {})
        title = (m.get("title") or "?")[:55]
        cites = m.get("citation_count", 0) or 0
        cidx = cluster_assignments.get(aid, -1)
        cluster_counts[cidx] = cluster_counts.get(cidx, 0) + 1
        print(f"    {i+1:2d}. [C{cidx}] [{cites:>6} cites] {title}")

    print(f"\n  Cluster distribution in top 20:")
    for cidx, count in sorted(cluster_counts.items()):
        print(f"    Cluster {cidx}: {count} papers")

    total_feed = (time.perf_counter() - t0) * 1000
    print(f"  Total feed generation: {total_feed:.0f}ms")

    return valid_ids, emb_matrix, meta_list


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — LIGHTGBM RERANKER
# ═══════════════════════════════════════════════════════════════════════════════

async def step4_reranker(valid_ids, emb_matrix, meta_list, lt_vec, st_vec):
    banner("STEP 4: LIGHTGBM RERANKER")

    if valid_ids is None:
        print("  SKIPPED: No candidates from Step 3")
        return

    print(f"  Model loaded: {is_model_loaded()}")
    if is_model_loaded():
        print(f"  Trees: {get_num_trees()}")
    else:
        MISSING.append("LightGBM model not loaded — using heuristic fallback")

    n = min(len(valid_ids), 20)
    ids_subset = valid_ids[:n]
    emb_subset = emb_matrix[:n]
    meta_subset = meta_list[:n]

    print(f"  Running reranker on {n} candidates...")
    t0 = time.perf_counter()

    try:
        sorted_ids, sorted_scores, sorted_embs = rerank_candidates(
            ids_subset,
            emb_subset,
            meta_subset,
            lt_vec,
            st_vec,
            None,  # no negative profile
        )
        elapsed = (time.perf_counter() - t0) * 1000
    except Exception as e:
        ERRORS.append(f"Step 4: rerank_candidates failed: {e}")
        print(f"  ERROR: {e}")
        return

    print(f"  Reranker latency: {elapsed:.0f}ms")
    print(f"\n  Reranked order (top 10):")

    # Fetch metadata for display
    re_meta = {}
    try:
        re_meta = await turso_svc.fetch_metadata_batch(sorted_ids[:10])
    except Exception:
        pass

    for i, (aid, score) in enumerate(zip(sorted_ids[:10], sorted_scores[:10]), 1):
        m = re_meta.get(aid, {})
        title = (m.get("title") or "?")[:55]
        cites = m.get("citation_count", 0) or 0
        old_rank = ids_subset.index(aid) + 1 if aid in ids_subset else "?"
        print(f"    {i:2d}. (was #{old_rank:>2}) [{cites:>6} cites] score={score:.4f}  {title}")

    # Feature analysis for top 3 and bottom 3
    features = compute_features(emb_subset, meta_subset, lt_vec, st_vec, None)
    print(f"\n  Feature snapshot (top 3 reranked papers):")
    for rank_idx in range(min(3, len(sorted_ids))):
        aid = sorted_ids[rank_idx]
        orig_idx = ids_subset.index(aid)
        f = features[orig_idx]
        print(f"    #{rank_idx+1} {aid}:")
        print(f"      qdrant_cosine={f[0]:.3f}  lt_sim={f[20]:.3f}  st_sim={f[21]:.3f}  "
              f"cites={f[2]:.0f}  recency={f[6]:.3f}  age_days={f[5]:.0f}")

    if len(sorted_ids) >= 3:
        print(f"\n  Feature snapshot (bottom 3 reranked papers):")
        for rank_idx in range(max(0, len(sorted_ids)-3), len(sorted_ids)):
            aid = sorted_ids[rank_idx]
            orig_idx = ids_subset.index(aid)
            f = features[orig_idx]
            print(f"    #{rank_idx+1} {aid}:")
            print(f"      qdrant_cosine={f[0]:.3f}  lt_sim={f[20]:.3f}  st_sim={f[21]:.3f}  "
                  f"cites={f[2]:.0f}  recency={f[6]:.3f}  age_days={f[5]:.0f}")

    # Check: did reranking change anything?
    moved = sum(1 for i, aid in enumerate(sorted_ids) if aid != ids_subset[i])
    check(f"Reranker changed {moved}/{n} positions", moved > 0,
          "Reranker should reorder candidates based on features")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — MMR DIVERSITY + EXPLORATION
# ═══════════════════════════════════════════════════════════════════════════════

async def step5_diversity(valid_ids, emb_matrix, lt_vec):
    banner("STEP 5: MMR DIVERSITY + EXPLORATION")

    if valid_ids is None or lt_vec is None:
        print("  SKIPPED: No data from previous steps")
        return

    n = min(len(valid_ids), 30)
    print(f"  Running MMR (lambda=0.6) on {n} candidates, selecting 15...")

    t0 = time.perf_counter()
    try:
        mmr_ids = mmr_rerank(
            lt_vec, emb_matrix[:n], valid_ids[:n],
            lambda_param=0.6, top_k=15,
        )
        elapsed = (time.perf_counter() - t0) * 1000
    except Exception as e:
        ERRORS.append(f"Step 5: mmr_rerank failed: {e}")
        print(f"  ERROR: {e}")
        return

    print(f"  MMR latency: {elapsed:.0f}ms")
    print(f"  MMR selected {len(mmr_ids)} papers")

    # Check rank changes
    moved = sum(1 for i, aid in enumerate(mmr_ids) if i < len(valid_ids) and aid != valid_ids[i])
    print(f"  Rank changes vs input: {moved}/{len(mmr_ids)}")

    # Exploration injection
    with_explore = inject_exploration(mmr_ids, valid_ids[:n], n_explore=2, seed=42)
    explore_count = len(with_explore) - len(mmr_ids)
    print(f"  Exploration injected: {explore_count} papers")
    check("Exploration added papers", explore_count > 0 or len(valid_ids[:n]) <= len(mmr_ids))

    # Check diversity: compute avg pairwise cosine among selected
    selected_embs = []
    for aid in mmr_ids[:10]:
        if aid in valid_ids:
            idx = valid_ids.index(aid)
            if idx < len(emb_matrix):
                selected_embs.append(emb_matrix[idx])

    if len(selected_embs) >= 2:
        sel_matrix = np.stack(selected_embs)
        norms = sel_matrix / (np.linalg.norm(sel_matrix, axis=1, keepdims=True) + 1e-10)
        sim_matrix = norms @ norms.T
        # Average off-diagonal similarity
        mask = ~np.eye(len(sel_matrix), dtype=bool)
        avg_sim = sim_matrix[mask].mean()
        print(f"  Avg pairwise cosine among top 10 MMR picks: {avg_sim:.3f}")
        check("MMR diversity (avg pairwise sim < 0.85)", avg_sim < 0.85,
              f"actual: {avg_sim:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — GAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def step6_gap_analysis():
    banner("STEP 6: GAP ANALYSIS")

    print("  ERRORS (things that threw exceptions or returned empty):")
    if ERRORS:
        for e in ERRORS:
            print(f"    - {e}")
    else:
        print("    (none)")

    print("\n  WRONG OUTPUTS (things that ran but returned bad results):")
    if WRONG_OUTPUTS:
        for w in WRONG_OUTPUTS:
            print(f"    - {w}")
    else:
        print("    (none)")

    print("\n  MISSING PIECES (not implemented or not loaded):")
    if MISSING:
        for m in MISSING:
            print(f"    - {m}")
    else:
        print("    (none)")

    print(f"\n  Totals: {len(ERRORS)} errors, {len(WRONG_OUTPUTS)} wrong outputs, {len(MISSING)} missing")

    # Verdict
    total_issues = len(ERRORS) + len(WRONG_OUTPUTS) + len(MISSING)
    if total_issues == 0:
        print("\n  VERDICT: ALL CLEAR")
    else:
        print(f"\n  VERDICT: {total_issues} issues found")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    banner("RESEARCHIT E2E PIPELINE AUDIT")
    print("  Warming up BGE-M3 + services...")
    embed_svc.encode_query("warmup")
    await turso_svc.fetch_metadata_batch(["1706.03762"])
    print("  Ready.\n")

    # Step 1: Search
    await step1_search()

    # Step 2: Profiles + Clustering
    lt_vec, st_vec = await step2_profiles()

    # Step 3: Recommendation feed
    valid_ids, emb_matrix, meta_list = await step3_recommendation_feed(lt_vec, st_vec)

    # Step 4: Reranker
    await step4_reranker(valid_ids, emb_matrix, meta_list, lt_vec, st_vec)

    # Step 5: MMR Diversity
    await step5_diversity(valid_ids, emb_matrix, lt_vec)

    # Step 6: Gap analysis
    step6_gap_analysis()

    banner("AUDIT COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
