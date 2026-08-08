"""
Comprehensive benchmark & diagnostics:
  1. Collection config (BQ, HNSW, quantization)
  2. Latency benchmarks for every pipeline stage
  3. Latest paper in the collection
"""
import asyncio
import time
import uuid
import re
import numpy as np
from fastapi.testclient import TestClient
from bs4 import BeautifulSoup

from app.main import app
from app import qdrant_svc, arxiv_svc, db, config
from app.recommend import profiles
from app.recommend.clustering import compute_clusters
from app.recommend.reranker import rerank_candidates
from app.recommend.diversity import mmr_rerank, inject_exploration


def _timer():
    """Simple context-manager-like timer."""
    class T:
        def __init__(self): self.start = time.perf_counter()
        def elapsed_ms(self): return (time.perf_counter() - self.start) * 1000
    return T()


def run():
    loop = asyncio.get_event_loop()
    client_api = TestClient(app)

    # =================================================================
    # SECTION 1: QDRANT COLLECTION CONFIG (BQ, HNSW, Quantization)
    # =================================================================
    print("=" * 70)
    print("SECTION 1: QDRANT COLLECTION CONFIG")
    print("=" * 70)

    from qdrant_client import QdrantClient
    qclient = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        timeout=15,
        check_compatibility=False,
    )

    info = qclient.get_collection(config.QDRANT_COLLECTION)
    print(f"\n  Collection:      {config.QDRANT_COLLECTION}")
    print(f"  Points count:    {info.points_count:,}")
    print(f"  Status:          {info.status}")

    # Vector config
    vc = info.config.params.vectors
    if hasattr(vc, 'size'):
        print(f"  Vector dim:      {vc.size}")
        print(f"  Distance:        {vc.distance}")
    elif isinstance(vc, dict):
        for name, v in vc.items():
            print(f"  Vector '{name}': dim={v.size}, distance={v.distance}")
    else:
        print(f"  Vectors config:  {vc}")

    # HNSW config
    hnsw = info.config.hnsw_config
    print(f"\n  HNSW m:          {hnsw.m}")
    print(f"  HNSW ef_construct: {hnsw.ef_construct}")
    print(f"  HNSW on_disk:    {hnsw.on_disk}")

    # Quantization config
    quant = info.config.quantization_config
    if quant is not None:
        print(f"\n  Quantization:    YES")
        if hasattr(quant, 'binary'):
            bq = quant.binary
            print(f"  Type:            Binary Quantization (BQ)")
            print(f"  Always RAM:      {bq.always_ram if hasattr(bq, 'always_ram') else 'N/A'}")
        elif hasattr(quant, 'scalar'):
            sq = quant.scalar
            print(f"  Type:            Scalar Quantization")
            print(f"  Scalar type:     {sq.type}")
            print(f"  Quantile:        {sq.quantile}")
            print(f"  Always RAM:      {sq.always_ram}")
        elif hasattr(quant, 'product'):
            pq = quant.product
            print(f"  Type:            Product Quantization (PQ)")
            print(f"  Compression:     {pq.compression}")
            print(f"  Always RAM:      {pq.always_ram}")
        else:
            print(f"  Type:            {quant}")
    else:
        print(f"\n  Quantization:    NONE (full float32 vectors)")

    # Optimizer config
    opt = info.config.optimizer_config
    print(f"\n  Indexing threshold: {opt.indexing_threshold}")
    print(f"  Memmap threshold:  {opt.memmap_threshold}")

    # =================================================================
    # SECTION 2: LATENCY BENCHMARKS
    # =================================================================
    print("\n\n" + "=" * 70)
    print("SECTION 2: LATENCY BENCHMARKS (per pipeline stage)")
    print("=" * 70)

    # Setup: save papers to create a realistic user
    user_id = f"bench-{uuid.uuid4().hex[:8]}"
    cookies = {"arxiv_user_id": user_id}

    # Search and save papers
    res = client_api.get("/search?q=Machine+Learning", cookies=cookies)
    soup = BeautifulSoup(res.text, "html.parser")
    ml_ids = []
    for btn in soup.find_all("button", attrs={"hx-post": re.compile(r"/api/papers/.+/save")}):
        m = re.search(r"/api/papers/([^/]+)/save", btn["hx-post"])
        if m: ml_ids.append(m.group(1))
    ml_ids = ml_ids[:4]

    res2 = client_api.get("/search?q=Neural+Networks", cookies=cookies)
    soup2 = BeautifulSoup(res2.text, "html.parser")
    nn_ids = []
    for btn in soup2.find_all("button", attrs={"hx-post": re.compile(r"/api/papers/.+/save")}):
        m = re.search(r"/api/papers/([^/]+)/save", btn["hx-post"])
        if m: nn_ids.append(m.group(1))
    nn_ids = nn_ids[:3]

    all_saved = ml_ids + nn_ids
    for pid in all_saved:
        client_api.post(f"/api/papers/{pid}/save", cookies=cookies)

    # Force EWMA inline
    async def setup():
        for pid in all_saved:
            vecs = await qdrant_svc.get_paper_vectors([pid])
            if pid in vecs:
                emb = np.array(vecs[pid], dtype=np.float32)
                await profiles.update_on_save(user_id, emb)
    loop.run_until_complete(setup())

    print(f"\n  Setup: saved {len(all_saved)} papers, EWMA profiles computed")

    # --- Benchmark each stage ---
    timings = {}

    # Stage 1: Fetch vectors from Qdrant
    async def bench_fetch_vectors():
        t = _timer()
        vecs = await qdrant_svc.get_paper_vectors(all_saved)
        elapsed = t.elapsed_ms()
        return vecs, elapsed
    vecs, t1 = loop.run_until_complete(bench_fetch_vectors())
    timings["1. Fetch vectors from Qdrant"] = t1
    print(f"\n  1. Fetch {len(all_saved)} vectors from Qdrant:   {t1:.1f} ms")

    # Stage 2: Ward clustering
    aligned_ids = [pid for pid in all_saved if pid in vecs]
    aligned_embs = np.array([vecs[pid] for pid in aligned_ids], dtype=np.float32)

    t = _timer()
    clusters = compute_clusters(aligned_ids, aligned_embs)
    t2 = t.elapsed_ms()
    timings["2. Ward clustering"] = t2
    print(f"  2. Ward clustering ({len(aligned_ids)} papers -> {len(clusters)} clusters):  {t2:.2f} ms")

    # Stage 3: Multi-interest retrieval (Qdrant prefetch + RRF)
    async def bench_retrieval():
        interest_vectors = []
        limits = [40, 30, 25]
        for i, c in enumerate(clusters):
            lim = limits[i] if i < len(limits) else 15
            interest_vectors.append((c.medoid_embedding.tolist(), lim))

        lt = await profiles.load_profile(user_id, "long_term")
        st = await profiles.load_profile(user_id, "short_term")
        st_list = st.tolist() if st is not None else None
        seen = set(all_saved)

        t = _timer()
        candidates = await qdrant_svc.multi_interest_search(
            interest_vectors=interest_vectors,
            short_term_vector=st_list,
            exclude_ids=seen,
            total_limit=100,
        )
        elapsed = t.elapsed_ms()
        return candidates, lt, st, elapsed
    candidates, lt_vec, st_vec, t3 = loop.run_until_complete(bench_retrieval())
    timings["3. Prefetch + RRF retrieval"] = t3
    print(f"  3. Prefetch + RRF ({len(clusters)} clusters + session):  {t3:.1f} ms  ({len(candidates)} candidates)")

    # Stage 4: Fetch candidate vectors + metadata for re-ranking
    async def bench_cand_fetch():
        t = _timer()
        cand_vecs = await qdrant_svc.get_paper_vectors(candidates[:50])
        cand_meta = await arxiv_svc.fetch_metadata_batch(candidates[:50])
        elapsed = t.elapsed_ms()
        return cand_vecs, cand_meta, elapsed
    cand_vecs, cand_meta, t4 = loop.run_until_complete(bench_cand_fetch())
    timings["4. Fetch candidate vectors+meta"] = t4
    print(f"  4. Fetch candidate vectors + metadata:  {t4:.1f} ms  ({len(cand_vecs)} vectors, {len(cand_meta)} metadata)")

    # Stage 5: Heuristic re-ranking
    valid_ids = [cid for cid in candidates if cid in cand_vecs and cid in cand_meta]
    valid_embs = np.array([cand_vecs[cid] for cid in valid_ids], dtype=np.float32)
    valid_meta = [cand_meta[cid] for cid in valid_ids]

    t = _timer()
    reranked_ids, reranked_scores, reranked_embs = rerank_candidates(
        candidate_ids=valid_ids,
        candidate_embeddings=valid_embs,
        candidate_metadata=valid_meta,
        long_term_vec=lt_vec,
        short_term_vec=st_vec,
    )
    t5 = t.elapsed_ms()
    timings["5. Heuristic re-ranking"] = t5
    print(f"  5. Heuristic re-ranking ({len(valid_ids)} candidates):  {t5:.2f} ms")

    # Stage 6: MMR diversity
    query_vec = lt_vec if lt_vec is not None else aligned_embs.mean(axis=0)
    t = _timer()
    mmr_selected = mmr_rerank(
        query_embedding=query_vec,
        candidate_embeddings=reranked_embs,
        candidate_ids=reranked_ids,
        scores=reranked_scores,
        lambda_param=0.6,
        top_k=10,
    )
    t6 = t.elapsed_ms()
    timings["6. MMR diversity selection"] = t6
    print(f"  6. MMR diversity selection (top 10):  {t6:.2f} ms")

    # Stage 7: Exploration injection
    t = _timer()
    final = inject_exploration(mmr_selected, reranked_ids, n_explore=2)
    t7 = t.elapsed_ms()
    timings["7. Exploration injection"] = t7
    print(f"  7. Exploration injection:  {t7:.3f} ms")

    # Stage 8: Template rendering (end-to-end HTTP)
    t = _timer()
    rec_res = client_api.get("/api/recommendations", cookies=cookies)
    t8 = t.elapsed_ms()
    timings["8. Full HTTP request (/api/recommendations)"] = t8
    print(f"\n  8. FULL HTTP /api/recommendations:  {t8:.1f} ms  (status={rec_res.status_code})")

    # Totals
    compute_total = t2 + t5 + t6 + t7
    network_total = t1 + t3 + t4
    print(f"\n  --- TOTALS ---")
    print(f"  Pure compute (clustering + rerank + MMR + explore):  {compute_total:.2f} ms")
    print(f"  Network I/O (Qdrant + arXiv metadata):               {network_total:.1f} ms")
    print(f"  Full end-to-end HTTP request:                        {t8:.1f} ms")

    # =================================================================
    # SECTION 3: HOW THE PIPELINE WORKS (Step by step)
    # =================================================================
    print("\n\n" + "=" * 70)
    print("SECTION 3: HOW THE PIPELINE WORKS")
    print("=" * 70)
    print("""
  User saves papers -> events.py fires background EWMA update
                       (fetches paper's 1024-dim BGE-M3 vector from Qdrant,
                        blends into user's long-term/short-term profile)

  User loads home page -> GET /api/recommendations fires

  Step 1: Load user_state (in-memory deque of saved/dismissed IDs)
  Step 2: Check tier eligibility:
          >= 5 saves? -> Tier 1 (clustering + RRF)
          >= 3 saves? -> Tier 2 (EWMA vector search)
          >= 1 save?  -> Tier 3 (Qdrant BEST_SCORE with raw IDs)

  TIER 1 PIPELINE:
    a) Fetch BGE-M3 embeddings for all saved papers from Qdrant
    b) Run Ward hierarchical clustering on those embeddings
       -> Finds 1-7 interest groups automatically (adaptive gap method)
       -> Each cluster's "medoid" = the real paper closest to cluster center
    c) Send medoid embeddings as parallel ANN queries to Qdrant
       (Prefetch API: single network call, server runs them in parallel)
    d) Qdrant fuses results via Reciprocal Rank Fusion (RRF, k=60)
       -> Papers appearing in multiple cluster results get boosted
    e) Fetch candidate embeddings + arXiv metadata
    f) Heuristic re-ranking: score = 0.45*cos_sim_LT + 0.25*cos_sim_ST
                                    + 0.20*recency + 0.10*rrf_rank
    g) MMR diversity: greedily select top-10 maximizing
       lambda*relevance - (1-lambda)*max_similarity_to_already_selected
    h) Inject 1-2 random "exploration" papers from the candidate pool
    i) Fetch arXiv metadata, render Jinja2 HTML, return via HTMX
""")

    # =================================================================
    # SECTION 4: LATEST PAPER IN QDRANT
    # =================================================================
    print("=" * 70)
    print("SECTION 4: LATEST PAPER IN QDRANT")
    print("=" * 70)

    # Strategy: scroll with ordering by a date field, or sample recent IDs
    # Qdrant doesn't have a "sort by payload" for scroll, so we'll sample
    # papers with high point IDs (usually later additions) and check dates
    print("\n  Sampling papers with highest point IDs (latest additions)...")

    # Get collection info to find point ID range
    try:
        # Scroll from the end (highest IDs)
        # Use reverse scroll by getting points near the max count
        max_id = info.points_count
        sample_ids = list(range(max(1, max_id - 20), max_id + 1))

        points = qclient.retrieve(
            collection_name=config.QDRANT_COLLECTION,
            ids=sample_ids,
            with_payload=True,
            with_vectors=False,
        )

        if points:
            # Sort by published date
            dated_papers = []
            for p in points:
                pub = p.payload.get("published", "")
                arxiv_id = p.payload.get("arxiv_id", "?")
                title = p.payload.get("title", "?")
                cats = p.payload.get("categories", p.payload.get("category", "?"))
                dated_papers.append((pub, arxiv_id, title, cats, p.id))

            dated_papers.sort(key=lambda x: x[0], reverse=True)

            print(f"\n  Top 10 most recent papers (by published date) from high-ID sample:\n")
            for i, (pub, aid, title, cats, pid) in enumerate(dated_papers[:10], 1):
                t_short = (title[:65] + "...") if len(str(title)) > 65 else title
                print(f"    {i:2d}. [{pub}] {aid}")
                print(f"        {t_short}")
                print(f"        Categories: {cats}  (Qdrant ID: {pid})")
                print()

            latest = dated_papers[0]
            print(f"  LATEST PAPER: {latest[1]} published {latest[0]}")
        else:
            print("  Could not retrieve high-ID points")

    except Exception as e:
        print(f"  Error sampling latest papers: {e}")

    # Also try searching for very recent papers by scrolling with a filter
    print("\n  Also checking for 2025+ papers across the collection...")
    try:
        from qdrant_client.models import Filter, FieldCondition, Range
        pts_2025, _ = qclient.scroll(
            collection_name=config.QDRANT_COLLECTION,
            scroll_filter=Filter(must=[
                FieldCondition(key="year", range=Range(gte=2025))
            ]),
            limit=10,
            with_payload=True,
            with_vectors=False,
        )
        if pts_2025:
            latest_2025 = []
            for p in pts_2025:
                pub = p.payload.get("published", "")
                aid = p.payload.get("arxiv_id", "?")
                title = p.payload.get("title", "?")
                latest_2025.append((pub, aid, title))
            latest_2025.sort(key=lambda x: x[0], reverse=True)

            print(f"  Found {len(pts_2025)} papers from 2025+:\n")
            for pub, aid, title in latest_2025[:5]:
                t_short = (str(title)[:65] + "...") if len(str(title)) > 65 else title
                print(f"    [{pub}] {aid}: {t_short}")
        else:
            print("  No papers with year >= 2025 found (or 'year' field not indexed)")
            # Try without year filter, just get random sample
            print("  Trying broader search...")

    except Exception as e:
        print(f"  Year filter failed ({e}), trying published date range...")
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            # Just scroll a few and show dates
            pts, _ = qclient.scroll(
                collection_name=config.QDRANT_COLLECTION,
                limit=5,
                with_payload=True,
                with_vectors=False,
            )
            if pts:
                for p in pts:
                    print(f"    Sample: {p.payload.get('arxiv_id','?')} "
                          f"published={p.payload.get('published','?')}")
        except Exception as e2:
            print(f"  Could not query: {e2}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    run()
