# RFC: Multi-interest recommendation architecture for ArXiv paper discovery

**The multi-vector RRF approach (Path B) is directionally correct and validated by industry practice at Twitter, Pinterest, and Alibaba — but the proposed implementation over-engineers the storage model while under-investing in the ranking layer.** The optimal Phase 2 architecture combines PinnerSage-style dynamic clustering (not fixed subject vectors) with Qdrant's native prefetch+RRF fusion, a LightGBM re-ranker (~3ms on CPU), and MMR diversity enforcement. This report synthesizes findings from Twitter's open-sourced algorithm, Pinterest's PinnerSage, TikTok's Monolith, YouTube's DNN architecture, and Qdrant's API internals to justify a specific architectural recommendation.

The core tension — how to serve users with distinct interests (e.g., computer vision, reinforcement learning, and LLM alignment) without the feed collapsing into a single dominant topic — is one of the most studied problems in industrial recommendation systems. Every major platform has converged on some variant of multi-interest retrieval, but none use hand-curated subject vectors. The interest structure should emerge from user behavior, not be predefined.

---

## 1. Path A vs Path B: neither is quite right

### Path A limitations are real but well-understood

Qdrant's Recommend API with `AVERAGE_VECTOR` strategy computes a single query vector via the formula `2 × avg(positives) − avg(negatives)`, then performs standard HNSW traversal. This is computationally identical to a normal nearest-neighbor search — fast and cheap, but fatally prone to the centroid-in-nowhere problem. PinnerSage's authors demonstrated this vividly: a user interested in painting, shoes, and sci-fi produces a centroid landing in the "energy-boosting breakfast" region of embedding space.

The `BEST_SCORE` strategy avoids this by evaluating each candidate against every positive/negative example independently, selecting the best match via a sigmoid scoring function. This produces **genuinely diverse results** that grow more diverse as examples increase. However, latency scales linearly with example count — with 20 positives and 50 negatives, every HNSW traversal step computes 70 distance calculations. For a cloud-hosted Qdrant instance, this creates meaningful cost and latency pressure.

Path A's fundamental constraint is architectural: no matter which strategy is chosen, the Recommend API treats all positive examples as equally important and offers **no mechanism to weight recent interactions over older ones**, no temporal decay, and no way to control the accuracy-diversity tradeoff.

### Path B is validated but misspecifies the interest model

The multi-vector retrieval pattern — maintaining multiple user embeddings and querying each independently — is not an anti-pattern. It is the **dominant approach in industrial multi-interest recommendation** as of 2025:

- **Twitter/X** uses SimClusters (sparse vectors over 145,000 communities), TwHIN multi-modal mixture embeddings, and kNN-Embed — all explicitly multi-vector. kNN-Embed achieved a **534% relative improvement in Recall@10** vs. single-embedding retrieval on the Twitter-Follow dataset.
- **Pinterest's PinnerSage** generates multiple medoid-based embeddings per user via Ward hierarchical clustering, producing **3–5 clusters for light users and 75–100 for heavy users**. A/B tests showed 7% engagement lift on Homefeed and 20% on Shopping.
- **Alibaba's MIND** uses capsule network dynamic routing to extract K interest vectors (optimal K=4–7), deployed on Mobile Tmall handling major production traffic.
- **ComiRec** (KDD 2020) extends this with controllable diversity via a tunable λ parameter, finding K=4 optimal for e-commerce datasets.

However, Path B's specification of **fixed subject-specific vectors** (CV, LLMs, RL) is where it diverges from best practice. Every production system cited above derives interest clusters from user behavior, not predetermined categories. Fixed categories create three problems: (1) they can't adapt as new research areas emerge (e.g., a sudden interest in mechanistic interpretability), (2) they force a taxonomy decision that may not match individual user interest boundaries, and (3) they require manual maintenance as the field evolves.

### RRF is a sound fusion method for this use case

RRF (formula: `score(d) = Σ 1/(k + r(d))`, k=60) was originally designed for information retrieval but has become the **standard fusion method** across vector databases. Elasticsearch, Azure AI Search, MongoDB, OpenSearch, and Qdrant all implement it natively. Its key advantage: it operates on rank positions rather than raw scores, requiring no score normalization when combining results from different interest vectors that may have incomparable similarity distributions.

Qdrant's prefetch+fusion mechanism (v1.10+) executes all interest queries in a **single API call**, eliminating the multiple network round-trip concern:

```python
client.query_points(
    collection_name="arxiv_papers",
    prefetch=[
        models.Prefetch(query=interest_vector_1, limit=30),
        models.Prefetch(query=interest_vector_2, limit=30),
        models.Prefetch(query=interest_vector_3, limit=30),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=50
)
```

This architecture — multiple ANN queries fused via RRF within a single request — is computationally reasonable. Each HNSW query on 1.6M vectors takes ~5–15ms; with server-side parallelism, the wall-clock time for 3–4 prefetch queries is roughly equal to one query plus modest overhead.

---

## 2. What the platforms actually do about interest collapse

### Twitter solved it architecturally, not algorithmically

Twitter's recommendation pipeline generates ~1,500 candidate tweets from ~500 million daily tweets, split roughly 50/50 between in-network (accounts you follow) and out-of-network (discovery). The critical insight is that **diversity is embedded in the candidate generation architecture itself** through multiple heterogeneous retrieval channels:

| Source | Method | Purpose |
|--------|--------|---------|
| SimClusters | Sparse 145K-dim community vectors | Multi-interest out-of-network discovery |
| TwHIN | Knowledge graph embeddings (TransE) | Cross-entity relationship modeling |
| RealGraph | Follow graph traversal | In-network relevance |
| GraphJet | Real-time interaction graph | Trending/recent engagement |

SimClusters represents each user as a **sparse vector over 145,000 overlapping communities** detected via Sparse Binary Factorization on the follow graph. Because users belong to multiple communities simultaneously, retrieval naturally covers multiple interests. Tweet embeddings are computed in real-time by aggregating engagement signals from users who interacted with the tweet, projected into community space.

Twitter's Heavy Ranker (a parallel MaskNet with ~48 million parameters) scores candidates on 10 simultaneous objectives. The scoring formula heavily penalizes negative signals: `P(report) × −369` and `P(negative_reaction) × −74`, while positive engagement weights are modest (`P(favorite) × 0.5`). Post-ranking heuristics enforce author diversity (score halved for consecutive same-author tweets) and maintain the in-network/out-of-network ratio.

### TikTok relies on real-time adaptation rather than multi-vectors

TikTok takes a fundamentally different approach. Rather than maintaining multiple static user representations, ByteDance's **Monolith** system enables near-real-time model updates through a streaming architecture that closes the feedback loop within minutes. Key innovations include collisionless embedding tables (Cuckoo hashing, eliminating embedding collisions), streaming online training via Flink-based Online Joiner, and minute-level parameter synchronization between training and inference servers.

TikTok enforces diversity through **Determinantal Point Processes** (controlling intra-list similarity), creator rotation rules, and explicitly allocating **15–25% of recommendations** to content outside the user's expressed preferences. Thompson Sampling and Epsilon-Greedy strategies inject calculated exploration. This works because TikTok captures rich implicit signals (watch time, loop count, scroll speed, completion percentage) that enable rapid interest detection.

### Pinterest provides the clearest blueprint for this system

PinnerSage is the most directly applicable model for the ArXiv recommender because it operates under similar constraints: pre-computed item embeddings in a fixed space (PinSage GCN embeddings, analogous to BGE-M3), no joint user-item training, and a need to support users with 3–100 distinct interests.

PinnerSage's design choices are instructive:

- **Ward hierarchical clustering** on the user's interacted item embeddings, with a threshold parameter α controlling merge stopping — this automatically determines K per user
- **Medoid representation** (the actual item closest to cluster center), not centroids — prevents topic drift and enables cache sharing across users
- **At serving time, 3 medoids are sampled** proportional to importance scores for ANN retrieval — this controls query cost while maintaining coverage
- **Dual temporal architecture**: daily batch job processes 60–90 days of history; online component captures same-day interactions; end-of-day reconciliation merges both

The medoid approach is particularly elegant for this system: each interest cluster is represented by an actual ArXiv paper's embedding, stored as a simple paper ID reference rather than a separate vector.

---

## 3. Temporal dynamics: EWMA beats rolling windows

### The math favors exponential decay

EWMA updates user embeddings with the formula `embedding_t = α × item_embedding_t + (1−α) × embedding_{t-1}`, where α controls responsiveness. The operation is **O(d) per interaction** with O(d) storage per user — for BGE-M3's 1024 dimensions, that's ~4KB per user embedding.

| α value | Effective window | Behavior | Use case |
|---------|-----------------|----------|----------|
| 0.05 | ~40 interactions | Very stable, slow drift | Core long-term interests |
| 0.10–0.15 | ~13–20 interactions | Balanced | General user profile |
| 0.30–0.50 | ~3–5 interactions | Highly responsive | Session-level interests |

Rolling windows (recompute from last N interactions) require storing N item embeddings per user — at N=50 with 1024-dimensional BGE-M3 vectors, that's **~200KB per user** vs. EWMA's 4KB. More importantly, rolling windows create a hard boundary: interaction N+1 is completely forgotten regardless of how significant it was. Koren's seminal work on temporal dynamics in collaborative filtering (Netflix Prize, CACM 2010) explicitly warns against this: "Classical time-window approaches cannot work, as they lose too many signals when discarding data instances."

### The recommended temporal architecture

Adopt Spotify's two-component pattern, validated in their production system serving 600M+ users:

- **Long-term interest embedding**: EWMA with α=0.10, capturing enduring research interests across ~20 interactions
- **Short-term session embedding**: EWMA with α=0.40, capturing current reading session context across ~3–5 interactions

Both embeddings update on each save/dismiss action in a FastAPI background task. The long-term embedding feeds into PinnerSage-style clustering (recomputed daily or on-demand when the embedding has drifted significantly). The short-term embedding serves as an additional prefetch query to boost recency-relevant candidates.

This is strictly superior to storing rolling windows: less memory, simpler code (no circular buffer management), natural handling of irregular interaction frequencies, and no signal loss from hard cutoffs.

---

## 4. The re-ranking layer is the highest-leverage investment

### LightGBM is the clear winner for zero-GPU constraints

Cross-encoder reranking (BGE-reranker-v2-m3) requires **~800ms for 100 candidates on CPU** — completely impractical for interactive recommendations. ColBERT-style late interaction models offer ~50–100ms for 100 candidates but require significant infrastructure for precomputing token-level embeddings across 1.6M papers.

LightGBM with a lambdarank objective scores **500 candidates in 2–5ms on a single CPU core**. This is not a compromise — tree-based re-rankers are used in production at Airbnb (which called their GBDT ranker "one of the largest step improvements in home bookings in Airbnb history"), LinkedIn, and Delivery Hero. The RecSys Challenge 2024 winning approach used LightGBM Ranker, outperforming neural methods on tabular ranking data.

| Approach | 100 candidates (CPU) | 500 candidates (CPU) | Feasibility |
|----------|----------------------|----------------------|-------------|
| LightGBM | ~1–2ms | ~3–5ms | ✅ Excellent |
| XGBoost | ~2–3ms | ~5–8ms | ✅ Good |
| BGE-reranker-v2-m3 | ~800ms | ~4,000ms | ❌ Impractical |
| ColBERT (PLAID) | ~50–100ms | ~250–500ms | ⚠️ Marginal |

Typical features for a LightGBM re-ranker in this context would include: cosine similarity between user embedding and paper embedding, paper recency (days since publication), paper citation velocity, category overlap with user history, author overlap with previously saved papers, abstract length, and engagement signals from similar users (once collaborative data accumulates).

### MMR provides practical diversity enforcement

Maximal Marginal Relevance (formula: `MMR = argmax[λ × Sim(d_i, Q) − (1−λ) × max Sim(d_i, d_j)]`) is the recommended starting point. For selecting 20 papers from 200 re-ranked candidates, MMR completes in **<1ms** with precomputed embeddings. Setting λ=0.6 provides a good balance between relevance and diversity for academic paper discovery.

DPP (Determinantal Point Processes) provides theoretically superior global diversity — YouTube's A/B tests showed increased user satisfaction vs. both no-diversity and MMR baselines. However, DPP's implementation complexity (greedy MAP approximation with incremental Cholesky updates) is significantly higher, and the practical difference at k=20 is modest. Graduate to DPP if MMR proves insufficient.

Category-based quotas (e.g., "ensure at least 2 papers from each of the user's top 3 ArXiv categories") can serve as a simple, interpretable diversity floor alongside MMR, despite Airbnb's negative experience with rigid quota-based diversification in their domain.

---

## 5. Recommended Phase 2 architecture

### The proposed design

```
User Action (save/dismiss)
    │
    ▼
┌─────────────────────────────────────────────┐
│  Profile Update Service (FastAPI background) │
│  • EWMA update: long-term (α=0.10)          │
│  • EWMA update: short-term (α=0.40)         │
│  • Store negative centroid from dismissals   │
│  • Trigger re-clustering if drift > θ        │
└──────────────┬──────────────────────────────┘
               │
    ▼ (daily batch or on-demand)
┌─────────────────────────────────────────────┐
│  Interest Clustering (Ward's method)         │
│  • Cluster saved paper embeddings            │
│  • K determined automatically (typically 3-5)│
│  • Store medoid paper IDs per cluster        │
│  • Compute importance weights per cluster    │
└──────────────┬──────────────────────────────┘
               │
    ▼ (on feed request)
┌─────────────────────────────────────────────┐
│  Candidate Retrieval (Qdrant prefetch+RRF)   │
│  • Prefetch 1: top medoid vector (limit=40)  │
│  • Prefetch 2: 2nd medoid vector (limit=30)  │
│  • Prefetch 3: 3rd medoid vector (limit=25)  │
│  • Prefetch 4: short-term embedding (limit=25)│
│  • Fusion: RRF (k=60)                        │
│  • Filter: exclude dismissed paper IDs       │
│  • Output: ~100 candidates                   │
│  • Latency: ~15-25ms (single API call)       │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  Re-Ranking (LightGBM lambdarank)            │
│  • Features: user-paper similarity,          │
│    paper recency, citation velocity,         │
│    category match, author overlap            │
│  • Score 100 candidates                      │
│  • Latency: ~1-2ms                           │
│  • Output: scored + sorted candidates        │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  Diversity Enforcement (MMR, λ=0.6)          │
│  • Select top-k from scored candidates       │
│  • Penalize similarity to already-selected   │
│  • Inject 1-2 exploration papers (random     │
│    high-quality papers from adjacent topics)  │
│  • Latency: <1ms                             │
│  • Output: final feed (20-30 papers)         │
└─────────────────────────────────────────────┘
```

### Why not Two-Tower or graph-based alternatives

**Two-Tower models** (separate user and item encoder networks producing embeddings for dot-product comparison) are the industry standard at scale but require GPU training infrastructure, large volumes of interaction data for learning, and ongoing model retraining. With a small initial user base and zero-GPU constraint, Two-Tower is premature — it solves a scale problem this system doesn't yet have.

**Graph-based approaches** (like Pinterest's Pixie random walks on a bipartite user-item graph) require dense interaction graphs to be effective. Pixie operates on a graph with 3B+ nodes and 17B+ edges. A small user base on 1.6M papers will produce an extremely sparse graph where random walks yield poor recommendations. This approach becomes viable only after accumulating substantial collaborative signal.

The proposed architecture is designed for **graceful evolution**: the LightGBM re-ranker can incorporate collaborative features (what did similar users save?) as the user base grows, and the retrieval layer can eventually be augmented with a Two-Tower candidate generator when sufficient training data exists, without disrupting the existing pipeline.

### Optimal number of user embeddings

Based on converging evidence from PinnerSage, MIND, and ComiRec, the optimal configuration for this system is:

- **3–4 interest medoids** from Ward clustering on saved papers (data-driven K, not fixed)
- **1 short-term session embedding** (EWMA α=0.40)
- **1 negative centroid** from dismissed papers (used as Qdrant negative example or filter)
- **Total: 4–6 query vectors per feed request**, executed as prefetch queries in a single Qdrant API call

This aligns with MIND's finding that K=4–7 is optimal for moderately diverse interests, while keeping query costs manageable. As users accumulate more interactions, the clustering will naturally produce more clusters — PinnerSage demonstrated that heavy users may require up to 75–100 clusters, but sampling 3 medoids at serving time keeps latency constant.

### What to build first

The implementation should be phased to deliver value incrementally:

**Phase 2a (immediate):** Replace the simple Recommend API call with Qdrant's `BEST_SCORE` strategy using all saved papers as positives and dismissed papers as negatives. This is a one-line change that provides meaningfully better diversity. Add a dismissed-paper ID filter to exclude already-seen content. Implement EWMA user embedding updates.

**Phase 2b (1–2 weeks):** Implement Ward clustering on saved paper embeddings. Switch to prefetch+RRF retrieval with dynamic interest medoids. Add the short-term session embedding as an additional prefetch query. This is the core multi-interest architecture.

**Phase 2c (2–4 weeks):** Train a LightGBM re-ranker on accumulated save/dismiss data. Add MMR diversity enforcement. Implement exploration injection (2–3 randomly sampled high-quality recent papers from adjacent ArXiv categories per feed).

**Phase 2d (future):** When sufficient interaction data exists, add collaborative filtering features to LightGBM (similar-user signals). Evaluate DPP if MMR diversity proves insufficient. Consider a lightweight Two-Tower model if/when a GPU budget becomes available.

---

## Conclusion

The proposed system's instinct toward multi-vector retrieval with RRF fusion is validated by converging evidence from Twitter (kNN-Embed, SimClusters), Pinterest (PinnerSage), and Alibaba (MIND, ComiRec). **The critical correction is to derive interest clusters from user behavior rather than predefined subject categories.** Ward hierarchical clustering on saved paper embeddings — producing 3–5 medoid-based interest vectors for typical users — is simpler, more adaptive, and better validated than maintaining fixed CV/LLM/RL vectors.

The highest-leverage Phase 2 investment is the **LightGBM re-ranking layer**, not retrieval sophistication. At 2–5ms inference on CPU for hundreds of candidates, it provides learned ranking quality comparable to neural approaches without any GPU requirement. Cross-encoder reranking is definitively impractical at this scale without GPU (~800ms for 100 candidates). The combination of multi-interest retrieval, lightweight learned re-ranking, and MMR diversity enforcement creates a system that can evolve gracefully from dozens to thousands of users while keeping total feed generation latency under **30ms per request** — well within the budget for a responsive FastAPI application.

---

## Addendum: Corrections from Doc 06 (Deep Research Verdict)

> **Added April 2025.** Doc 06 (`06-Deep-Research-Verdict.md`) performed a comprehensive review
> of this RFC against the original PinnerSage paper, empirical literature, and production
> systems. It identified **4 concrete architectural faults** in this document. Three have been
> fixed in code; one is pending. This addendum preserves the original text above for reference
> while documenting the corrections.

### Fault 1: RRF is wrong for multi-interest fusion (§1, §5) — PENDING

**What this doc says (§1):** "RRF is a sound fusion method for this use case."

**What Doc 06 found:** RRF was designed to fuse *different retrievers answering the same query* (BM25 + vector + Condorcet). Using it to fuse *different interest queries for the same user* means papers near the centroid of ALL interests get boosted — the exact failure mode multi-interest models exist to prevent. PinnerSage itself does not use RRF; they sample 3 medoids proportional to importance and concatenate into the downstream ranker. Taobao ULIM, Pinterest Bucketized-ANN, Twitter kNN-Embed, and ComiRec all use quota-based allocation, not RRF.

**The correction:** Replace RRF with importance-weighted quota allocation:
```
slot_k = max(⌊F × w_k⌋, F_min=3)
```
Each cluster gets feed slots proportional to its importance, with a floor of 3 to protect minority interests.

**Note:** RRF *is* correct for the search bar (fusing dense + sparse for the *same* query). Only the recommendation pipeline needs quota.

**Status:** ⚠️ Code still uses RRF. Phase 4 planned — see `docs/phases/PHASE4-Recommendation-Pipeline-Fixes.md`.

---

### Fault 2: α_long = 0.10 is too aggressive (§3) — FIXED ✅

**What this doc says (§3):** "Long-term interest embedding: EWMA with α=0.10, capturing enduring research interests across ~20 interactions."

**What Doc 06 found:** PinnerSage (KDD 2020) tested λ=0.1 and **explicitly rejected it as too recent-biased**. Their optimal was λ=0.01. With α=0.10, a minority interest retains only 12% of its signal after 20 saves in a different topic (`0.9^20 ≈ 0.12`).

**The correction:** α_long changed from 0.10 → **0.03** (effective window ~66 interactions). At α=0.03, minority interests retain 54% after 20 saves (`0.97^20 ≈ 0.54`).

**Code change:** `app/recommend/profiles.py` — `ALPHA_LONG_TERM = 0.03`

---

### Fault 3: BGE-reranker-v2 in the hot path is infeasible (§4) — N/A (never built)

**What this doc says (§4):** Table shows BGE-reranker-v2-m3 at ~800ms for 100 candidates, marked "❌ Impractical." The text already warns against it. However, the Phase 2c plan at the end could be misread as suggesting a cross-encoder in the hot path.

**What Doc 06 found:** Confirms the rejection. If cross-encoder signal is needed, distill BGE-reranker-v2 offline into a TinyBERT-L2 student (FlashRank recipe) and use as a LightGBM feature on top-20 only. Never serve the full cross-encoder at request time.

**Status:** ✅ Not an issue — the heuristic scorer was built instead, with LightGBM planned as replacement.

---

### Fault 4: Ward needs L2 normalization (§2, §5) — FIXED ✅

**What this doc says (§2):** "Ward hierarchical clustering on the user's interacted item embeddings."

**What Doc 06 found:** Ward is mathematically Euclidean-only (Murtagh & Legendre, J. Classification 2014). BGE-M3 operates in cosine space. Without explicit L2 normalization, Euclidean Ward is silently wrong because vector magnitudes affect cluster assignments. On L2-normalized vectors, ‖a−b‖² = 2(1−cos(a,b)), so Euclidean Ward correctly gives cosine-based clustering.

**Code change:** `app/recommend/clustering.py` — explicit L2 normalization added before `pdist()`.

---

### Additional correction: Negative profile wiring — FIXED ✅

**What this doc says (§5):** "1 negative centroid from dismissed papers (used as Qdrant negative example or filter)."

**What Doc 06 found:** The negative profile was computed and stored but never read by the recommendation pipeline. YouTube (2023, Xia et al.) showed a 3× gain from using dislikes as both features and training labels. The recommended design is a three-layer negative system: session hard-filter + short-term penalty + long-term EWMA negative medoid.

**Code change:** `app/recommend/reranker.py` — added `cosine_sim_negative` as Feature 5 with 0.15 penalty weight. `app/routers/recommendations.py` — loads negative profile and passes to reranker.

---

### Correction summary

| Fault | This doc said | Doc 06 correction | Code status |
|---|---|---|---|
| RRF for rec fusion | Sound method | Wrong — use quota with F_min floor | ⚠️ Phase 4 |
| α_long = 0.10 | Balanced | Too aggressive — use 0.03 | ✅ Fixed |
| BGE-reranker in hot path | Impractical | Confirmed — distill offline only | ✅ N/A |
| Ward without L2-norm | Implicit | Must L2-normalize first | ✅ Fixed |
| Negative profile unused | Mentioned | Must wire into reranking | ✅ Fixed |