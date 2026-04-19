# Phase 2 — Multi-Interest Recommender Walkthrough

## What Was Built

A PinnerSage-style multi-interest recommendation engine that replaces Phase 1's raw-ID Qdrant queries with computed EWMA user profile embeddings, Ward hierarchical clustering for interest detection, heuristic re-ranking, and MMR diversity enforcement.

**The old pipeline (Phase 1):**
```
User saves papers → raw IDs → Qdrant BEST_SCORE → results
```

**The new pipeline (Phase 2):**
```
User saves papers
    ↓
EWMA profiles update (background, non-blocking)
    ↓
Ward clustering → K distinct interest medoids (auto K per user)
    ↓
Qdrant prefetch + RRF fusion (~15-25ms, single API call)
    ↓
Heuristic re-ranking of ~100 candidates (~1-2ms)
    ↓
MMR diversity selection → top 10-12 papers (<1ms)
    ↓
Exploration injection → 1-2 serendipitous papers
    ↓
Render HTML via HTMX
```

**Total pipeline latency: <30ms** (excluding metadata fetch if cold)

---

## Why This Architecture

This architecture was chosen after deep research documented in [03-MultiInterest-Recommender-Architecture.md](../research/03-MultiInterest-Recommender-Architecture.md). The key insights:

### The Interest Collapse Problem
A single average embedding for a user interested in both *NLP* and *computer vision* lands in meaningless embedding space — Pinterest called this the "energy-boosting breakfast" problem. PinnerSage (KDD 2020) solved it with multiple user vectors.

### Why EWMA Over Rolling Windows
Rolling windows (last 30 days) lose valuable historical signal abruptly. EWMA (Exponentially Weighted Moving Average) provides smooth decay:
- **Long-term (α=0.10):** Effective window ~20 interactions. Tracks enduring research interests.
- **Short-term (α=0.40):** Effective window ~3-5 interactions. Captures current session context.
- **Negative (α=0.15):** Tracks papers the user explicitly dislikes.

### Why Ward Over K-Means
K-Means requires pre-specifying K (number of clusters). Ward hierarchical clustering auto-determines K per user via a distance threshold — a user with 2 interests gets 2 clusters, a user with 5 gets 5. No hyperparameter tuning per user.

### Why LightGBM Over BGE-reranker
The older `Research-Recommender_Technical_Roadmap.md` suggested BGE-reranker-v2 at ~800ms for 100 candidates on CPU. LightGBM scores 500 candidates in 2-5ms. On Render Free Tier (CPU-only, 512MB RAM), this is the only viable option. Currently using a heuristic scorer with the same feature interface — drop-in LightGBM upgrade when training data accumulates.

---

## 3-Tier Cascading Fallback

The recommender degrades gracefully based on how much data the user has:

| User State | Tier | Strategy | Latency |
|---|---|---|---|
| ≥5 saves | **Tier 1** | Clustering → RRF → Rerank → MMR → Explore | ~25ms |
| 3-4 saves | **Tier 2** | EWMA long-term vector → ANN search | ~10ms |
| 1-2 saves | **Tier 3** | Qdrant BEST_SCORE (Phase 1 path) | ~15ms |
| 0 saves | Empty | "Save at least 1 paper..." | 0ms |

Each tier falls through to the next if it can't produce results.

---

## New Files Created

### `app/recommend/__init__.py`
Package init for the recommendation engine module.

### `app/recommend/profiles.py`
EWMA temporal embedding profiles:
- `ewma_update(current, new_embedding, alpha)` — core blending function
- `update_on_save(user_id, paper_embedding)` — updates both LT and ST profiles
- `update_on_dismiss(user_id, paper_embedding)` — updates negative profile
- `load_profile()` / `save_profile()` — SQLite persistence as binary numpy blobs (4KB each)

### `app/recommend/clustering.py`
Ward hierarchical clustering:
- `compute_clusters(paper_ids, embeddings)` → list of `InterestCluster`
- Each cluster: medoid paper ID, medoid embedding, member paper IDs, importance score
- Auto K (1-7 clusters), recency-weighted importance
- Falls back to single cluster if <5 saved papers

### `app/recommend/reranker.py`
Heuristic scorer (LightGBM-ready):
- `compute_features()` → 4 features per candidate: cosine_sim_LT, cosine_sim_ST, paper_age, rrf_position
- `heuristic_score()` → weighted sum: 45% relevance, 25% session, 20% recency, 10% rank
- `rerank_candidates()` → end-to-end: features → scores → sorted output

### `app/recommend/diversity.py`
MMR diversity + exploration:
- `mmr_rerank(query, candidates, scores, λ=0.6, top_k=20)` — greedy diverse selection
- `inject_exploration(selected, pool, n_explore=2)` — random serendipity injection

---

## Modified Files

### `app/db.py`
- Added `user_profiles` table — EWMA vectors as BLOBs with interaction counts
- Added `user_clusters` table — Ward clustering results (medoid IDs, importance, paper lists)
- Added 4 helper functions: `get_user_profile`, `upsert_user_profile`, `save_user_clusters`, `get_user_clusters`

### `app/qdrant_svc.py`
- Added `get_paper_vectors()` — fetch actual BGE-M3 embeddings from Qdrant (needed for EWMA)
- Added `search_by_vector()` — raw ANN search by embedding vector
- Added `multi_interest_search()` — prefetch + RRF fusion in a single API call
- Imported new Qdrant models: `Prefetch`, `FusionQuery`, `Fusion`

### `app/routers/events.py`
- Save handler now triggers background EWMA profile update (LT + ST) via `asyncio.create_task`
- Dismiss handler triggers background negative profile update
- Both are non-blocking — user response is sent before the update completes

### `app/routers/recommendations.py`
- Complete rewrite with 3-tier cascading fallback
- Tier 1: full 5-step pipeline (cluster → retrieve → rerank → MMR → explore)
- Tier 2: EWMA long-term single-vector search
- Tier 3: original BEST_SCORE (unchanged from Phase 1)

### `requirements.txt`
- Added `numpy>=1.24` — vector computations
- Added `scipy>=1.11` — Ward hierarchical clustering

---

## What Was NOT Changed

These files are intentionally untouched:
- `app/user_state.py` — still manages ID deques for the hot cache
- `app/routers/search.py` — search is a separate concern (see PHASE2-Hybrid-Search-Plan)
- `app/routers/saved.py` — saved papers page is unaffected
- All templates — no UI changes needed, same HTMX partials

---

## Test Coverage

| Test File | Tests | Description |
|---|---|---|
| `test_profiles.py` | 11 | EWMA math, convergence, normalisation, DB round-trips |
| `test_clustering.py` | 10 | Ward clustering, medoid validity, max clusters, DB persistence |
| `test_reranker_diversity.py` | 13 | Heuristic scoring, MMR diversity, exploration injection |
| Existing tests | 52 | Integration, events, saved page, qdrant_svc |
| **Total** | **86 passed** | 2 pre-existing live Qdrant failures (network-dependent) |

---

## Upgrade Path: Heuristic → LightGBM

The heuristic scorer in `reranker.py` is designed for a zero-data-required drop-in to LightGBM:

1. **When:** Interactions table has ≥500 save/dismiss rows
2. **How:** Train offline with `lgb.train(params={'objective': 'lambdarank'}, ...)`
3. **Where:** Save model to `models/reranker.lgb`, replace `heuristic_score()` with `model.predict(features)`
4. **Impact:** Same features, same interface — zero code changes in the router

---

## Key Design Decisions & Rationale

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| User profile | EWMA (3 vectors) | Rolling window | Smooth decay, no abrupt signal loss |
| Clustering | Ward hierarchical | Fixed K-Means | Auto-determines K per user |
| Re-ranking | Heuristic → LightGBM | BGE-reranker-v2 | 800ms → 2ms on CPU |
| Diversity | MMR (λ=0.6) | Random sampling | Principled relevance/diversity trade-off |
| Exploration | Random injection (2 papers) | None | Prevents filter bubbles |
| Multi-query | Qdrant prefetch+RRF | Sequential queries | Single network round-trip |
