# ResearchIT — Master Task Tracker

> **Purpose**: Single source of truth for all completed, in-progress, and upcoming work.  
> **Last updated**: 2026-04-20  
> **Current phase**: Phase 3.5 (Turso Metadata DB) — complete, integration pending  

---

## Legend

- `[x]` — Done
- `[/]` — In progress
- `[ ]` — Not started
- `[~]` — Intentionally deferred (blocked by data/users/scale)
- `[!]` — Backlog item (documented, not yet coded)

---

## Phase 1: Zero-ML Recommender ✅ COMPLETE

> *Built the foundation: Qdrant connection, arXiv search, save/dismiss, cookie identity, HTMX frontend.*

- [x] Qdrant Cloud connection (1.6M BGE-M3 papers, BQ, HNSW m=32)
  - Collection: `arxiv_bgem3_dense`, 1024-dim dense vectors
  - File: `app/qdrant_svc.py` → `_get_client()`
- [x] BEST_SCORE Recommend API (raw paper IDs → Qdrant)
  - File: `app/qdrant_svc.py` → `recommend()`
- [x] arXiv keyword API search (placeholder — replaced in Phase 3)
  - File: `app/arxiv_svc.py` → `search()`
- [x] arXiv metadata fetching + SQLite cache
  - File: `app/arxiv_svc.py` → `fetch_metadata_batch()`
- [x] SQLite database schema (interactions, paper_metadata)
  - File: `app/db.py` → `init_db()`
  - WAL mode, async via aiosqlite
- [x] Cookie-based user identity
  - File: `app/config.py` → `COOKIE_NAME`
- [x] User state management (positive/negative deques)
  - File: `app/user_state.py` → `UserState`
- [x] Save/Dismiss event logging
  - File: `app/routers/events.py`
- [x] HTMX + Jinja2 frontend (search, recs, save, dismiss)
  - Files: `app/templates/` (base.html, index.html, search.html, saved.html, partials/)
- [x] Test suite — **55 tests passing**

**Gaps**: None.

---

## Phase 2a: EWMA Profile Embeddings ✅ COMPLETE

> *Replaced raw ID-list approach with temporal decay vectors so recent interests outweigh old ones.*

- [x] Create `app/recommend/` module with `__init__.py`
- [x] Create `app/recommend/profiles.py` — EWMA computation + storage
  - Long-term: α=0.03 ✅ (corrected from 0.10 per Doc 06)
  - Short-term: α=0.40
  - Negative: α=0.15
  - All embeddings L2-normalized
- [x] Modify `app/db.py` — add `user_profiles` table + `user_clusters` table
- [x] Modify `app/qdrant_svc.py` — add `get_paper_vectors()` and `search_by_vector()`
- [x] Modify `app/routers/events.py` — trigger EWMA updates on save/dismiss
- [x] Modify `app/routers/recommendations.py` — EWMA vector search with Tier 2 fallback
- [x] Add `numpy` + `scipy` to `requirements.txt`
- [x] Tests for profiles module — **11 passed**
- [x] Full test suite — no regressions

**Doc 06 correction applied**: α_long 0.10 → 0.03 (PinnerSage rejected 0.10 as too recent-biased).

**Gaps**: None.

---

## Phase 2b: Ward Clustering + Multi-Interest Retrieval ✅ COMPLETE

> *Detect distinct user interests via hierarchical clustering, retrieve candidates per interest.*

- [x] Create `app/recommend/clustering.py` — Ward clustering + medoid extraction
  - L2-normalize embeddings before Ward ✅ (Doc 06 correction)
  - Adaptive gap-based threshold (no fixed K)
  - Medoid representation (real papers, not centroids) ✅
  - Dynamic K (1–7 clusters, auto-determined)
  - Recency-weighted importance scores
- [x] Modify `app/qdrant_svc.py` — add `multi_interest_search()` with prefetch+RRF
- [x] Modify `app/routers/recommendations.py` — 3-tier cascading pipeline
  - Tier 1 (≥5 saves): Multi-interest clustering → prefetch + RRF
  - Tier 2 (≥3 saves): EWMA long-term vector → single ANN search
  - Tier 3 (≥1 save): Qdrant BEST_SCORE Recommend API
- [x] Tests for clustering module — **10 passed**
- [x] Full test suite — no regressions

**Doc 06 corrections applied**: L2-normalization before Ward, medoid not centroid.

**Gaps (deferred to Phase 4)**:
- [!] RRF → quota fusion (dominant clusters can swamp minority interests)
- [!] Hungarian matching for cluster ID stability across reclusterings

---

## Phase 2c: Heuristic Re-ranking + MMR Diversity ✅ COMPLETE

> *Added scoring and diversity layers on top of retrieval to produce the final feed.*

- [x] Create `app/recommend/reranker.py` — 5-feature heuristic scorer
  - Feature 1: cosine_sim_longterm (weight 0.40)
  - Feature 2: cosine_sim_shortterm (weight 0.25)
  - Feature 3: paper_age_days / recency (weight 0.15)
  - Feature 4: rrf_position (weight 0.10)
  - Feature 5: cosine_sim_negative (weight -0.15) ✅ (Doc 06 addition)
- [x] Create `app/recommend/diversity.py` — MMR + exploration injection
  - MMR with λ=0.6
  - 2 serendipitous exploration papers per feed
- [x] Modify `app/routers/recommendations.py` — full 5-step pipeline
  - Step 1: Clustering → Step 2: Retrieval → Step 3: Rerank → Step 4: MMR → Step 5: Exploration
- [x] Tests for reranker + diversity — **13 passed**
- [x] Full test suite — **88 passed** (86 + 2 pre-existing live Qdrant failures resolved)

**Doc 06 correction applied**: Negative EWMA profile wired as Feature 5 with 0.15 penalty.

**Gaps (deferred to Phase 6)**:
- [~] LightGBM lambdarank model (requires ≥500 labeled interactions)

---

## Phase 2d: Advanced Models ❌ DEFERRED (Blocked by data/users)

> *These logically belong to the recommendation engine but cannot be built without real user data or scale.*

- [~] LightGBM lambdarank model — requires ≥500 labeled save/dismiss interactions → Phase 6
- [~] Collaborative filtering features — requires ≥500 users → Phase 9
- [~] DPP diversity — explicitly ruled out for v1 by Doc 06 → Phase 9+
- [~] Two-Tower model — requires GPU + large dataset → Phase 9+

---

## Phase 3: Hybrid Semantic Search ✅ COMPLETE

> *Replace the arXiv keyword API placeholder with real vector-based semantic search using Qdrant dense + Zilliz sparse + RRF.*  
> *Detailed plan: `docs/phases/PHASE3-Hybrid-Semantic-Search.md`*  
> *Prototype reference: `docs/phases/PHASE2-Hybrid-Search-Plan.md`*  
> *Deployment target: Hugging Face Spaces (Docker SDK, 16GB RAM, 2 vCPUs)*

### New files created
- [x] `app/embed_svc.py` — BGE-M3 model singleton (load BAAI/bge-m3 once at startup, ~570MB, ~15s cold)
  - `encode_query(text)` → `(dense: np.ndarray[1024], sparse: dict)`
  - LRU cache for repeat queries
  - Thread-safe, lazy loading with double-check locking
- [x] `app/zilliz_svc.py` — Zilliz Cloud sparse search client
  - Collection: `arxiv_bgem3_sparse`
  - Schema: `id` (INT64 auto PK), `arxiv_id` (VARCHAR), `sparse_vector` (SPARSE_FLOAT_VECTOR)
  - Index: SPARSE_INVERTED_INDEX, metric_type=IP
  - Sparse format: `{int_token_id: float_weight}` (BGE-M3 lexical weights, NOT string words)
  - `search_sparse(sparse_dict, limit)` → `list[dict]` with arxiv_id + score
  - gRPC reconnect handling
- [x] `app/groq_svc.py` — LLM query rewriter (Groq / llama-3.3-70b)
  - `rewrite(user_query)` → academic query string
  - Graceful fallback to original query on error
  - Academic-detection heuristic to skip unnecessary rewrites
  - 2s hard timeout
- [x] `app/hybrid_search_svc.py` — search orchestrator
  - Rewrite → Encode → Parallel (Qdrant dense + Zilliz sparse) → RRF → Rerank
  - Each step has independent failure handling
  - Recency reranking: 0.80 RRF + 0.20 recency

### Files modified
- [x] `app/config.py` — added `ZILLIZ_URI`, `ZILLIZ_TOKEN`, `ZILLIZ_COLLECTION`, `GROQ_API_KEY`, `BGE_M3_MODEL`, `BGE_M3_DEVICE`, `ENCODE_CACHE_SIZE`, search weights, `APP_PORT`
- [x] `app/qdrant_svc.py` — added `search_dense(dense_vec, limit)` for raw vector search returning scores
- [x] `app/routers/search.py` — swapped `arxiv_svc.search()` → `hybrid_search_svc.search()` with arXiv fallback
- [x] `app/main.py` — added graceful BGE-M3 warm-up to lifespan
- [x] `requirements.txt` — added `FlagEmbedding`, `pymilvus`, `groq`
- [x] `run.py` — configurable port (7860 default for HF Spaces)

### Deployment files created
- [x] `Dockerfile` — HF Spaces Docker SDK, CPU-only PyTorch, pre-baked BGE-M3 model
- [x] `.dockerignore` — excludes notebooks, PDFs, databases, caches

### Implementation steps completed
- [x] Step 1: BGE-M3 model service (`embed_svc.py`) + unit tests
- [x] Step 2: Zilliz client (`zilliz_svc.py`)
- [x] Step 3: Dense search in Qdrant service
- [x] Step 4: Groq rewriter (`groq_svc.py`)
- [x] Step 5: Hybrid search orchestrator (`hybrid_search_svc.py`)
- [x] Step 6: Swap search router
- [x] Step 7: Model warm-up + deployment config
- [x] Step 8: Tests — **21 new tests passing** (RRF, recency, Groq heuristics, embed edge cases, orchestrator mocks)

### Test results
- 88 original tests: ✅ All pass (zero regressions)
- 21 Phase 3 unit tests: ✅ All pass (RRF, recency, Groq, embed, orchestrator mocks)
- 6 search router tests: ✅ All pass (ranking, fallback, HTMX, saved state)
- 8 live service tests: ✅ All pass (Qdrant dense, Zilliz sparse, Groq rewrite, parallel)
- **Total: 123 tests passing**

### Latency budget
| Stage | Time |
|---|---|
| LLM rewrite (Groq) | ~300ms (skippable) |
| BGE-M3 encode (CPU) | ~300ms first, ~0ms cached |
| Qdrant + Zilliz (parallel) | ~300ms |
| RRF + rerank | <5ms |
| **Total (warm)** | **~600ms** |

---

## Phase 3.5: Turso ArXiv Metadata DB ✅ COMPLETE

> *Bulk-loaded 1.23 GB of arXiv paper metadata + citation data to Turso (libSQL) cloud DB.*  
> *Eliminates the unstable arXiv API dependency for metadata fetching (Phase 4.2 solved early).*  
> *Created from Kaggle notebook — no code changes to ResearchIT codebase yet.*

### Infrastructure
- [x] Turso cloud DB created: `arxiv-data` on `aws-ap-south-1`
  - URL: `https://arxiv-data-siddhm11.aws-ap-south-1.turso.io`
  - Auth: Platform token + DB auth token (minted via CLI)
- [x] Table: `papers` with columns:
  - `arxiv_id` (TEXT, UNIQUE INDEX `idx_papers_arxiv_id`)
  - `title` (TEXT)
  - `authors` (TEXT)
  - `categories` (TEXT)
  - `primary_topic` (TEXT)
  - `update_date` (TEXT)
  - `abstract_preview` (TEXT, truncated to 500 chars)
  - `citation_count` (INTEGER, default 0)
  - `influential_citations` (INTEGER, default 0)
- [x] Data sources:
  - `arxiv_comprehensive_papers.csv` (Kaggle: siddhm11/arxivdata)
  - `arxiv_citations_summary.csv` (Kaggle: siddhm11/citation-data-letsgoo)
  - Joined on `id` = `arxiv_id_clean`, deduplicated
- [x] Row count verified: local ↔ remote match
- [x] Unique index on `arxiv_id` for fast lookups

### Integration plan (not yet wired into code)
- [ ] Add `TURSO_URL` and `TURSO_DB_TOKEN` to `config.py` / `.env`
- [ ] Create `app/turso_svc.py` — metadata lookup service
  - `fetch_metadata_batch(arxiv_ids)` → `{arxiv_id: paper_dict}`
  - Uses `libsql-experimental` or `libsql-client` (HTTP)
- [ ] Replace `arxiv_svc.fetch_metadata_batch()` with `turso_svc.fetch_metadata_batch()` in `search.py`
- [ ] arXiv API becomes fallback for papers not in Turso DB
- [ ] **Impact**: Metadata fetch drops from ~7,600ms to <50ms

---

## Phase 4: Recommendation Pipeline Fixes 📋 NOT STARTED

> *Fix the known architectural debt in the recommendation pipeline.*  
> *Estimated effort: ~1 week*

### 4.1 — Replace RRF with Importance-Weighted Quota Fusion
- [ ] Create `app/recommend/fusion.py` — quota allocation logic
  - `w_k = importance_k / sum(importance_k)`
  - `slot_k = max(floor(F × w_k), F_min=3)` — every cluster gets at least 3 slots
  - Distribute remainder by largest fractional part
- [ ] Refactor `_multi_interest_recommend()` in `recommendations.py`
  - Replace `multi_interest_search()` with per-cluster separate ANN queries
  - Allocate feed slots proportionally
  - Deduplicate across clusters (assign to highest-ranked)
  - MMR over merged union

### 4.2 — Pre-populate Metadata Store ✅ DONE (via Turso)
- [x] Bulk-loaded arXiv metadata from Kaggle to Turso cloud DB (Phase 3.5)
- [x] 1.23 GB, includes citation counts from Semantic Scholar
- [ ] Wire Turso service into `search.py` to replace arXiv API calls
- [ ] arXiv API becomes fallback for genuinely new papers only
- [ ] **Impact**: Metadata fetch drops from ~7,600ms to <50ms

### 4.3 — Hungarian Matching for Cluster Stability
- [ ] Implement Hungarian matching in `clustering.py`
  - Match new cluster IDs to previous IDs by medoid similarity
  - Prevents cluster IDs from shuffling between reclusterings

### 4.4 — Wire Remaining Negative Signal Components
- [ ] Per-item short-term decay: `score -= α × exp(-dt / τ_neg)` — needs per-item timestamp tracking
- [ ] Category-level suppression: if ≥3 dismissals hit the same arXiv category within a week, suppress for 2 weeks

---

## Phase 5: Cold-Start Onboarding 📋 NOT STARTED

> *Build the hybrid onboarding pipeline for new users.*  
> *Estimated effort: ~1-2 weeks*  
> *Reference: Doc 06 — "4-37% lift even once behavioral data exists"*

### 5.1 — arXiv Category Multi-Select
- [ ] UI screen on first visit: select 3-5 arXiv categories
- [ ] Store selections in SQLite
- [ ] Use as pool filter for first 1-3 sessions
- [ ] Preserve as LightGBM feature permanently
- [ ] Does NOT create "subject vectors" — just filters

### 5.2 — Seed Paper Import
- [ ] Let users search for and save 3-5 seed papers during onboarding
- [ ] Immediately create EWMA profiles + Ward clusters
- [ ] Uses hybrid search (Phase 3) for discovery

### 5.3 — ORCID / Semantic Scholar Import (Stretch)
- [ ] Accept ORCID ID → fetch authored papers → initial saves
- [ ] Gives 10-50 papers of signal instantly

### 5.4 — Popularity Fallback
- [ ] If user skips all onboarding: serve popularity-per-selected-category feed

---

## Phase 6: LightGBM Re-ranker 📋 NOT STARTED

> *Replace heuristic scorer with a trained LightGBM lambdarank model.*  
> *Blocked by: ≥500 labeled interactions OR citation-graph bootstrap*  
> *Estimated effort: ~2-4 weeks*

- [ ] Citation-graph pseudo-labels from unarXive 2022 (cited = relevance 2, co-cited = 1, random = 0)
- [ ] Author-as-user simulation
- [ ] ~30-50 features including sparse/dense scores, citation count, category match, author overlap
- [ ] Train LightGBM with `objective='lambdarank'`
- [ ] Target: ~1ms for 100 candidates

---

## Phase 7: Evaluation Framework 📋 NOT STARTED

> *Build offline and online evaluation before scaling users.*  
> *Estimated effort: ~1 week*

- [ ] Offline metrics: nDCG@10, Recall@50, HR@10, ILS, category entropy
- [ ] Time-split evaluation on unarXive 2022 + S2ORC
- [ ] Online metrics (once users exist): CTR, save rate, dwell time, return rate

---

## Phase 8: LLM Interest Summaries + Distilled Re-ranker 📋 NOT STARTED

> *Estimated effort: ~2 weeks*

- [ ] Claude/Groq interest summaries per cluster (human-readable descriptions)
- [ ] Distill BGE-reranker-v2-m3 offline → TinyBERT-L2 student (FlashRank recipe)
- [ ] Deploy student score as LightGBM feature on top-20

---

## Phase 9: Exploration + Collaborative Filtering 📋 NOT STARTED

> *Blocked by: ≥500 users*

- [ ] Epsilon-greedy exploration (ε=0.25 new users, ε=0.05 established)
- [ ] LightFM hybrid CF model with switching strategy
- [ ] Category-level negative suppression
- [ ] Retrain LightGBM with dismissals as negative labels

---

## Appendix: Infrastructure Status

| Component | Status | Details |
|---|---|---|
| **Qdrant Cloud** | ✅ Live | 1.6M papers, BGE-M3 1024-dim, BQ enabled, HNSW m=32 |
| **Zilliz Cloud** | ✅ Live | 1.6M papers, BGE-M3 sparse vectors, collection `arxiv_bgem3_sparse` |
| **Turso (libSQL)** | ✅ Live | 1.23 GB arXiv metadata + citations, `arxiv-data` DB, `papers` table, unique index on `arxiv_id` |
| **SQLite** | ✅ Live | interactions, paper_metadata (local cache), user_profiles, user_clusters |
| **HF Spaces** | ✅ Deployed | Docker SDK, free tier, port 7860 — https://siddhm11-researchit.hf.space |
| **Render** | ⚠️ Previous target (512MB RAM too small for BGE-M3) | May still be used for non-ML services |
| **arXiv API** | ✅ Live | Keyword search fallback + metadata fetch (to be replaced by Turso) |
| **BGE-M3 Model** | ✅ Live | Pre-baked in Docker image, warm-up at startup |
| **Groq API** | ✅ Code written, fallback-enabled | `app/groq_svc.py` — 2s timeout, academic heuristic skip |
| **Notebooks** | ✅ Organized | `notebooks/` — 01-upload, 02-test, 03-search-benchmark |

### Credentials Status

| Credential | Status | Env Var | Notes |
|---|---|---|---|
| **Qdrant Cloud** | ✅ In `.env` | `QDRANT_URL`, `QDRANT_API_KEY` | Already wired |
| **Zilliz Cloud** | ✅ In `.env` | `ZILLIZ_URI`, `ZILLIZ_TOKEN` | Phase 3, wired |
| **Turso (libSQL)** | ✅ Token minted | `TURSO_URL`, `TURSO_DB_TOKEN` | Phase 3.5, not yet in config.py |
| **Groq** | ✅ In `.env` | `GROQ_API_KEY` | Phase 3, wired |
| **HF Spaces** | ✅ Deployed | Secrets panel | Need to add all env vars |

---

## Appendix: Test Suite

| Test File | Count | Status |
|---|---|---|
| `tests/test_profiles.py` | 11 | ✅ Passing |
| `tests/test_clustering.py` | 10 | ✅ Passing |
| `tests/test_reranker_diversity.py` | 13 | ✅ Passing |
| `tests/test_db.py` | — | ✅ Passing |
| `tests/test_qdrant_svc.py` | — | ✅ Passing |
| `tests/test_arxiv_svc.py` | — | ✅ Passing |
| `tests/test_integration.py` | — | ✅ Passing |
| `tests/test_user_state.py` | — | ✅ Passing |
| `tests/test_saved.py` | — | ✅ Passing |
| `tests/test_hybrid_search.py` | 21 | ✅ Passing |
| `tests/test_search_router.py` | 6 | ✅ Passing |
| `tests/test_live_search.py` | 8 | ✅ Passing |
| **Total** | **123** | ✅ |
| `test_e2e_recs.py` (standalone) | 1 | ✅ E2E simulation |

---

## Appendix: Doc 06 Corrections — Tracking

| Correction | Status | Where |
|---|---|---|
| α_long 0.10 → 0.03 | ✅ Applied | `app/recommend/profiles.py:30` |
| L2-normalize before Ward clustering | ✅ Applied | `app/recommend/clustering.py` |
| Medoid not centroid | ✅ Applied | `app/recommend/clustering.py` → `_find_medoid()` |
| Negative EWMA wired into reranking | ✅ Applied | `app/recommend/reranker.py` → Feature 5 |
| RRF → quota fusion for recommendations | [!] Backlog | Phase 4.1 |
| Hungarian cluster matching | [!] Backlog | Phase 4.3 |
| Per-item short-term negative decay | [!] Backlog | Phase 4.4 |
| Category-level suppression | [!] Backlog | Phase 4.4 |
| BGE-reranker NEVER in hot path | ✅ Followed | Heuristic scorer used instead |
