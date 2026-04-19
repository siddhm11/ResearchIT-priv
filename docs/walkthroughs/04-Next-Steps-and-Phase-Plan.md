# ResearchIT — Next Steps & Revised Phase Plan

> This document synthesizes all research findings (Docs 01–06), the current codebase state,
> E2E test results, and the deep research verdict into a single, actionable roadmap.
> It captures the evolution of the founder's thinking, resolves contradictions between documents,
> and lays out exactly what to build next and why.

---

## Part 1: Where We Are Today (Status Snapshot)

### What's Built and Working

| Component | Status | Evidence |
|---|---|---|
| Qdrant Cloud (1.6M BGE-M3 papers) | ✅ Live | BQ enabled, HNSW m=32, `arxiv_bgem3_dense` collection |
| Phase 1: Zero-ML Recommender | ✅ Complete | Qdrant BEST_SCORE with raw IDs, 55 tests |
| Phase 2a: EWMA Profiles | ✅ Complete | Long-term (α=0.10), Short-term (α=0.40), Negative (α=0.15) |
| Phase 2b: Ward Clustering + Prefetch+RRF | ✅ Complete | Adaptive gap-based threshold, 2+ clusters detected on real data |
| Phase 2c: Heuristic Re-ranking + MMR | ✅ Complete | 4-feature scorer, MMR λ=0.6, exploration injection |
| SQLite (interactions, profiles, clusters, metadata cache) | ✅ Live | WAL mode, async via aiosqlite |
| HTMX Frontend | ✅ Live | Search, save, dismiss, recommendations |
| Test Suite | ✅ 88 tests passing | Unit, integration, and E2E simulation |

### What's NOT Built Yet

| Component | Planned In | Blocked By |
|---|---|---|
| **Hybrid Search (BGE-M3 encode + Zilliz sparse)** | **Phase 3 (NEXT)** | BGE-M3 model loading (~570MB, ~15s cold start) |
| Recommendation fixes (RRF→quota, α tuning) | Phase 4 | Code refactor only |
| LightGBM lambdarank re-ranker | Phase 6 | Need ≥500 labeled save/dismiss interactions |
| Cold-start onboarding (category picker / ORCID) | Phase 5 | Not yet designed |
| Negative profile used in retrieval | Phase 4 | Stored but not wired |
| Pre-populated metadata store | Phase 4 | arXiv API is the latency bottleneck (~7.6s cold) |
| LLM interest summaries per cluster | Phase 8 | Needs Claude/Groq API integration |

> **Note on search architecture:** The current arXiv keyword API search was always a Phase 1 placeholder.
> The entire point of building 1.6M BGE-M3 embeddings in Qdrant (with BQ + HNSW) is to power
> vector-based semantic search. Replacing the arXiv API with Qdrant dense + Zilliz sparse
> hybrid search is the **#1 priority** for the next phase.

### Dataset Coverage

| Field | Value |
|---|---|
| Oldest paper | `0704.0004` (~April 2007) |
| Newest paper | `2505.04101` (~May 2025) |
| Total papers | 1,596,587 |
| Payload stored in Qdrant | `arxiv_id` only |
| Metadata source | arXiv API (live) → SQLite cache |

---

## Part 2: The Founder's Thinking — How It Evolved

This section captures the chronological evolution of the project's architectural decisions, so future contributors understand *why* things are the way they are.

### The Original Vision (Docs 01–02)

The founder's initial concept was "Instagram/TikTok for Research Papers":

- **Discovery-first**: A personalized "For You" feed, not a search box
- **Visual cards**: Paper cards with key figures, TLDRs, and lightweight engagement signals
- **Subject vectors at onboarding**: Users would select topics (CV, NLP, RL, etc.) when they first open the app. The system would maintain fixed vectors for each subject and query Qdrant with them
- **Social layer**: Community playlists, reading paths, emoji reactions
- **Business model**: Freemium ($10-12/mo for Pro)

Key ideas from this era:
- Community-curated reading paths ("paper playlists")
- Prerequisite paper chains
- Research Wrapped (annual retrospective)
- Cross-disciplinary method bridges
- Paper difficulty ratings

### The Architectural Pivot (Doc 03)

Deep research into Twitter, Pinterest, TikTok, and Alibaba revealed that **no production system uses fixed subject vectors**:

- Fixed categories can't adapt to emerging sub-fields
- Category granularity never matches individual user needs
- Averaging cross-topic vectors produces the "centroid-in-nowhere" problem

**The pivot**: Replace explicit subject selection with implicit behavioral tracking:
- EWMA temporal profiles (from Spotify's two-component pattern)
- Ward hierarchical clustering (from Pinterest's PinnerSage)
- Dynamic K per user (auto-determined, not fixed)
- Medoid representation (real papers, not fabricated centroids)

### The Founder's Realization (Doc 05)

The founder explicitly acknowledged this transition:

> "Subject vectors are things which people will select in the start when they open the app — that was my thought earlier, and now it's updated."

This was documented in `05-Evolution-Of-Onboarding-And-Interests.md` as a permanent record of the architectural shift.

### The Deep Research Correction (Doc 06)

The latest deep research (Doc 06) adds critical nuance that **neither pure-behavioral nor pure-subject is right**:

> "The pure-behavioral position in Doc 03/05 is directionally right but structurally incomplete... item-level seeds + adaptive refinement beats both fixed-category questionnaires and pure-behavior-from-zero, and onboarding cues remain a 4–37% lift even once behavioral data exists."

**The corrected position**: A three-layer hybrid:
1. **Coarse arXiv-category multiselect** — filter and LightGBM feature (5-second cold-start signal)
2. **Seed-paper / ORCID import** — initial behavioral profile (strong cold-start signal)
3. **Ward clustering + medoid retrieval** — takes over at ~10 saves (production-grade personalization)

This resolves the tension: subject categories aren't the *primary* user model, but they *are* a useful prior for cold-start, filtering, and as re-ranking features.

---

## Part 3: Contradictions Between Documents — Resolved

The 6 research documents contain several contradictions. Here is each one and its resolution:

### 1. RRF vs Quota Fusion for Recommendations

| Doc 03 (Implemented) | Doc 06 (Correction) |
|---|---|
| Use RRF to fuse results from multiple interest clusters | RRF lets the dominant cluster swamp minor interests |

**Resolution**: Doc 06 is correct. RRF was designed for fusing *different retrievers on the same query*, not *different queries for the same user*. Replace with **importance-weighted quota allocation with a minimum floor**:
- Normalize importance scores to weights
- Allocate feed slots proportional to weight: `slot_k = max(⌊F·w_k⌋, F_min=3)`
- Each cluster gets at least 3 slots regardless of importance

This is what PinnerSage, Taobao ULIM, and Pinterest Bucketized-ANN actually deploy.

**Current status**: RRF is still in the codebase. Needs to be replaced.

### 2. EWMA α_long = 0.10 vs 0.03

| Doc 03 (Implemented) | Doc 06 (Correction) |
|---|---|
| α_long = 0.10 (effective window ~20 interactions) | α_long = 0.03 (effective window ~66 interactions) |

**Resolution**: PinnerSage tested λ=0.1 and **explicitly rejected it as too recent-biased**. Their optimal was λ=0.01. Doc 06 recommends α_long=0.03 as a compromise.

**Current status**: α=0.10 is in the codebase. Should be tuned down to 0.03.

### 3. BGE-reranker-v2 in the Hot Path

| Doc 02/04 (Suggested) | Doc 03 (Rejected) | Doc 06 (Confirmed rejection) |
|---|---|---|
| Use BGE-reranker-v2 for re-ranking | ~800ms for 100 candidates, impractical | Distill offline, never serve hot |

**Resolution**: All documents agree: BGE-reranker-v2 is too slow for CPU serving. Use LightGBM. If cross-encoder signal is needed, distill BGE-reranker-v2 into a TinyBERT student and use as a LightGBM feature on top-20 only.

### 4. Subject Vectors: Use vs Don't Use

| Docs 01/02/04 | Doc 05 | Doc 06 |
|---|---|---|
| Primary user model | Fully rejected | Useful as prior, not primary |

**Resolution**: Doc 06 is the final word. Subject categories serve three roles:
- Cold-start filter (first 1-3 sessions)
- LightGBM categorical feature
- Pool pre-filter when a cluster is small

They are **not** used as primary vectors for retrieval.

### 5. RRF for Search vs Recommendations

| Doc 06 | Current codebase |
|---|---|
| RRF is correct for *search* (fusing dense+sparse), wrong for *recommendations* | RRF used for both |

**Resolution**: These are different problems:
- **Search**: Multiple retrievers answering the *same* query → RRF is correct
- **Recommendations**: Multiple queries for the *same* user → Quota is correct

Keep RRF in the hybrid search pipeline. Replace with quota in the recommendation pipeline.

---

## Part 4: The Revised Phase Plan

Based on all research, here are the next phases in priority order.

### Phase 3: Hybrid Semantic Search (HIGHEST PRIORITY, ~2-3 weeks)

**Why this is #1**: The entire reason we built 1.6M BGE-M3 embeddings in Qdrant with BQ + HNSW is to power real vector-based search. The arXiv keyword API was a Phase 1 throwaway placeholder — it can't understand meaning, only match exact words. A user searching "when AI makes up fake facts" gets nothing useful from the arXiv API but would immediately find hallucination papers via Qdrant dense search. **This is the core product experience.**

This is the existing `PHASE2-Hybrid-Search-Plan.md`, now elevated to the top priority.

#### The Search Pipeline (Dense + Sparse + RRF)

```
User types: "when AI makes up fake facts"
    │
    ▼
[1] LLM Rewriter (Groq / llama-3.3-70b, ~300ms)
    │  → "LLM hallucination factual errors sycophancy truthfulness survey"
    ▼
[2] BGE-M3 Encode (CPU, ~300ms, cached for repeat queries)
    │  → dense_vec (1024-dim) + sparse_dict (lexical weights)
    ▼
[3a] Qdrant dense search (ANN)     [3b] Zilliz sparse search (keyword)
     │  semantic meaning match       │  exact keyword match
     └──────────── parallel ─────────┘
                    │
                    ▼
[4] RRF Fusion (correct here — same query, different retrievers)
    │  Papers in BOTH lists get boosted
    ▼
[5] Rerank: 0.80 × rrf_rank + 0.20 × recency
    ▼
Final results → fetch metadata → render
```

**RRF is correct here** — this is fusing *different retrievers on the same query*, unlike recommendations where we fuse *different queries for the same user*.

#### Components to Build
1. `app/embed_svc.py` — BGE-M3 model singleton (load once at startup, ~15s)
2. `app/zilliz_svc.py` — Zilliz sparse search client
3. `app/groq_svc.py` — LLM query rewriter (Groq / llama-3.3-70b)
4. `app/hybrid_search_svc.py` — Orchestrator (rewrite → encode → parallel search → RRF)
5. Swap `search.py` router to use hybrid pipeline
6. Add BGE-M3 warm-up to `main.py` lifespan

#### Latency Budget
| Stage | Time |
|---|---|
| LLM rewrite (Groq) | ~300ms (skippable for academic queries) |
| BGE-M3 encode | ~300ms first, ~0ms cached |
| Qdrant + Zilliz (parallel) | ~300ms |
| RRF + rerank | <5ms |
| **Total** | **~600ms warm** |

---

### Phase 4: Recommendation Pipeline Fixes (~1 week)

Corrections to the existing recommendation pipeline based on Doc 06's findings.

#### 4.1 Replace RRF with Importance-Weighted Quota Fusion
**Why**: RRF lets dominant clusters swamp minor interests — the exact failure mode multi-interest models exist to prevent.

**What to change**: In `app/routers/recommendations.py`, replace `multi_interest_search()` (which uses Qdrant's server-side RRF) with per-cluster separate ANN queries, then allocate feed slots proportional to cluster importance with a floor of F_min=3.

**New flow**:
```
clusters = compute_clusters(...)
weights = normalize_importance(clusters)
for each cluster k:
    slots_k = max(floor(total_slots × weight_k), 3)
    candidates_k = qdrant search with medoid_k (limit = slots_k × 3)
    rerank within cluster_k via LightGBM / heuristic
    take top slots_k
deduplicate across clusters (assign to highest-ranked)
MMR over the merged union
```

#### 4.2 Tune α_long from 0.10 → 0.03
**Why**: PinnerSage explicitly rejected 0.10 as too recent-biased.

**What to change**: Single constant in `app/recommend/profiles.py`.

#### 4.3 Wire the Negative Profile into Re-ranking
**Why**: Currently computed and stored but never used. YouTube showed a 3× gain from using dislikes as both features and labels.

**What to add**: In `app/recommend/reranker.py`, add a negative-similarity penalty:
```python
neg_penalty = cosine_sim(candidate, neg_profile) * penalty_weight
final_score = base_score - neg_penalty
```

#### 4.4 Pre-populate Metadata Store
**Why**: The arXiv API is the #1 latency bottleneck (~7.6 seconds cold for 50 papers).

**What to do**: Download the Kaggle arXiv metadata dataset (~4GB JSON). Bulk-insert all 1.6M papers' metadata into SQLite's `paper_metadata` table. The arXiv API becomes a fallback for genuinely new papers only.

**Impact**: Metadata fetch drops from ~7,600ms to <5ms.

---

### Phase 5: Cold-Start Onboarding (~1-2 weeks)

Build the onboarding pipeline that Doc 06 identifies as a 4-37% lift even once behavioral data exists.

#### 5.1 arXiv Category Multi-Select
A simple UI screen on first visit: select 3-5 arXiv categories (cs.CL, cs.CV, stat.ML, etc.).
- Used as pool filter for first 1-3 sessions
- Stored as a LightGBM feature permanently
- Does NOT create "subject vectors" — just filters

#### 5.2 Seed Paper Import
Let users search for and save 3-5 seed papers during onboarding.
- These immediately create EWMA profiles and Ward clusters
- Bypasses the "save 5 papers before any recs" cold-start trap
- Scholar Inbox found this sufficient for good initial recommendations
- **With hybrid search in place (Phase 3), seed paper search will use Qdrant vectors, not the arXiv API**

#### 5.3 ORCID / Semantic Scholar ID Import (Stretch)
If the user pastes their ORCID, ingest their authored papers as initial saves.
- This gives the system 10-50 papers worth of signal instantly
- Creates highly personalized clusters from Day 1

---

### Phase 6: LightGBM Re-ranker (~2-4 weeks, when data exists)

Replace the heuristic scorer with a trained LightGBM lambdarank model.

#### Training Data Bootstrap (No Real Users Needed)
From Doc 06, use citation-graph pseudo-labels:
- Each paper as query user → its cited papers = relevance 2, co-cited = relevance 1, random = 0
- Author-as-user simulation: author's first N papers = profile, next M papers' citations = positives
- Use unarXive 2022 dataset (all arXiv with structured citations)

#### Features (~30-50)
- Cosine similarity to long-term medoid, short-term vector
- Sparse score (Zilliz), dense score (Qdrant)
- Log-citation count, citation velocity (age-decayed)
- arXiv primary-category match to cluster medoid
- Author overlap with saved papers
- Citation-graph overlap (second-order)
- Paper age (days since publication)
- Onboarding category match (if available)

#### Target Latency
LightGBM: ~1ms for 100 candidates (sub-millisecond per item). Well under 30ms budget.

---

### Phase 7: Evaluation Framework (~1 week)

Build offline and online evaluation before scaling users.

#### Offline Metrics
- **nDCG@10** — primary (rewards relevance + correct ranking)
- **Recall@50** — secondary (coverage of relevant items)
- **HR@10** — sanity check (at least one relevant item in top 10)
- **ILS** — intra-list similarity (diversity measurement)
- **Category entropy** — how many different arXiv categories in the list
- **Novel-in-top-10** — discovery measurement

#### Offline Protocol
Time-split evaluation on unarXive 2022 + S2ORC:
- Train on interactions before timestamp T
- Test on interactions after T
- Compare against SPECTER2 baselines using SciDocs/SciRepEval conventions

#### Online Metrics (Once Users Exist)
- CTR on recommendations
- Save rate (stronger signal than clicks)
- Dwell time on clicked recommendations
- Return rate (daily active users)

---

### Phase 8: LLM Interest Summaries + Distilled Re-ranker (~2 weeks)

#### 8.1 Claude/Groq Interest Summaries
Generate human-readable per-cluster descriptions:
> "You're reading about retrieval-augmented generation, particularly evaluation and hallucination detection."

Store alongside medoids. Display in the UI as "Your Research Interests."

#### 8.2 Distilled Cross-Encoder
Offline: run BGE-reranker-v2-m3 on top-20 candidates for 10K queries.
Train TinyBERT-L2 student on teacher scores (FlashRank recipe).
Deploy student as a LightGBM feature. ~95% of teacher quality at 1/10 latency.

**Never put the full BGE-reranker in the hot path.**

---

### Phase 9: Exploration & Collaborative Filtering (Future)

#### 9.1 Epsilon-Greedy Exploration
Reserve 5-10% of feed slots for papers outside the user's clusters.
- New users: ε=0.25 (more exploration)
- Established users: ε=0.05 (mostly exploitation)

#### 9.2 Collaborative Filtering (≥500 Users)
Add LightFM hybrid model with switching strategy:
- <10 interactions: content-based
- ≥10 interactions: LightFM
Retrain LightGBM with dismissals as negative labels (YouTube's 3× gain from dual labels).

#### 9.3 Category-Level Negative Suppression
If ≥3 dismissals hit the same arXiv category within a week, suppress that category for 2 weeks.

---

## Part 5: The Three Highest-Impact Next Actions

If you can only do three things, do these:

### 1. Build hybrid semantic search (Phase 3)
**Impact**: Replaces the arXiv keyword API placeholder with real vector-based search. This is what the 1.6M BGE-M3 embeddings in Qdrant were built for. Transforms the product from a keyword aggregator into a semantic discovery engine.
**Effort**: 4 new service files + router swap. ~2-3 weeks.

### 2. Pre-populate the metadata store (Phase 4.4)
**Impact**: Drops cold metadata fetch from 7,600ms to <5ms. Single biggest latency win.
**Effort**: Download Kaggle dataset, write a bulk-insert script, run once.

### 3. Replace RRF with quota fusion in recommendations (Phase 4.1)
**Impact**: Prevents the dominant cluster from drowning out minority interests. Fixes the core multi-interest failure mode.
**Effort**: Refactor `_multi_interest_recommend()` in recommendations.py.

---

## Appendix: Document Index (All Research & Plans)

| # | Document | Purpose | Status |
|---|---|---|---|
| 01 | [Vision: Instagram for Research](../research/01-Vision-Instagram-for-Research.md) | Strategic north star, competitive landscape, UX patterns | ✅ Complete |
| 02 | [Recommendation System Blueprint](../research/02-Recommendation-System-Blueprint.md) | Initial rec system research (user modeling, CF, cold start) | ✅ Complete |
| 03 | [Multi-Interest Architecture RFC](../research/03-MultiInterest-Recommender-Architecture.md) | The architecture we implemented (EWMA, Ward, RRF, LightGBM) | ✅ Implemented |
| 04 | [Technical Roadmap (Legacy)](../research/04-Technical-Roadmap-Legacy.md) | Earlier roadmap, partially superseded | ⚠️ Legacy |
| 05 | [Evolution of Onboarding & Interests](../research/05-Evolution-Of-Onboarding-And-Interests.md) | Documents the subject-vector → behavioral pivot | ✅ Complete |
| 06 | [Deep Research Verdict](../research/06-Deep-Research-Verdict.md) | Final verdict: hybrid approach, RRF→quota, α correction | ✅ Complete |
| — | [Phase 1 Walkthrough](PHASE1-Zero-ML-Recommender.md) | Zero-ML recommender code tour | ✅ Complete |
| — | [Phase 2 Recommender Walkthrough](02-Phase2-MultiInterest-Recommender.md) | Multi-interest engine implementation | ✅ Complete |
| — | [Code Summary & Test Plan](03-Code-Summary-and-Test-Plan.md) | Codebase summary and testing strategy | ✅ Complete |
| — | [Phase 2 Hybrid Search Plan](../phases/PHASE2-Hybrid-Search-Plan.md) | BGE-M3 + Zilliz hybrid search (not yet built) | 📋 Planned |
| — | **This Document** | Revised phase plan synthesizing all research | ✅ Current |
