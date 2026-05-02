---
title: ResearchIT
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# ResearchIT — Personalized ArXiv Paper Recommender

> An "Instagram for research" — a multi-interest aware feed that surfaces relevant papers across a researcher's distinct areas without collapsing toward a dominant interest.

**Stack:** FastAPI · HTMX · Jinja2 · BGE-M3 (1024-dim) · Qdrant Cloud · Zilliz Cloud · Turso (libSQL) · Groq · LightGBM · HuggingFace Spaces

**Live demo:** https://siddhm11-researchit.hf.space

---

## Architecture Overview

```
User → [HTMX Frontend] → [FastAPI Backend]
                              │
                ┌─────────────┼─────────────────┐
                │             │                  │
         [Qdrant Cloud]  [Zilliz Cloud]   [Turso Cloud]
         Dense vectors   Sparse vectors   Paper metadata
         1.6M papers     1.6M papers      ~1.6M rows
         BGE-M3 1024d    BGE-M3 lexical   + citations
                │             │                  │
                └─────────────┼──────────────────┘
                              │
                    [Recommendation Engine]
                     ├── EWMA Profiles
                     ├── Ward Clustering
                     ├── Quota Fusion
                     ├── LightGBM Reranker (37 features)
                     ├── MMR Diversity
                     └── Exploration Injection
```

---

## Data Infrastructure & Schemas

### Qdrant Cloud — Dense Vector Store

| Property | Value |
|----------|-------|
| **Collection** | `arxiv_bgem3_dense` |
| **Documents** | ~1,600,000 arXiv papers |
| **Vector dim** | 1024 (BGE-M3 dense embeddings, float32) |
| **Quantization** | Binary Quantization (BQ) enabled |
| **HNSW** | m=32 |
| **Point ID** | Integer (auto-generated) |
| **Payload** | `arxiv_id` (TEXT, keyword-indexed) |
| **Region** | Qdrant Cloud |

### Zilliz Cloud — Sparse Vector Store

| Property | Value |
|----------|-------|
| **Collection** | `arxiv_bgem3_sparse` |
| **Documents** | ~1,600,000 arXiv papers |
| **Schema** | `id` (INT64 auto PK), `arxiv_id` (VARCHAR), `sparse_vector` (SPARSE_FLOAT_VECTOR) |
| **Index** | SPARSE_INVERTED_INDEX, metric_type=IP |
| **Sparse format** | Integer token IDs as keys (BGE-M3 tokenizer), e.g. `{29: 0.0427, 6083: 0.1852}` |

### Turso (libSQL) — Paper Metadata DB

| Property | Value |
|----------|-------|
| **Database** | `arxiv-data` on `aws-ap-south-1` |
| **URL** | `https://arxiv-data-siddhm11.aws-ap-south-1.turso.io` |
| **Rows** | ~1,600,000 papers |
| **Data sources** | Kaggle `siddhm11/arxivdata` + `siddhm11/citation-data-letsgoo` |

**Table: `papers`**

```sql
CREATE TABLE papers (
    arxiv_id              TEXT UNIQUE,    -- e.g. "2401.12345"
    title                 TEXT,
    authors               TEXT,           -- comma-separated
    categories            TEXT,           -- space-separated arXiv categories
    primary_topic         TEXT,           -- e.g. "cs.CL"
    update_date           TEXT,           -- "YYYY-MM-DD"
    abstract_preview      TEXT,           -- truncated to 500 chars
    citation_count        INTEGER DEFAULT 0,
    influential_citations INTEGER DEFAULT 0
);
CREATE UNIQUE INDEX idx_papers_arxiv_id ON papers(arxiv_id);
```

### SQLite — Local Application DB

**File:** `interactions.db` (WAL mode, async via aiosqlite)

```sql
-- User interactions (saves, dismissals, clicks, views)
CREATE TABLE interactions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          TEXT NOT NULL,
    paper_id         TEXT NOT NULL,
    event_type       TEXT NOT NULL,    -- save | not_interested | click | view
    source           TEXT,             -- search | recommendation
    position         INTEGER,
    query_id         TEXT,
    ranker_version   TEXT,             -- Phase 4.5: pipeline version tag
    candidate_source TEXT,             -- Phase 4.5: cluster_0 | exploration | ewma
    cluster_id       INTEGER,          -- Phase 4.5: interest cluster index
    timestamp        TEXT NOT NULL DEFAULT (datetime('now'))
);

-- arXiv ID → Qdrant integer point ID mapping (lazy cache)
CREATE TABLE paper_qdrant_map (
    arxiv_id        TEXT PRIMARY KEY,
    qdrant_point_id INTEGER NOT NULL,
    mapped_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Paper metadata cache (from Turso/arXiv API)
CREATE TABLE paper_metadata (
    arxiv_id  TEXT PRIMARY KEY,
    title     TEXT, abstract TEXT, authors TEXT,
    category  TEXT, published TEXT,
    cached_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- EWMA user profile embeddings (1024-dim float32 blobs)
CREATE TABLE user_profiles (
    user_id           TEXT NOT NULL,
    profile_type      TEXT NOT NULL,   -- long_term | short_term | negative
    vector            BLOB NOT NULL,   -- 4096 bytes (1024 × float32)
    interaction_count INTEGER DEFAULT 0,
    updated_at        TEXT,
    PRIMARY KEY (user_id, profile_type)
);

-- Ward clustering results per user
CREATE TABLE user_clusters (
    user_id         TEXT NOT NULL,
    cluster_idx     INTEGER NOT NULL,
    medoid_paper_id TEXT NOT NULL,
    importance      REAL NOT NULL,
    paper_ids       TEXT NOT NULL,     -- JSON array of arxiv_ids
    computed_at     TEXT,
    PRIMARY KEY (user_id, cluster_idx)
);

-- Onboarding wizard state
CREATE TABLE user_onboarding (
    user_id              TEXT PRIMARY KEY,
    selected_categories  TEXT,          -- JSON array: ["nlp", "cv", "ml"]
    onboarding_completed INTEGER DEFAULT 0,
    created_at           TEXT, updated_at TEXT
);
```

### LightGBM Reranker — ML Model

| Property | Value |
|----------|-------|
| **File** | `models/reranker-phase6/production_model/reranker_v1.txt` |
| **HuggingFace** | [siddhm11/researchit-reranker-phase6](https://huggingface.co/siddhm11/researchit-reranker-phase6) |
| **Format** | LightGBM v4 text (plain text, no pickle) |
| **Objective** | LambdaRank (optimizes nDCG) |
| **Trees** | 141 (early stopped from 500) |
| **Features** | 37 (see `docs/PHASE6-HANDOFF.md` for full schema) |
| **Size** | 974 KB |
| **Latency** | 0.143ms per 100 candidates |
| **Fallback** | Heuristic scorer when model unavailable |

---

## Recommendation Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│ Tier 1 (≥5 saves): Multi-Interest Clustering + Quota Fusion     │
│   1. Ward clustering → identify distinct interests              │
│   2. Hungarian matching → stabilize cluster IDs                 │
│   3. Quota allocation → per-cluster slot budgets                │
│   4. Parallel per-cluster ANN searches                          │
│   5. LightGBM reranking (37 features) + heuristic fallback     │
│   6. Category suppression (≥3 dismissals in 14 days)            │
│   7. MMR diversity (λ=0.6)                                      │
│   8. Exploration injection (2 serendipitous papers)             │
├──────────────────────────────────────────────────────────────────┤
│ Tier 2 (≥3 saves): EWMA long-term vector → single ANN search   │
├──────────────────────────────────────────────────────────────────┤
│ Tier 3 (≥1 save): Qdrant BEST_SCORE Recommend API               │
├──────────────────────────────────────────────────────────────────┤
│ Tier 0 (onboarded, 0 saves): Trending papers by category        │
└──────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (see .env.example)
cp .env.example .env
# Edit .env with your Qdrant, Zilliz, Turso, Groq credentials

# Run dev server
python run.py
# → http://127.0.0.1:7860

# Run tests
python -m pytest tests/ -v

# Run Phase 6 reranker integration tests
python tests/test_reranker_integration.py
```

---

## Phase Completion Status

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Complete | Zero-ML Recommender (Qdrant + HTMX) |
| 2a | ✅ Complete | EWMA Profile Embeddings |
| 2b | ✅ Complete | Ward Clustering + Multi-Interest |
| 2c | ✅ Complete | Heuristic Re-ranking + MMR |
| 3 | ✅ Complete | Hybrid Semantic Search (BGE-M3 + Qdrant + Zilliz + RRF) |
| 3.5 | ✅ Complete | Turso Metadata DB (2.9x faster search) |
| 4 | ✅ Complete | Quota Fusion + Hungarian Matching + Category Suppression |
| 4.5 | ✅ Complete | Instrumentation Foundation |
| 5 | ✅ Complete | Cold-Start Onboarding + UI Redesign |
| 6 | ✅ Complete | LightGBM Reranker (nDCG@10: 0.879, +233%) |
| 7 | 📋 Planned | Evaluation Framework |
| 8 | 📋 Planned | LLM Summaries + Distilled Reranker |
| 9 | 📋 Planned | Exploration + Collaborative Filtering |

---

## Key Documentation

| Document | Purpose |
|----------|---------|
| `CLAUDE.md` | Agent rulebook — architectural rules, doc precedence, code conventions |
| `docs/TASK-TRACKER.md` | Master task checklist with all phase details |
| `docs/PHASE6-HANDOFF.md` | LightGBM reranker handoff — model provenance, schema, reproduction |
| `docs/phases/PHASE6-Reranker-Framing.md` | Phase 6.1-6.3 framing — feature wiring, deployment verification, retraining strategy |
| `docs/research/06-Deep-Research-Verdict.md` | **Source of truth** for architecture decisions |
| `docs/walkthroughs/04-Next-Steps-and-Phase-Plan.md` | Master roadmap (Phases 3–9) |
| `docs/ML Intern docs/` | ML Intern conversation logs for model training |

---

## Health & Monitoring

```bash
# Phase 6.3: Verify reranker deployment
curl -s https://siddhm11-researchit.hf.space/healthz/reranker | python -m json.tool
# Expected: {"model_loaded": true, "n_trees": 141, "fallback_active": false, ...}
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `QDRANT_URL` | Yes | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Yes | Qdrant Cloud API key |
| `ZILLIZ_URI` | Yes | Zilliz Cloud gRPC endpoint |
| `ZILLIZ_TOKEN` | Yes | Zilliz Cloud API token |
| `TURSO_URL` | Yes | Turso database URL |
| `TURSO_DB_TOKEN` | Yes | Turso auth token |
| `GROQ_API_KEY` | Yes | Groq API key for query rewriting |
| `S2_API_KEY` | No | Semantic Scholar API key (training only) |
| `RERANKER_MODEL_PATH` | No | Override LightGBM model file path |
| `DB_PATH` | No | SQLite path (default: `interactions.db`) |

---

## Test Suite

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_profiles.py` | 11 | EWMA profile computation |
| `test_clustering.py` | 21 | Ward clustering + Hungarian matching |
| `test_reranker_diversity.py` | 13 | Reranker (37-feature) + MMR diversity |
| `test_reranker_integration.py` | 7 | Phase 6 LightGBM integration |
| `test_phase6_feature_wiring.py` | 9 | Phase 6.1+6.2 feature wiring + per-candidate cluster |
| `test_fusion.py` | 20 | Quota allocation |
| `test_db.py` | 19 | SQLite schema + suppression |
| `test_onboarding.py` | 11 | Onboarding wizard |
| `test_hybrid_search.py` | 21 | Hybrid search pipeline |
| `test_search_router.py` | 6 | Search router |
| Others | ~13 | User state, saved, arxiv, qdrant, integration |
| **Total** | **~203** | |
