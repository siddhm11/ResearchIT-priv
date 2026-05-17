# CLAUDE.md — ResearchIT Coding Agent Rulebook

> **Read this file first, every session, before touching anything else.** This file tells you which docs to trust, in what order, and the non-negotiable rules for this codebase. If you skip this file you will produce code that contradicts months of architectural research.
>
> **Last updated**: 2026-05-03

---

## 1. What this codebase is

ResearchIT is a personalized arXiv paper recommendation engine. ~1.6M papers with pre-computed BGE-M3 (1024-dim) dense embeddings. CPU-only (zero GPU). FastAPI + HTMX + Jinja2 on the front, Qdrant Cloud (dense, `arxiv_bgem3_dense` collection, BQ enabled, HNSW m=32) + Zilliz Cloud (sparse, `arxiv_bgem3_sparse` collection — wiring in Phase 3) for vectors, SQLite for interactions/profiles/clusters/metadata cache, Hugging Face Spaces (Docker SDK, free tier: 16GB RAM, 2 vCPUs) for deployment. Single developer (Amin). Pre-launch — no real users yet.

**Endgame:** an "Instagram for research" — multi-interest aware feed that surfaces relevant papers across a user's distinct research areas without collapsing toward a dominant interest.

**You are a coding agent operating inside this codebase.** Optimize for: (a) not contradicting the architectural decisions in `docs/research/06-Deep-Research-Verdict.md`, (b) shipping working code under tight latency budgets, (c) flagging uncertainty rather than guessing.

---

## 2. The document map — read this before consulting any doc

There are six research documents in `docs/research/`, four walkthroughs in `docs/walkthroughs/`, and two phase plans in `docs/phases/`. The research docs were written at different times and **they contradict each other**. Follow this precedence strictly:

### Research documents (`docs/research/`)

| Priority | File | Status | Use it for | Do NOT use it for |
|---|---|---|---|---|
| **1 (canonical)** | `06-Deep-Research-Verdict.md` | **Source of truth** | Any architectural question. Algorithm choices, parameter values, fusion strategy, phase plan, what's in/out of scope. | N/A — always trust this first. |
| **2** | `03-MultiInterest-Recommender-Architecture.md` | Implemented, with corrections from Doc 06 (see addendum at bottom of doc) | Detailed component descriptions where 06 is silent (e.g., specific Qdrant query shapes, MMR mechanics). | Anything 06 contradicts (RRF fusion for recs, alpha_long=0.10, BGE-reranker in hot path). |
| **3** | `02-Recommendation-System-Blueprint.md` | Older blueprint | Background context on the original blueprint. | Onboarding design (it advocates fixed subject vectors — superseded). |
| **4** | `01-Vision-Instagram-for-Research.md` | Product vision | Product/UX intent, growth loops, swipe mechanics, what the user-facing product should *feel* like. | Any backend architecture decisions. |
| **5 (legacy)** | `04-Technical-Roadmap-Legacy.md` | **Superseded** | Historical reference only. | Anything actionable. Treat as read-only context. |
| **6 (history)** | `05-Evolution-Of-Onboarding-And-Interests.md` | Historical narrative | Understanding *why* we pivoted from subject vectors to behavioral. | Implementation guidance — its "pure behavioral" conclusion was itself superseded by 06's hybrid verdict. |

### Phase plans (`docs/phases/`)

| File | What it covers |
|---|---|
| `PHASE1-Zero-ML-Recommender.md` | What Phase 1 built (Qdrant, arXiv API, HTMX) |
| `PHASE2-Hybrid-Search-Plan.md` | Prototype reference for search pipeline (superseded by Phase 3 doc) |
| `PHASE3-Hybrid-Semantic-Search.md` | **Active Phase 3 implementation plan** — BGE-M3 + Qdrant dense + Zilliz sparse + RRF |

### Walkthroughs (`docs/walkthroughs/`)

| File | What it covers |
|---|---|
| `01-Phase1-Code-Tour.md` | File-by-file walkthrough of the Phase 1 zero-ML codebase |
| `02-Phase2-MultiInterest-Recommender.md` | Multi-interest engine implementation (EWMA, Ward, RRF, reranker, MMR) |
| `03-Code-Summary-and-Test-Plan.md` | Codebase summary and three-layered testing strategy |
| `04-Next-Steps-and-Phase-Plan.md` | **Master roadmap** synthesizing all 6 research docs into Phases 3-9 |

### How to use the map in practice

- **Architecture question?** Open `docs/research/06-Deep-Research-Verdict.md`. Stop there if it answers. If 06 is silent, fall through to 03.
- **Product/UX question?** Open `docs/research/01-Vision-Instagram-for-Research.md`.
- **"What phase are we in? What is next?"** Open `docs/walkthroughs/04-Next-Steps-and-Phase-Plan.md`.
- **"How does module X work?"** Open `docs/walkthroughs/02-Phase2-MultiInterest-Recommender.md` or `03-Code-Summary-and-Test-Plan.md`.
- **Conflict between docs?** Higher-priority doc wins. **Never average or merge contradictory guidance.**
- **The user references a doc by number** (e.g., "per doc 02") — read that doc but flag if 06 contradicts it before acting.

---

## 3. Non-negotiable rules (from doc 06)

These are the hard architectural commitments. **Violating any of these is a regression.** If a task seems to require violating one, stop and ask the user.

### 3.1 Fusion

- **Search uses RRF.** (Different retrievers — dense + sparse — answering the same query. RRF is correct here. Search is currently arXiv keyword API but will become hybrid semantic search in Phase 3.)
- **Zilliz collection schema** for Phase 3: collection `arxiv_bgem3_sparse`, fields: `id` (INT64, auto_id PK), `arxiv_id` (VARCHAR), `sparse_vector` (SPARSE_FLOAT_VECTOR). Index: SPARSE_INVERTED_INDEX, metric_type=IP. Sparse format uses **integer token IDs** as keys (from BGE-M3 tokenizer), NOT string words. Example: `{29: 0.0427, 6083: 0.1852, ...}`.
- **Recommendations use importance-weighted quota with a floor.** (Different queries — K medoid queries — over the same user. RRF would let the dominant cluster dominate; quota preserves minor interests.)
- **Never use RRF to merge multi-medoid recommendation results.** This is the most common mistake to avoid in this codebase.
- **Current status:** Recommendations use per-cluster quota fusion in `app/recommend/fusion.py` and `app/routers/recommendations.py`. `multi_interest_search()` remains only as a legacy helper; do not use it for new recommendation code.

Quota formula:
```
w_k = importance_k / sum(importance_k)
slot_k = max(floor(F * w_k), F_min)   # F = feed size, F_min = 3
# distribute remainder by largest fractional part
```

### 3.2 EWMA decay parameters

- `alpha_long = 0.03` — lives in `app/recommend/profiles.py` as `ALPHA_LONG_TERM`
- `alpha_short = 0.40` — lives in `app/recommend/profiles.py` as `ALPHA_SHORT_TERM`
- `alpha_neg = 0.15` — lives in `app/recommend/profiles.py` as `ALPHA_NEGATIVE`

If you find `alpha_long = 0.10` anywhere in code or config, it is a bug from doc 03. Fix it and reference doc 06 in the commit message.

### 3.3 Clustering

- Algorithm: **Ward hierarchical agglomerative** via `scipy.cluster.hierarchy.ward`.
- Code lives in: `app/recommend/clustering.py`.
- **L2-normalize embeddings BEFORE Ward, then use Euclidean distance.** Cosine Ward via sklearn is mathematically not Ward (Murtagh and Legendre 2014). L2-norm + Euclidean is monotonically equivalent to cosine and gives the intended behavior. This normalization is already in the code.
- **No fixed K.** Cut the dendrogram by adaptive gap-based threshold (see `_adaptive_threshold()`). Cap at `K_max = 7` (currently; doc 06 says `K_max = 20` for heavy users — raise this when users exist).
- **Medoid, not centroid.** Medoid = arg min over cluster members of sum of squared distances. Cache medoid paper IDs. This is implemented in `_find_medoid()`.
- **Hungarian-match cluster IDs across reclusterings** — NOT YET IMPLEMENTED. Planned for Phase 4.
- Recompute on each feed request currently (not nightly batch — no batch job infrastructure yet).

### 3.4 Reranking

- Terminal CPU-path reranker: **LightGBM LambdaRank** in `app/recommend/reranker.py`, with `heuristic_score()` as a fallback when the model file is missing.
- **37-feature schema** (see `models/reranker-phase6/production_model/feature_schema.json`):
  - Features 0-19: Content/retrieval (Qdrant score, citations, age, categories)
  - Features 20-22: EWMA similarities (longterm, shortterm, negative)
  - Features 23-30: User behavior (cluster importance, suppression, onboarding match, save/dismiss counts)
  - Features 31-36: Cross features (cosine×recency, cosine×citations, etc.)
- The caller in `recommendations.py` passes **all 37 features** including per-candidate cluster importance and medoid distances (Phase 6.1+6.2).
- The heuristic fallback uses features 0, 6, 20-22, 35.
- Weight budget (heuristic only): `0.40 * lt + 0.25 * st + 0.15 * recency + 0.10 * position - 0.15 * negative_penalty`.
- **Do NOT put `BGE-reranker-v2-m3` in the serving path.** ~8ms per pair on CPU = ~800ms for 100 pairs. Far over the 30ms budget.
- If a cross-encoder signal is wanted: distill BGE-reranker-v2 offline into a TinyBERT-L2 student (FlashRank recipe) and use the student score as a LightGBM feature on top-20. Phase 8.
- **Model trained on citation pseudo-labels, NOT real user signal.** Features 23-30 were zero during training. Retraining is deferred to Phase 6.4 (100 users or synthetic simulator).
- Health check: `GET /healthz/reranker` → reports `model_loaded`, `n_trees`, `feature_schema_hash`.

### 3.5 Diversity

- MMR with `lambda = 0.6` over the merged feed, on BGE-M3 embeddings. Code in `app/recommend/diversity.py` via `mmr_rerank()`.
- Exploration injection: 2 serendipitous papers per feed. Code in `app/recommend/diversity.py` via `inject_exploration()`.
- Quota (3.1) handles cross-cluster diversity. MMR handles within-quota redundancy.
- Do NOT use DPPs in v1.

### 3.6 Cold start / onboarding (the hybrid verdict)

IMPLEMENTED (Phase 5 core flow). ORCID / Scholar import is still pending. The right onboarding is **three-layer hybrid**:

1. arXiv category multi-select — used as a **filter and LightGBM feature**, NOT as the primary user vector.
2. ORCID / Semantic Scholar / Google Scholar author import — ingest authored paper embeddings as initial seeds. (NOT YET)
3. "Add 5 seed papers" library seeder — explicit user-chosen seeds. (DONE)
4. Fallback: popularity-per-selected-category feed for first session if user skips all three. (DONE)

Behavioral takes over once the user crosses **~10 saved papers**. Subject categories remain a feature/filter forever, never the primary vector.

### 3.7 Negative signals

The negative EWMA profile IS wired into reranking (Feature 5 in `reranker.py`). The full three-layer system described in Doc 06 is partially implemented:

1. **Session hard filter** — never re-show dismissed items (`seen` set in `recommendations.py`). DONE.
2. **Short-term item penalty** at rerank: `score -= alpha * exp(-dt / tau_neg)` — NOT YET (needs per-item decay tracking).
3. **Long-term EWMA negative profile** — wired as Feature 5 with 0.15 penalty weight. DONE.
4. **Category-level suppression** — DONE (db category suppression + rec filter).
5. **LightGBM dismissal labels** — NOT YET (Phase 6, needs 10K+ dismissals).

### 3.8 Latency budget

End-to-end feed generation target: **<30ms on CPU** (excluding metadata fetch, which is I/O-bound). Approximate budget per stage:
- Qdrant queries (3 medoids, parallel): ~10ms
- Heuristic rerank (LightGBM later): ~1ms
- MMR over union: ~2ms
- Quota + dedup: <1ms
- Negative-profile penalty: <1ms
- Headroom: ~15ms

**Note:** Metadata fetching from arXiv API is now largely avoided via the Turso metadata DB (Phase 3.5). arXiv remains a fallback for missing IDs.

### 3.9 ArXiv ID integrity

ArXiv IDs can have leading zeros (e.g., `0704.0001`). **Treat all arXiv IDs as strings, never integers.** Pandas will silently coerce them — always pass `dtype=str` to `read_csv`. This is a real bug that has bitten this project before.

### 3.10 Per-candidate cluster identity (Phase 6)

The per-cluster origin of each retrieved candidate is preserved end-to-end via `paper_cluster_map: dict[str, int]` (built in `recommendations.py` before `merge_quota_results()`). This mapping flows through to the reranker as per-candidate `cluster_importance` (N,) and `cluster_medoid` (N, 1024) arrays. **Do not re-introduce dominant-cluster shortcuts as "simplifications"** — LightGBM feature slot 24 (`cluster_distance_to_medoid`) depends on per-candidate medoids to correctly score papers from minority-interest clusters.

### 3.11 Interaction instrumentation invariants (Phase 6.5)

Every interaction logged via `db.log_interaction()` must carry **`query_id`**, **`propensity`**, and **`policy_id`**. These are required for Phase 7 evaluation:
- `query_id` (UUID): links all papers in a single feed request for per-feed CTR.
- `propensity` (float): probability the serving policy chose to show this paper (1.0 for deterministic, `n_explore/pool_size` for exploration).
- `policy_id` (string): identifies the pipeline version (`_RANKER_VERSION`).

**When adding a new recommendation tier or call path**, always include these three fields in the `paper_tags` dict. The round-trip is: `recommendations.py` → paper dict → `action_buttons.html` `hx-vals` → `events.py` Form params → `db.log_interaction()`.

---

## 4. What is in scope vs out of scope right now

**Current phase: Phase 6.5 COMPLETE; Phase 7 (Evaluation Framework) next.** Phase 2 (a, b, c) is complete with Doc 06 corrections applied. Phase 3 (Hybrid Semantic Search) and Phase 3.5 (Turso metadata DB) are implemented and tested.

**What has been built (Phases 1-2c):**
- Qdrant BEST_SCORE recommend API (Tier 3 fallback)
- EWMA profiles (long/short/negative, alpha corrected)
- Ward clustering with L2-norm + adaptive threshold + medoids
- Prefetch+RRF retrieval (Tier 1, will be replaced with quota in Phase 4)
- EWMA vector search (Tier 2 fallback)
- 5-feature heuristic reranker (with negative penalty)
- MMR diversity + exploration injection
- 3-tier cascading pipeline (5+ saves, 3+ saves, 1+ save)
- 88 tests passing

**Phase 3 — implemented (Hybrid Semantic Search):**
*See `docs/TASK-TRACKER.md` Phase 3 section for full details.*
- `app/embed_svc.py` — BGE-M3 model singleton (lazy load, LRU cache, CPU float32)
- `app/zilliz_svc.py` — Zilliz sparse search client (gRPC reconnect, graceful fallback)
- `app/groq_svc.py` — LLM query rewriter (2s timeout, academic heuristic, unconditional fallback)
- `app/hybrid_search_svc.py` — Orchestrator (rewrite → encode → parallel search → RRF → recency rerank)
- Swapped `app/routers/search.py` to use hybrid pipeline, with arXiv API fallback
- `Dockerfile` + `.dockerignore` — HF Spaces deployment (Docker SDK, port 7860)
- 21 new tests passing, 109 total (zero regressions)

**Phase 4 — recommendation fixes (complete):**
- Replace RRF with importance-weighted quota fusion
- Hungarian matching for cluster stability
- Category-level suppression in recommendations

**Phase 5 — cold-start onboarding + UI (complete):**
- Onboarding wizard (category multi-select + seed search)
- Category-filtered trending fallback
- Dark-mode base UI + updated paper cards
- S2/ORCID author import was explored and **removed** — not the direction we want

**Phase 6 — LightGBM reranker (COMPLETE ✅):**
- LightGBM LambdaRank (141 trees, 37 features) integrated with heuristic fallback
- Phase 6.1+6.2: All 37 features wired into caller (per-candidate cluster identity)
- Phase 6.3: `/healthz/reranker` endpoint, model accessors, feature logging
- Phase 6.3: Bug B fixed — medoid embedding persisted as BLOB fallback
- Model stored under `models/reranker-phase6/production_model/`
- HF model repo: `siddhm11/researchit-reranker-phase6`
- Phase 6.4 (retraining) deferred: gated on 100 users or synthetic simulator

**Out of scope until later phases — do not build:**
- S2/ORCID author import for onboarding (removed — not the direction we want).
- Collaborative filtering / LightFM (Phase 9, 500+ users).
- Cross-encoder reranking in serving path (never; only distilled — Phase 8).
- Claude/Groq-generated cluster summaries (Phase 8).
- Epsilon-greedy exploration beyond the current simple stub (Phase 9).
- DPPs, Semantic IDs, TIGER, PinnerFormer-style single-vector models (Phase 9+, only if scale warrants).
- Migration to Supabase (until 10+ concurrent writes/sec observed).
- React SPA (explicitly ruled out — stick with HTMX + Jinja2).
- Redis (explicitly ruled out — in-process caches are fine at this scale).
- Real-time streaming (explicitly ruled out).
- Custom embedding fine-tuning (explicitly ruled out — BGE-M3 is frozen).

If a request asks for one of these, surface that it is out of scope per doc 06 phase plan, then ask whether to proceed anyway or defer.

---

## 5. Workflow rules for the agent

### 5.1 Before doing anything

- Read this file (`CLAUDE.md`).
- If the task touches recommendation logic, also read `docs/research/06-Deep-Research-Verdict.md`.
- If the task touches a specific component, load the relevant doc per section 2.
- Check existing code in the affected module before writing. Do not duplicate utilities.

### 5.2 When to ask vs when to act

**Ask first** if any of these are true:
- The request would violate a section 3 rule.
- The request is ambiguous about which phase it belongs to.
- The request involves changing EWMA parameters, quota formulas, or cluster hyperparameters that exist in code with rationale comments.
- The request would add a new dependency not in `requirements.txt`.

**Act directly** for:
- Bug fixes with a clear repro.
- Adding tests.
- Refactors within a single module that do not change behavior.
- Implementing something explicitly described in doc 06.

### 5.3 Code style

- Python 3.12+. Type hints on all public functions.
- Async by default for FastAPI handlers. Sync is fine for batch scripts.
- No bare `except:`. Catch specific exceptions.
- Logging: use `print()` with `[module_name]` prefix (current convention). Will migrate to structured logging later.
- All embeddings are 1024-dim float32 (BGE-M3 dense). Normalize before storage/comparison.

### 5.4 Tests

- Every new function in `app/recommend/` gets a unit test.
- Every new endpoint gets at least one integration test.
- Use `pytest` + `pytest-asyncio` (asyncio_mode = auto, configured in `pytest.ini`).
- Test files go in `tests/`. No `tests/fixtures/` directory exists yet — inline fixtures or use `tmp_path`.
- Run tests: `python -m pytest tests/ -v`
- Run E2E: `python test_e2e_recs.py`

### 5.5 File and folder conventions

This is the **actual** project structure. Do not create directories that do not exist unless building a new phase component.

```
ResearchIT-Final/
|-- CLAUDE.md                    # THIS FILE — agent rulebook
|-- run.py                       # Dev server entry (python run.py)
|-- requirements.txt             # pip dependencies
|-- pytest.ini                   # pytest config (asyncio_mode=auto)
|-- interactions.db              # SQLite database (auto-created)
|-- test_e2e_recs.py             # E2E simulation test (standalone)
|
|-- app/                         # FastAPI application
|   |-- main.py                  # App entry, lifespan, router includes
|   |-- config.py                # Settings (QDRANT_URL, COOKIE_NAME, etc.)
|   |-- db.py                    # SQLite schema + async CRUD (aiosqlite)
|   |-- qdrant_svc.py            # Qdrant client: recommend, search_by_vector,
|   |                            #   get_paper_vectors, multi_interest_search
|   |-- arxiv_svc.py             # arXiv API search + metadata fetch + SQLite cache
|   |-- user_state.py            # In-memory user state (positive/negative deques)
|   |-- templates_env.py         # Jinja2 environment setup
|   |
|   |-- embed_svc.py             # BGE-M3 model singleton (Phase 3)
|   |-- groq_svc.py              # LLM query rewriter (Phase 3)
|   |-- hybrid_search_svc.py     # Hybrid search orchestrator (Phase 3)
|   |-- turso_svc.py             # Turso metadata client (Phase 3.5)
|   |-- zilliz_svc.py            # Zilliz sparse search client (Phase 3)
|   |-- routers/                 # FastAPI route handlers
|   |   |-- search.py            # GET /search — hybrid semantic search
|   |   |-- recommendations.py   # GET /api/recommendations — 3-tier cascade + quota
|   |   |-- events.py            # POST /api/save, /api/dismiss — triggers EWMA update
|   |   |-- saved.py             # GET /saved — user saved papers
|   |   |-- onboarding.py        # GET /onboarding — onboarding wizard
|   |   |-- health.py            # GET /healthz/reranker — model deployment status
|   |
|   |-- recommend/               # Recommendation engine (Phase 2/4/6)
|   |   |-- __init__.py          # Module docstring
|   |   |-- profiles.py          # EWMA profiles (long/short/negative)
|   |   |-- clustering.py        # Ward clustering + medoids + adaptive threshold
|   |   |-- fusion.py            # Quota fusion (Phase 4)
|   |   |-- reranker.py          # LightGBM reranker + heuristic fallback
|   |   |-- diversity.py         # MMR reranking + exploration injection
|   |
|   |-- static/                  # CSS, images
|   |-- templates/               # Jinja2 + HTMX templates
|       |-- base.html            # Base layout
|       |-- index.html           # Home page with recommendations
|       |-- search.html          # Search page
|       |-- onboarding.html      # Onboarding wizard
|       |-- partials/            # HTMX partial templates
|
|-- docs/                        # Documentation (see section 2 for precedence)
|   |-- README.md                # Master doc index with reading order
|   |-- TASK-TRACKER.md          # Master task checklist (all phases)
|   |-- research/                # Research documents (01-06)
|   |   |-- 01-Vision-Instagram-for-Research.md
|   |   |-- 02-Recommendation-System-Blueprint.md
|   |   |-- 03-MultiInterest-Recommender-Architecture.md  # Has addendum with corrections
|   |   |-- 04-Technical-Roadmap-Legacy.md
|   |   |-- 05-Evolution-Of-Onboarding-And-Interests.md
|   |   |-- 06-Deep-Research-Verdict.md                   # SOURCE OF TRUTH
|   |-- phases/
|   |   |-- PHASE1-Zero-ML-Recommender.md
|   |   |-- PHASE2-Hybrid-Search-Plan.md                  # Prototype reference
|   |   |-- PHASE3-Hybrid-Semantic-Search.md               # ACTIVE PHASE 3 PLAN
|   |-- walkthroughs/
|       |-- 01-Phase1-Code-Tour.md
|       |-- 02-Phase2-MultiInterest-Recommender.md
|       |-- 03-Code-Summary-and-Test-Plan.md
|       |-- 04-Next-Steps-and-Phase-Plan.md               # MASTER ROADMAP
|
|-- notebooks/                   # Kaggle/Jupyter notebooks (reference only)
|   |-- README.md                # Notebook index + extracted schema details
|   |-- 01-bme-upload.ipynb      # BGE-M3 encode + upload to Qdrant/Zilliz (1.6M papers)
|   |-- 02-bme-arxiv-test.ipynb  # Search quality tests + BGE-M3 prototype
|   |-- 03-check-search-bq-prm.ipynb  # BQ vs PRM quantization benchmark
|
|-- tests/                       # pytest test suite (203+ tests)
    |-- test_profiles.py         # EWMA profile tests (11)
    |-- test_clustering.py       # Ward clustering tests (10)
    |-- test_reranker_diversity.py # Reranker + MMR tests (13)
    |-- test_phase6_feature_wiring.py # Phase 6.1+6.2 feature wiring (9)
    |-- test_reranker_integration.py # Phase 6 LightGBM integration (7)
    |-- test_db.py               # SQLite schema tests
    |-- test_qdrant_svc.py       # Qdrant client tests
    |-- test_arxiv_svc.py        # arXiv service tests
    |-- test_integration.py      # Cross-module integration tests
    |-- test_user_state.py       # User state tests
    |-- test_saved.py            # Saved papers tests
```

**Modules planned for future phases:**
- None listed here yet. Add when new components are scoped.

### 5.6 Common commands

```bash
# Run the app (dev server with hot reload)
python run.py
# serves at http://127.0.0.1:7860 (port 7860 for HF Spaces compat)

# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_clustering.py -v

# Run E2E simulation (hits live Qdrant)
python test_e2e_recs.py

# Install dependencies
pip install -r requirements.txt
```

### 5.7 Commits

- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`.
- If a commit implements a decision from a doc, reference it: `feat(fusion): implement importance-weighted quota per doc 06`.
- Never commit secrets. Environment variables are read from `app/config.py` via `os.getenv()`.
- Phase 3 env vars: `ZILLIZ_URI`, `ZILLIZ_TOKEN`, `ZILLIZ_COLLECTION`, `GROQ_API_KEY`, `BGE_M3_MODEL`, `BGE_M3_DEVICE`. These are set in HF Spaces Secrets, not hardcoded.
- Qdrant env vars (already in config.py): `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`.

---

## 6. How to update the docs

Architecture evolves. The agent will sometimes encounter decisions that need to be amended. Follow this protocol.

### 6.1 The golden rule

**Doc 06 is the primary architectural reference.** Docs 01-05 are historical. If a new decision contradicts 03/02/01/04/05, do not edit those — instead, either update doc 06 changelog or create a new doc (07+).

**Exception:** Doc 03 has an "Addendum" section at the bottom specifically for recording corrections from Doc 06. This addendum can be updated when new corrections are applied.

### 6.2 When the user makes a new architectural decision

1. Append a dated entry to the bottom of `docs/research/06-Deep-Research-Verdict.md` under a `## Changelog` section.
2. Format:
   ```
   ### YYYY-MM-DD — [short title]
   **Decision:** [the new rule]
   **Supersedes:** [which earlier doc/section, if any]
   **Rationale:** [why — 1-3 sentences]
   **Action items:** [what code changes are implied]
   ```
3. If the decision invalidates a section 3 rule in *this* `CLAUDE.md` file, also update section 3 to match.
4. Update the correction summary table in Doc 03 addendum if applicable.
5. Mention in the user-facing reply: "Logged in doc 06 changelog and updated CLAUDE.md."

### 6.3 When you discover a contradiction the user has not resolved

Do not silently pick a side. Surface it: "Doc 03 says X, doc 06 says Y, the code does Z. Which should I follow?" Then act on the answer and log per 6.2.

### 6.4 Editing other docs

You may edit docs 01-05 only to:
- Fix a typo.
- Update the addendum in Doc 03.
- Add a banner noting it is superseded.
- Nothing else. No content edits, no architectural revisions.

### 6.5 New docs

If a topic is too large for a 06 changelog entry, create `docs/research/07-[topic].md` and add it to the section 2 table in this file with priority and use/don't-use guidance. Do not create new docs without prompting from the user.

---

## 7. Quick reference card

| Question | Answer |
|---|---|
| Source of truth? | `docs/research/06-Deep-Research-Verdict.md` |
| Master roadmap? | `docs/walkthroughs/04-Next-Steps-and-Phase-Plan.md` |
| Recommendation fusion? | Importance-weighted quota with `F_min=3` (Phase 4 complete). |
| Search fusion? | RRF (hybrid search in Phase 3). |
| alpha_long? | `0.03` — in `app/recommend/profiles.py` |
| alpha_short? | `0.40` — in `app/recommend/profiles.py` |
| alpha_neg? | `0.15` — in `app/recommend/profiles.py` |
| MMR lambda? | `0.6` — in `app/recommend/diversity.py` |
| Cluster algorithm? | Ward, L2-normalized, Euclidean, adaptive gap threshold, `K_max=7`. In `app/recommend/clustering.py`. |
| Reranker? | LightGBM lambdarank with heuristic fallback (Phase 6). |
| Latency budget? | <30ms end-to-end (compute only; metadata I/O excluded). |
| Cold start? | Hybrid: categories + seed papers + popularity fallback (Phase 5 complete). ORCID/Scholar import pending. |
| When does behavioral take over? | ~10 saved papers. Currently activates at 5 (clustering) / 3 (EWMA) / 1 (BEST_SCORE). |
| When to add CF? | 500+ users (Phase 9). |
| Current phase? | **Phase 6 COMPLETE.** Phase 7 (evaluation) next. See `docs/TASK-TRACKER.md`. |
| ArXiv ID type? | String. Always. `dtype=str` in pandas. |
| Embedding model? | BAAI/bge-m3, 1024-dim dense + sparse lexical weights. Loaded at startup in `app/embed_svc.py`. Graceful fallback if not installed. |
| How to run? | `python run.py` at http://127.0.0.1:7860 (port 7860 for HF Spaces compat) |
| How to test? | `python -m pytest tests/ -v` |
| Storage? | SQLite (`interactions.db`) — ephemeral on HF Spaces. Supabase at 10+ concurrent writes/sec. |
| Deployment? | Hugging Face Spaces (Docker SDK, 16GB RAM, 2 vCPUs). Render abandoned (512MB too small for BGE-M3). |
| Forbidden in v1? | Redis, React SPA, real-time streaming, custom embedding fine-tuning, cross-encoder in hot path, DPPs, generative retrieval. |

---

*Last updated: 2026-05-05. Update this date when CLAUDE.md changes.*
