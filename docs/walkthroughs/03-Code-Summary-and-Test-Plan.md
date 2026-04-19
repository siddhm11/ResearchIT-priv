# Codebase Summary & Testing Plan

This document provides a concise summary of the codebase's current state (Phases 1 & 2) and outlines a comprehensive testing plan to ensure reliability, accuracy, and performance in production.

---

## 1. What Code is Written Till Now

The current application is a fully functional FastAPI + HTMX research paper discovery platform powered by BGE-M3 embeddings, Qdrant vector search, and a multi-interest recommendation engine. 

### 🏗️ Backend Core
- **`app/main.py`**: The FastAPI application entrypoint. Configures routing, CORS, and exception handling.
- **`app/config.py`**: Pydantic configuration loading environment variables (Qdrant URLs, API keys, tuning parameters).
- **`app/db.py`**: Lightweight SQLite (WAL mode) interface via `aiosqlite`. Manages schemas for `events` (interaction tracking), `arxiv_cache` (metadata), `user_profiles` (EWMA vectors), and `user_clusters`.
- **`app/user_state.py`**: In-memory cache using Python `deque` to store hot interaction paths (latest 50 positive/20 negative IDs) per user for extremely fast lookups.

### 🧠 Recommendation Engine (`app/recommend/`)
- **`profiles.py`**: Computes exponential moving averages (EWMA) to create Long-Term (α=0.10), Short-Term (α=0.40), and Negative (α=0.15) semantic profile vectors for each user.
- **`clustering.py`**: Implements Ward hierarchical clustering to identify multiple distinct interest areas (up to 7) based on a user's liked papers. Extracts actual paper embeddings as cluster medoids to prevent "topic drift."
- **`reranker.py`**: Heuristic scoring system combining long-term relevance (45%), session context (25%), paper recency (20%), and RRF retrieval rank (10%). Includes the feature extraction pipeline designed for a future drop-in LightGBM upgrade.
- **`diversity.py`**: Implements Maximal Marginal Relevance (MMR) with λ=0.6 to ensure top recommendations are diverse, plus an exploration injector to randomly add 1-2 serendipitous papers to break filter bubbles.

### 🔌 External Services
- **`app/qdrant_svc.py`**: Communicates with the Qdrant vector database. Handles `BEST_SCORE` recommendations, vector fetching, dense search, and the new Phase 2 **Prefetch + Reciprocal Rank Fusion (RRF)** parallel search.
- **`app/arxiv_svc.py`**: Fetches fresh metadata via the public arXiv HTTP API and caches it in SQLite to reduce network calls.

### 🌐 API Routers (`app/routers/`)
- **`recommendations.py`**: Implements the 3-tier cascading fallback pipeline (Multi-interest clustering → Single EWMA vector → Raw ID Qdrant Recommend API). Returns HTMX-rendered partials.
- **`events.py`**: Handles user interactions (`/save`, `/not-interested`). Fires background `asyncio` tasks to recalculate EWMA user profiles asynchronously.
- **`search.py` & `saved.py`**: Handle explicit user queries and listing saved papers. 

### 🎨 Frontend (`app/templates/`)
- Pure HTML + HTMX frontend utilizing Jinja2 templating. Uses TailwindCSS/DaisyUI for UI components without requiring a Node.js build step.

---

## 2. Comprehensive Testing Plan

The current test suite has **86 passing tests** executing via `pytest`. Our testing strategy is split into three layers: Automated, Manual, and Analytics-based evaluation.

### A. Automated Testing (Current & Ongoing)

#### 1. Unit Tests (Logic Verification)
- **Math & Vectors (`test_profiles.py`, `test_clustering.py`)**: 
  - Ensure EWMA updates decay exactly as expected over simulated interaction horizons.
  - Verify Ward clustering correctly bins distantly separated vectors and correctly assigns the medoid. 
  - Verify L2-normalization constraints are never violated.
- **Algorithms (`test_reranker_diversity.py`)**: 
  - MMR testing matching Edge constraints (e.g., handles clusters gracefully, handles `len(candidates) < K`).
  - Heuristic scorer scoring functions evaluate recency dates safely (no crashing on missing/malformed `published` keys).

#### 2. Integration Tests (System Wiring)
- **Database (`test_db.py`)**: Ensure SQLite writes do not lock the DB, caching functions hit the DB when missing from memory, and `user_clusters` persist complex JSON payloads.
- **Endpoints (`test_integration.py` / `test_routers.py`)**: 
  - Issue standard `TestClient` API requests.
  - Verify `/save` successfully launches the background EWMA calculation before concluding the request.
  - Verify `/api/recommendations` gracefully steps down the 3-Tier fallback logic if vectors are absent.

#### 3. Service Mocks vs Live E2E
- **Mocked Qdrant**: 95% of tests mock Qdrant to ensure fast, deterministic offline execution.
- **Live Qdrant Pipeline**: 2 pre-existing tests hit the live Qdrant payload via `test_qdrant_svc.py`. (Currently network-dependent; any failures here usually indicate transient timeouts rather than logic drops).

### B. Manual QA & UX Flow Verification

Before pushing to production branches, undergo manual browser-based verification:
1. **Cold Start Flow**: Load application as a new user (cleared cookies). Verify that recommendations inform the user to "Save at least 1 paper."
2. **Phase 1 Tier Check**: Search for "Transformers" and save 1 paper. Verify that recommendations populate via Qdrant's raw-ID API.
3. **Phase 2a/b Tier Check**: Save 5 distinct papers across 2 distinct topics (e.g., 3 LLM papers, 2 quantum computing papers). Reload the application and inspect logs to verify **Ward Clustering** kicked in and the Qdrant Multi-Interest Prefetch query was logged.
4. **Resiliency**: Quickly mash the "Save" and "Not Interested" buttons on the UI to test visual HTMX snappiness and ensure no `sqlite3.OperationalError: database is locked` occurs under rapid background event firing.

### C. Evaluation & Production Metrics (Phase 4 Validation)

Once users enter the system, the platform shifts from unit-test verification to **Data Science verification**.

#### 1. Offline Evaluation (Historical Split)
Run a script simulating user interactions using the time-split method (train on behavior before day X, test on day X+1):
- **NDCG@10**: Evaluates how efficiently we rank papers the user ultimately saves/clicks.
- **Hit Rate@10**: Verifies what percentage of users successfully interacted with at least 1 recommendation per session.
- **Coverage**: Evaluates if the recommendation queue is pulling from a wide diversity of our 1.6M paper collection, or if we are stuck recommending the same 100 benchmark papers.

#### 2. Online Telemetry (Production UX)
- **CTR (Click-Through-Rate)**: Measure the ratio of recommendation views vs clicks. (Target: 2-5%).
- **Save Rate**: The definitive proxy for success. (Target: 1-2%).
- **Dwell Time Analysis**: Monitor if clicked recommendations result in > 30-second reading sessions, discounting bounce-clicks.

---

### Moving Forward
With the multi-interest foundation established, the immediate next focus is upgrading the explicit search bar (Phase 2's Planned Hybrid Search Plan using Zilliz Sparse tracking) and observing cluster calculation stability on Render.
