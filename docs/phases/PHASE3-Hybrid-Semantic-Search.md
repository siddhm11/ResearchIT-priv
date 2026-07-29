# Phase 3 — Hybrid Semantic Search

> **Purpose**: Replace the Phase 1 placeholder arXiv keyword API search with real vector-based
> semantic search using BGE-M3 encoding + Qdrant dense + Zilliz sparse + RRF fusion.
>
> **Status**: ✅ Complete
> **Estimated effort**: ~2-3 weeks  
> **Predecessor**: Phase 2c (complete) — the recommendation pipeline  
> **Deployment target**: Hugging Face Spaces (Docker SDK, free tier: 16GB RAM, 2 vCPUs)

---

## Why This Is The #1 Priority

The entire reason we built 1.6M BGE-M3 embeddings across Qdrant Cloud (dense, 1024-dim) and
Zilliz Cloud (sparse, learned lexical weights) is to power semantic search. Right now, the
search bar calls the arXiv keyword API — a Phase 1 throwaway placeholder that:

- Can't understand meaning, only matches exact words
- "when AI makes up fake facts" returns nothing useful
- Misses papers using different terminology for the same concept
- Has no ranking signal beyond arXiv's own relevance score

With hybrid search, that same query would be rewritten by an LLM to
"LLM hallucination factual errors sycophancy truthfulness survey", encoded by BGE-M3 into
both dense and sparse vectors, and matched against 1.6M papers semantically.

**Doc 06 confirms this is load-bearing infrastructure:**
> "Phase 0 — already complete: hybrid BGE-M3 dense + sparse + RRF on Qdrant + Zilliz, 1.6M
> papers. Keep RRF *for search* (it's correct for fusing different retrievers over the *same*
> query); replace it with quota *for recommendations* (different queries over the same user)."

**RRF is correct for search** — this is fusing different retrievers (dense + sparse) answering
the same query. This is fundamentally different from recommendations, where RRF is wrong.

---

## Architecture: What Changes From Phase 1

### Phase 1 Search (current — being replaced)

```
User query (exact string)
      │
      ▼
arXiv keyword API  (https://export.arxiv.org/api/query?search_query=all:query)
      │  standard text match, not semantic
      ▼
SQLite metadata cache → render paper cards
```

**Problems**: Keyword-only. No semantic understanding. Depends on external API.

### Phase 3 Search (target)

```
User types: "when AI makes up fake facts"
      │
      ▼
[1] LLM Rewriter (Groq / llama-3.3-70b-versatile, ~300ms)
      │   → "LLM hallucination factual errors sycophancy truthfulness survey"
      │   Falls back to original query on error/timeout
      ▼
[2] BGE-M3 Encode (CPU, ~300ms first / ~0ms cached)
      │   Single forward pass produces TWO outputs:
      │   ├── dense_vec  : float32[1024]        — semantic meaning
      │   └── sparse_dict: {token_id: weight}   — lexical weights (NOT BM25)
      ▼
[3a] Qdrant dense search ───────┐
      │  HNSW ANN on 1024-dim    │
      │  collection: arxiv_bgem3_dense  │    PARALLEL
      │  returns: [(arxiv_id, score)]   │    (~300ms total)
                                 │
[3b] Zilliz sparse search ──────┘
      │  IP on sparse vectors
      │  collection: arxiv_bgem3_sparse
      │  returns: [(arxiv_id, score)]
      │
      ▼
[4] RRF Fusion (K=60)
      │  score[paper] = 1/(60+rank_dense) + 1/(60+rank_sparse)
      │  Pure rank-based — no score normalization needed
      ▼
[5] Rerank: 0.80 × norm_rrf + 0.20 × recency
      │  (Citation signal deferred — not available yet)
      ▼
[6] Return arxiv_ids → fetch metadata → render cards
```

### What Stays Unchanged

These are intentionally NOT changed in Phase 3:

1. **Recommendations pipeline** — Tiers 1/2/3 in `recommendations.py`. No dependency on BGE-M3.
2. **User state** — `user_state.py` deques + SQLite interactions. Source-agnostic.
3. **Metadata fetching** — `arxiv_svc.fetch_metadata_batch()`. Still needed for titles/abstracts.
4. **Event logging** — `db.log_interaction()`. Events are source-agnostic.
5. **All templates** — `paper_card.html`, `action_buttons.html`, etc. Same paper dict format.
6. **HTMX patterns** — save/dismiss flow, lazy-load recs, search-as-you-type.

---

## The Critical BGE-M3 Insight

BGE-M3 is a **dual-encoder**: one forward pass produces both a dense vector AND sparse
lexical weights simultaneously. The sparse output is NOT BM25 — it is BGE-M3's own learned
sparse representation (`lexical_weights` from FlagEmbedding).

**This has a critical consequence**: the Zilliz collection `arxiv_bgem3_sparse` was indexed
with BGE-M3 sparse outputs. Query-time sparse encoding **must** also use BGE-M3's sparse
encoder. You CANNOT substitute a BM25 tokenizer. The model must be loaded and run for every
search.

```python
# One call, two outputs:
out = model.encode(
    [text],
    return_dense=True,
    return_sparse=True,        # ← BGE-M3 lexical weights, not BM25
    return_colbert_vecs=False,
    max_length=512,
)
dense  = out["dense_vecs"][0]          # shape (1024,) float32
sparse = out["lexical_weights"][0]     # dict {token_id: float}
```

---

## Deployment: Hugging Face Spaces (Docker SDK)

### Why HF Spaces Instead of Render

| Constraint | Render Free | HF Spaces Free | Verdict |
|---|---|---|---|
| **RAM** | 512 MB | **16 GB** | BGE-M3 needs ~2GB (model + PyTorch runtime). Render can't do it. |
| **CPU** | Limited | 2 vCPUs | Sufficient for BGE-M3 CPU inference (~300ms/query) |
| **Disk** | Persistent | **Ephemeral** (50GB) | Need external DB for persistence → we already use Qdrant Cloud + Zilliz Cloud. SQLite needs a solution. |
| **Sleep** | After 15 min | After ~2 days | Better for a research tool |
| **Port** | Any | **7860** (required) | Must configure in run.py |
| **Cold start** | ~30-60s | ~15-30s + model download | Model caching via Docker layers helps |

### HF Spaces Constraints to Handle

1. **Ephemeral filesystem** — `interactions.db` (SQLite) data is lost on restart.
   - **Solution**: For now, accept this (pre-launch, no real users). Phase 4 can migrate to
     Supabase/external DB when persistence matters.
   - Alternative: Use HF Dataset repo as persistent store via `huggingface_hub` library.

2. **Port must be 7860** — HF Spaces requires apps to listen on port 7860.
   - **Solution**: Change `run.py` to use port 7860 (or read from `PORT` env var).

3. **Model download on cold start** — BGE-M3 (~570MB) downloads from HuggingFace Hub on first
   start. Subsequent starts use the Docker layer cache.
   - **Solution**: Download model in Dockerfile `RUN` step so it's baked into the image.

4. **Non-root user** — HF Spaces Docker runs as user ID 1000.
   - **Solution**: Add `USER 1000` in Dockerfile, ensure all paths are writable.

### Dockerfile Skeleton

```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Set up app directory
WORKDIR /app

# Install Python deps (torch CPU-only first for smaller image)
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download BGE-M3 model into the image (baked in, no cold-start download)
RUN python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3')"

# Copy application code
COPY . .

# HF Spaces requires port 7860 and non-root user
USER 1000
EXPOSE 7860

CMD ["python", "run.py"]
```

---

## New Files to Create

### `app/embed_svc.py` — BGE-M3 Model Singleton

```
Responsibilities:
  - Load BAAI/bge-m3 once at startup via lifespan
  - encode_query(text) → (dense: np.ndarray[1024], sparse: dict[int, float])
  - LRU cache (128 entries) on query text to avoid re-encoding repeats
  - CPU float32, no GPU dependency
  - use_fp16=False on CPU (fp16 is GPU-only)

Key API:
  get_model() → BGEM3FlagModel           # lazy singleton
  encode_query(text) → (ndarray, dict)    # cached, thread-safe
```

**Why a singleton**: BGE-M3 is ~570MB in memory. Loading it twice would waste RAM.
The model is loaded once at startup (or lazily on first query) and reused for all requests.

### `app/zilliz_svc.py` — Zilliz Cloud Sparse Search Client

```
Responsibilities:
  - Connect to Zilliz Cloud serverless via pymilvus MilvusClient
  - search_sparse(sparse_dict, limit) → list[dict] with arxiv_id + score
  - Handle gRPC reconnects (closed-channel error observed in prototype)
  - Collection: arxiv_bgem3_sparse
  - Schema: id (INT64 auto PK), arxiv_id (VARCHAR), sparse_vector (SPARSE_FLOAT_VECTOR)
  - Index: SPARSE_INVERTED_INDEX, metric_type=IP
  - Sparse format: {int_token_id: float_weight} (NOT string words)
  - Metric: IP (Inner Product)

Key API:
  search_sparse(sparse_dict, limit=50) → list[dict]
```

**Config needed**: `ZILLIZ_URI`, `ZILLIZ_TOKEN`, `ZILLIZ_COLLECTION` in config.py.

### `app/groq_svc.py` — LLM Query Rewriter

```
Responsibilities:
  - Groq API client (lazy init, reads GROQ_API_KEY from env)
  - rewrite(user_query) → academic_query_string
  - Uses llama-3.3-70b-versatile with few-shot prompt
  - Falls back to original query on ANY error or timeout (>2s)
  - Optional: skip rewriting for queries that already look academic

Key API:
  rewrite(query) → str    # graceful fallback, never crashes
```

**The rewrite is an enhancement, NOT a dependency.** If Groq is down, the system works
fine with the original query.

### `app/hybrid_search_svc.py` — Search Orchestrator

```
Responsibilities:
  - Orchestrates the full pipeline: rewrite → encode → parallel search → RRF → rerank
  - Calls groq_svc.rewrite() (optional, can be skipped)
  - Calls embed_svc.encode_query()
  - Calls qdrant_svc.search_dense() + zilliz_svc.search_sparse() in parallel
    via asyncio.gather with run_in_executor
  - RRF merge (K=60)
  - Recency rerank: 0.80 × norm_rrf + 0.20 × recency
  - Returns list of arxiv_ids, sorted by final score

Key API:
  search(query, limit=10) → list[str]   # returns arxiv_ids
```

---

## Files to Modify

### `app/config.py` — Add Search Config

```python
# ── Zilliz Cloud (BGE-M3 sparse) ─────────────────────────────────────────────
ZILLIZ_URI        = os.getenv("ZILLIZ_URI", "https://in03-...")
ZILLIZ_TOKEN      = os.getenv("ZILLIZ_TOKEN", "...")
ZILLIZ_COLLECTION = os.getenv("ZILLIZ_COLLECTION", "arxiv_bgem3_sparse")

# ── Groq (LLM query rewriter) ────────────────────────────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")

# ── BGE-M3 (embedding model) ─────────────────────────────────────────────────
BGE_M3_MODEL      = os.getenv("BGE_M3_MODEL", "BAAI/bge-m3")
BGE_M3_DEVICE     = os.getenv("BGE_M3_DEVICE", "cpu")
ENCODE_CACHE_SIZE = 128

# ── Hybrid search tuning ─────────────────────────────────────────────────────
SEARCH_RRF_K             = 60     # RRF denominator
SEARCH_FETCH_K_MULTIPLIER = 6    # candidates = top_k × 6 before rerank
SEARCH_SEMANTIC_WEIGHT   = 0.80   # RRF contribution to final score
SEARCH_RECENCY_WEIGHT    = 0.20   # recency contribution to final score

# ── Deployment ────────────────────────────────────────────────────────────────
APP_PORT = int(os.getenv("PORT", "7860"))   # HF Spaces requires 7860
```

### `app/qdrant_svc.py` — Add `search_dense()`

A new function for raw vector search (different from `search_by_vector()` which is used by
the recommendation pipeline). `search_dense()` returns score + arxiv_id tuples needed for RRF.

```python
async def search_dense(
    dense_vec: list[float],
    limit: int = 50,
) -> list[dict]:
    """
    ANN dense search for the search pipeline. Returns list of
    {'arxiv_id': str, 'score': float} dicts sorted by score desc.

    Different from search_by_vector() which returns only arxiv_ids.
    This version returns scores needed for RRF fusion.
    """
```

### `app/routers/search.py` — Swap Search Backend

Replace `arxiv_svc.search(q)` with `hybrid_search_svc.search(q)` +
`arxiv_svc.fetch_metadata_batch(arxiv_ids)`.

The router signature, template rendering, and response format stay IDENTICAL.
This is a one-function swap inside the router.

```python
# BEFORE (Phase 1):
papers = await arxiv_svc.search(q.strip())

# AFTER (Phase 3):
from app import hybrid_search_svc
arxiv_ids = await hybrid_search_svc.search(q.strip(), limit=config.ARXIV_MAX_RESULTS)
meta = await arxiv_svc.fetch_metadata_batch(arxiv_ids)
papers = [meta[aid] for aid in arxiv_ids if aid in meta]
```

### `app/main.py` — Add Model Warm-up to Lifespan

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    # Warm up BGE-M3 at startup, not on first request
    from app import embed_svc
    embed_svc.get_model()
    print("[main] BGE-M3 model loaded")
    yield
```

### `run.py` — Use Configurable Port

```python
import uvicorn
from app.config import APP_PORT

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=APP_PORT, reload=True)
```

### `requirements.txt` — Add Dependencies

```
# Existing
fastapi>=0.115
uvicorn>=0.30
jinja2>=3.1
httpx>=0.27
aiosqlite>=0.20
qdrant-client>=1.9
pydantic>=2.0
numpy>=1.24
scipy>=1.11
pytest>=8.0
pytest-asyncio>=0.23
anyio[asyncio]

# Phase 3 additions
FlagEmbedding>=1.2.9       # BGE-M3 model
pymilvus>=2.4              # Zilliz/Milvus client
groq>=0.9                  # Groq LLM API
```

Note: `torch` is installed separately with CPU-only wheels:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Implementation Order

Each step leaves the app in a working state. Order minimizes integration risk.

### Step 1 — BGE-M3 Model Service (`app/embed_svc.py`)

Build the embedding service in isolation. No app integration yet.

- Load `BAAI/bge-m3` with `use_fp16=False` (CPU)
- `encode_query(text)` → `(dense_vec, sparse_dict)` with LRU cache
- Thread-safe (model inference is CPU-bound, use `run_in_executor`)

**Test**: Encode "attention is all you need". Verify dense shape is `(1024,)`,
sparse dict has >5 keys, all values are floats.

### Step 2 — Zilliz Client (`app/zilliz_svc.py`)

Connect to `arxiv_bgem3_sparse` collection. Expose `search_sparse()`.

- Use `pymilvus.MilvusClient` with `uri` + `token`
- Search field: `sparse_vector` (SPARSE_FLOAT_VECTOR)
- Filter output: extract `arxiv_id` from results
- Search with `metric_type="IP"` on sparse vectors
- Handle gRPC reconnection (retry once on connection error)
- Return `[{'arxiv_id': str, 'score': float}]`

**Test**: Live sparse search with the sparse vector from Step 1.
Verify results contain valid arxiv_ids.

### Step 3 — Dense Search in Qdrant (`app/qdrant_svc.py`)

Add `search_dense()` function. Keep all existing functions unchanged.

- Raw vector search returning `[{'arxiv_id': str, 'score': float}]`
- Uses `query_points()` with the dense vector as query

**Test**: Live dense search with the dense vector from Step 1.
Verify arxiv_ids in results are valid strings.

### Step 4 — Groq Rewriter (`app/groq_svc.py`)

LLM-powered query expansion. Must be fully optional.

- Lazy Groq client init (only connects when first query arrives)
- Few-shot prompt from the prototype notebook (already tuned)
- Timeout: 2 seconds max
- Fallback: return original query on any error

**Test**: Rewrite "attention is all you need". Verify output contains "transformer"
or "self-attention". Test fallback by using invalid API key.

### Step 5 — Hybrid Search Orchestrator (`app/hybrid_search_svc.py`)

Wire everything together: rewrite → encode → parallel(dense, sparse) → RRF → rerank.

- `asyncio.gather` with `run_in_executor` for parallel dense + sparse search
- RRF fusion: `score = 1/(K + rank_dense) + 1/(K + rank_sparse)`, K=60
- Recency rerank: `0.80 × norm_rrf + 0.20 × recency_score`
- No citation data yet (deferred)

**Test**: Search "hallucination in language models". Verify results contain
hallucination-related papers.

### Step 6 — Swap Search Router

Replace `arxiv_svc.search(q)` in `app/routers/search.py` with
`hybrid_search_svc.search(q)` + `arxiv_svc.fetch_metadata_batch()`.

One-function swap. Router response format unchanged.

**Test**: All existing integration tests should still pass.
Run `python -m pytest tests/ -v`.

### Step 7 — Model Warm-up + Deployment Config

- Add `embed_svc.get_model()` to `main.py` lifespan
- Update `run.py` to use `APP_PORT` (7860 for HF Spaces)
- Create `Dockerfile` for HF Spaces deployment
- Create `.dockerignore` (exclude `.git`, `__pycache__`, `*.db`, notebooks)

**Test**: Start app with `python run.py`, verify search works end-to-end
at `http://localhost:7860`.

### Step 8 (Optional) — Citation Data

Decide on data source for citation counts, add to reranker.

| Option | Pros | Cons |
|---|---|---|
| Semantic Scholar API | Free, up-to-date | Extra HTTP call, rate limited |
| Skip for now | Simple | Weaker rerank (RRF + recency only) |

**Recommended**: Skip initially. Use `0.80 × rrf + 0.20 × recency`. The RRF signal
alone is strong — the prototype showed citation mostly helps surface seminal papers.

---

## Latency Budget

Assuming BGE-M3 is warm (loaded at startup):

| Stage | Time | Notes |
|---|---|---|
| LLM rewrite (Groq) | ~300ms | Can be skipped for academic queries |
| BGE-M3 encode (CPU) | ~300ms first, ~0ms cached | LRU cache on query text |
| Qdrant dense search | ~200ms | Network + HNSW |
| Zilliz sparse search | ~300ms | Network + sparse IP |
| Both (parallel) | ~300ms | Bottleneck = max(Qdrant, Zilliz) |
| RRF + rerank | <5ms | Pure Python, pre-built dicts |
| Metadata fetch | ~0ms (cached) / ~500ms (cold) | SQLite → arXiv API fallback |
| **Total (warm cache)** | **~600ms** | With LLM, warm encode, warm metadata |
| **Total (fully cached)** | **~300ms** | Encode cached, metadata cached |

Phase 1 search: ~500ms (arXiv API, no local computation).
Phase 3 search: ~600ms warm / ~1.1s cold. **Comparable latency, far better quality.**

---

## RRF — Why It Works Here

RRF's elegance is that it ignores raw scores entirely. Only rank matters:

```
Paper X: rank 1 in dense, rank 5 in sparse
  score = 1/(60+1) + 1/(60+5) = 0.01639 + 0.01538 = 0.03177

Paper Y: rank 3 in dense, rank 3 in sparse
  score = 1/(60+3) + 1/(60+3) = 0.01587 + 0.01587 = 0.03175
```

Papers consistently ranked well across BOTH channels get boosted. This property means
you don't need to normalize Qdrant's cosine scores vs Zilliz's IP scores — they're on
different scales but RRF doesn't care.

**Doc 06 confirms**: RRF is correct here because this is fusing *different retrievers
answering the same query*. Unlike recommendations (fusing *different queries for the same
user*), where quota is correct.

---

## LLM Query Rewriter Details

The rewriter converts casual queries into dense academic keyword strings:

| User Query | Rewritten Query |
|---|---|
| "when the AI makes up fake facts" | "LLM hallucination factual errors sycophancy truthfulness survey" |
| "the llama model by facebook" | "LLaMA open efficient foundation language model Meta AI" |
| "how to make images from text" | "text-to-image generation diffusion models latent space" |
| "whisper speech recognition" | "Whisper OpenAI ASR multilingual" |

**When to skip**: If the query already looks academic (contains arXiv-style terms, author
names, or model acronyms). A heuristic: if the query is >6 words and contains uppercase
acronyms, skip the rewrite.

**Fallback**: Always falls back to the original query on any Groq error. LLM rewriting is
an enhancement, not a dependency.

---

## Test Plan

### Unit Tests (new: `tests/test_embed_svc.py`)
- `test_encode_returns_dense_and_sparse` — verify shapes and types
- `test_encode_cache_hit` — second call returns same result without model invocation
- `test_encode_empty_string` — handles edge case gracefully

### Unit Tests (new: `tests/test_groq_svc.py`)
- `test_rewrite_produces_academic_query` — basic rewrite
- `test_rewrite_fallback_on_error` — returns original on API failure
- `test_rewrite_fallback_on_timeout` — returns original after timeout

### Integration Tests (new: `tests/test_hybrid_search.py`)
- `test_rrf_fusion_basic` — mock dense + sparse results, verify fusion ranking
- `test_search_end_to_end` — live search, verify results are relevant
- `test_search_fallback_no_groq` — search works without Groq API key

### Live Tests (new: `tests/test_zilliz_svc.py`)
- `test_sparse_search_returns_results` — live Zilliz search
- `test_sparse_search_valid_arxiv_ids` — results have valid arxiv_id strings

### Regression
- All 88 existing tests must still pass
- Router integration tests (`test_integration.py`) must work with the new search backend

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **BGE-M3 doesn't fit in memory** | App crashes | HF Spaces free tier = 16GB RAM. Model + PyTorch ≈ 2GB. Well within limits. |
| **Zilliz free tier rate limits** | Search fails | Graceful fallback: return dense-only results if Zilliz is down |
| **Groq API down** | No query rewriting | Already handled: fallback to original query. Enhancement, not dependency. |
| **HF Spaces ephemeral storage** | SQLite data lost on restart | Acceptable pre-launch. Phase 4 migrates to external DB if needed. |
| **Cold start (~15s model load)** | First request slow after sleep | Model baked into Docker image. Warm-up in lifespan. HF Spaces sleeps after ~2 days, not 15 min. |
| **Phase 1 arXiv search tests break** | CI fails | Update tests to mock `hybrid_search_svc.search()` instead of `arxiv_svc.search()` |

---

## File Structure After Phase 3

```
app/
├── __init__.py
├── config.py              # MODIFIED — Zilliz, Groq, BGE-M3, port config
├── db.py                  # UNCHANGED
├── main.py                # MODIFIED — BGE-M3 warm-up in lifespan
├── arxiv_svc.py           # UNCHANGED — still used for metadata fetch
├── qdrant_svc.py          # MODIFIED — add search_dense()
├── user_state.py          # UNCHANGED
├── templates_env.py       # UNCHANGED
├── embed_svc.py           # NEW — BGE-M3 model singleton
├── zilliz_svc.py          # NEW — Zilliz sparse search client
├── groq_svc.py            # NEW — LLM query rewriter
├── hybrid_search_svc.py   # NEW — search orchestrator
├── recommend/             # UNCHANGED
│   ├── __init__.py
│   ├── profiles.py
│   ├── clustering.py
│   ├── reranker.py
│   └── diversity.py
├── routers/               # search.py MODIFIED, rest UNCHANGED
│   ├── search.py          # MODIFIED — swap to hybrid search
│   ├── events.py
│   ├── recommendations.py
│   └── saved.py
└── templates/             # UNCHANGED
    ├── base.html
    ├── index.html
    ├── search.html
    ├── saved.html
    └── partials/

Dockerfile                 # NEW — HF Spaces deployment
.dockerignore              # NEW
run.py                     # MODIFIED — configurable port (7860)
requirements.txt           # MODIFIED — add FlagEmbedding, pymilvus, groq
```

---

## What Phase 3 Does NOT Do

These are explicitly out of scope:

- ❌ Change the recommendation pipeline (that's Phase 4)
- ❌ Replace RRF with quota fusion for recs (Phase 4)
- ❌ Add citation data to the search reranker (Phase 3 Step 8, optional)
- ❌ Load BGE-M3 for recommendations (recs use pre-computed vectors in Qdrant)
- ❌ Change templates or HTMX patterns
- ❌ Add onboarding (Phase 5)
- ❌ Train LightGBM (Phase 6)

---

## Verification Checklist

Before declaring Phase 3 complete:

- [ ] `python -m pytest tests/ -v` — all tests pass (88 existing + new)
- [ ] Search "attention is all you need" — top result is `1706.03762`
- [ ] Search "when AI makes up fake facts" — returns hallucination papers
- [ ] Search with Groq API key unset — still works (falls back to original query)
- [ ] Search with Zilliz down — falls back to dense-only results
- [ ] Save a paper from search results — EWMA profiles update correctly
- [ ] Recommendations still work — 3-tier cascade unaffected
- [ ] App starts on HF Spaces (port 7860, Docker SDK)
- [ ] Cold start completes within 60 seconds
- [ ] Warm search latency < 1 second

---

*Last updated: 2026-04-19*
