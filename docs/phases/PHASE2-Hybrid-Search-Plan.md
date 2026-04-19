# Phase 2 — Hybrid Semantic Search

## What the notebook script (`bgem3_search_client.py`) does

This is the research prototype that Phase 2 must productionise into the FastAPI app.
Understanding it completely is the prerequisite to knowing what to build.

---

## The Script's Full Pipeline

```
User query (casual language)
      │
      ▼
[1] LLM Query Rewriter (Groq / llama-3.3-70b)
      │  "whisper speech recognition" → "Whisper OpenAI ASR multilingual"
      ▼
[2] BGE-M3 Encoder (CPU, single forward pass)
      │  produces TWO outputs simultaneously:
      │  ├── dense_vec  : float32[1024]   — semantic meaning
      │  └── sparse_dict: {token_id: weight}  — lexical weights (BGE-M3's own sparse)
      ▼
[3a] Qdrant dense search          [3b] Zilliz sparse search
      │  HNSW ANN on 1024-dim           │  IP on sparse vectors
      │  arxiv_bgem3_dense              │  arxiv_bgem3_sparse
      │  returns: [(point_id,           │  returns: [(point_id,
      │            arxiv_id, score)]    │            arxiv_id, score)]
      │                                 │
      └──────────── parallel ───────────┘
                        │
                        ▼
[4] RRF Fusion  (Reciprocal Rank Fusion, K=60)
      │  score[point] = 1/(60+rank_dense) + 1/(60+rank_sparse)
      │  pure rank-based, no score normalisation needed
      ▼
[5] Reranker  (citation + recency + semantic)
      │  final = 0.70 × norm_rrf  +  0.25 × log_citations  +  0.05 × recency
      ▼
[6] Return top-K arxiv_ids → fetch metadata → display
```

---

## The Single Most Important Insight

BGE-M3 is a **dual-encoder**: one forward pass produces both a dense vector and a
sparse lexical-weights dict. The sparse side is NOT BM25 — it is BGE-M3's own learned
sparse representation (`lexical_weights` from FlagEmbedding).

This has a critical consequence: **the Zilliz collection was indexed with BGE-M3 sparse
outputs, so query-time sparse encoding must also use BGE-M3**. You cannot use a BM25
tokeniser for the Zilliz queries. The model must be loaded and run for every search.

```python
# From the script — one call, two outputs:
out = model.encode(
    [text],
    return_dense=True,
    return_sparse=True,       # ← BGE-M3 lexical weights, not BM25
    return_colbert_vecs=False,
    max_length=512,
)
dense  = out["dense_vecs"][0]          # shape (1024,) float32
sparse = out["lexical_weights"][0]     # dict {token_id: float}
```

---

## Gap Analysis: What Phase 1 Has vs What Phase 2 Needs

### Phase 1 search (current)
```
User query (exact string)
      │
      ▼
arXiv keyword API  (https://export.arxiv.org/api/query?search_query=all:query)
      │  standard text match, not semantic
      ▼
SQLite metadata cache → display cards
```

**Problems with Phase 1 search:**
- Keyword-only: "when the AI makes up fake facts" returns nothing useful
- No awareness of what the user means, only what they typed
- Misses papers that use different terminology for the same concept
- No ranking signal beyond arXiv's own relevance score

### Phase 2 search (target)
```
User query
      ▼
LLM rewrite → encode → dense search ‖ sparse search → RRF → rerank
      ▼
arXiv API (metadata only, for titles/abstracts not in Qdrant/Zilliz payloads)
```

---

## What Each Component Replaces or Adds

| Component | Phase 1 | Phase 2 | Notes |
|---|---|---|---|
| Search backend | arXiv API keyword | BGE-M3 + Qdrant + Zilliz | Core change |
| Query preprocessing | None | Groq LLM rewrite | Optional but high-quality gain |
| Metadata source | arXiv API + SQLite | arXiv API + SQLite | **Unchanged** — Qdrant/Zilliz payloads only store `arxiv_id` |
| Recommendations | Qdrant Recommend API | Qdrant Recommend API | **Unchanged** — already works well |
| User state | In-memory deque + SQLite | In-memory deque + SQLite | **Unchanged** |
| Citation signal | None | Semantic Scholar API or Zilliz payload | New |
| Model at runtime | None | BGE-M3 (~570MB, ~300ms CPU) | Big change — cold start is now slow |

---

## Files That Change in Phase 2

### New files to create

**`app/embed_svc.py`** — BGE-M3 model singleton
```
Responsibilities:
  - Load BAAI/bge-m3 once at startup (or lazily on first query)
  - encode_query(text) → (dense: np.ndarray[1024], sparse: dict)
  - Cache recent encode results to avoid re-encoding the same query
  - CPU float32 (no GPU dependency)
```

**`app/zilliz_svc.py`** — Zilliz sparse search client
```
Responsibilities:
  - Connect to Zilliz serverless (pymilvus MilvusClient)
  - search_sparse(sparse_dict, fetch_k) → list[(point_id, arxiv_id, score)]
  - Handle gRPC reconnects (closed-channel error, seen in the script)
```

**`app/groq_svc.py`** — LLM query rewriter
```
Responsibilities:
  - Groq API client (lazy init)
  - rewrite(user_query) → academic_query  (8-15 words, preserves model names)
  - Falls back to original query on failure or timeout
  - Same few-shot prompt as in the script (already tuned)
```

**`app/hybrid_search_svc.py`** — orchestrates the full pipeline
```
Responsibilities:
  - Calls groq_svc.rewrite()
  - Calls embed_svc.encode_query()
  - Calls qdrant_svc.search_dense() and zilliz_svc.search_sparse() in parallel
    (asyncio.gather with run_in_executor for both sync clients)
  - RRF merge
  - Citation+recency rerank
  - Returns list of arxiv_ids
```

### Files that change

**`app/config.py`** — add:
```python
ZILLIZ_URI        = os.getenv("ZILLIZ_URI", "https://in03-0c01933b42a8df1.serverless...")
ZILLIZ_TOKEN      = os.getenv("ZILLIZ_TOKEN", "...")
ZILLIZ_COLLECTION = "arxiv_bgem3_sparse"

GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "gsk_...")

BGE_M3_DEVICE     = os.getenv("BGE_M3_DEVICE", "cpu")
ENCODE_CACHE_SIZE = 128            # LRU cache for encoded queries
SEARCH_FETCH_K_MULTIPLIER = 6      # candidates = top_k × 6 before rerank
SEARCH_RRF_K      = 60             # RRF denominator
SEARCH_CITE_WEIGHT   = 0.25
SEARCH_RECENCY_WEIGHT = 0.05
SEARCH_SEMANTIC_WEIGHT = 0.70      # must sum to 1.0 with above
```

**`app/qdrant_svc.py`** — add `search_dense()` (vector search, separate from existing `recommend()`):
```python
async def search_dense(dense_vec: np.ndarray, fetch_k: int) -> list[tuple]:
    """
    ANN dense search. Returns [(point_id, arxiv_id, score)].
    This is different from recommend() — it takes a raw query vector,
    not a list of positive paper IDs.
    """
```

**`app/routers/search.py`** — replace `arxiv_svc.search()` call with `hybrid_search_svc.search()`.
The router signature and template rendering stay identical — the change is entirely inside the router.

**`app/main.py`** — add model warm-up to lifespan:
```python
@asynccontextmanager
async def lifespan(app):
    await db.init_db()
    embed_svc.get_model()   # warm up BGE-M3 at startup, not on first request
    yield
```

---

## The Metadata Problem (and Why the CSV Doesn't Apply to the Web App)

The script uses two CSV files loaded from Kaggle paths:
```python
CSV_PATH          = "/kaggle/input/.../arxiv_comprehensive_papers.csv"
CITATION_CSV_PATH = "/kaggle/input/.../arxiv_citations_summary.csv"
```

These are Kaggle notebook paths that don't exist in the web app. The script's `load_csv()`
and `lookup()` functions are the notebook's equivalent of our `arxiv_svc.py` + `db.py`.

**The web app already handles this better**: arXiv API → SQLite cache. The `fetch_metadata_batch()`
function in `arxiv_svc.py` is already doing what the CSV was doing, but dynamically and with
no dependency on a pre-downloaded file.

**What we do NOT have: citation counts.** The script uses `arxiv_citations_summary.csv` to
get citation counts per paper, which feeds the 0.25-weighted citation component of the reranker.

**Options for citation data in Phase 2:**

| Option | Pros | Cons |
|---|---|---|
| Semantic Scholar API | Free, up-to-date | Extra HTTP call per search, rate limited |
| Store in Zilliz payload | Zero extra calls | Requires re-indexing Zilliz collection |
| Store in SQLite when fetched | Persistent cache, one-time cost | Stale, extra complexity |
| Skip citation reranking initially | Simple | Weaker rerank quality |

**Recommended**: Start with **skip citation reranking**. Use only 2 components:
`0.75 × norm_rrf + 0.25 × recency`. Add citation later when we decide on the data source.
The RRF signal alone is strong — the script shows citation mostly helps surface seminal papers.

---

## The Reranker — What It Actually Does

```python
final = (
    0.70 × (rrf_score / max_rrf_score)         # semantic rank
  + 0.25 × (log1p(citations) / max_log_cite)   # popularity (log-normalised)
  + 0.05 × exp(-0.06 × paper_age_years)         # freshness (soft decay)
)
```

Why log-normalise citations: `log1p(214251) ≈ 12.3` vs `log1p(500) ≈ 6.2`. The ratio is
only 2×, so a mega-cited but semantically irrelevant paper can't overpower a relevant
low-cited one. Without log, a 200k-citation paper would always win regardless of relevance.

Why `0.05` recency and not higher: The script was tuned to stop penalising classics.
AlphaFold (2021), RLHF (2017), ResNet (2015) — you want these to appear even for
non-recent queries. With `lambda=0.06`, a 10-year-old paper still scores 0.55 on recency,
so it loses only a little to a 2024 paper.

**For Phase 2 initial implementation** (no citation data):
```python
final = 0.80 × norm_rrf + 0.20 × recency
```

---

## Parallel Search — Critical for Latency

The script runs Qdrant and Zilliz in parallel using `ThreadPoolExecutor`:
```python
with ThreadPoolExecutor(max_workers=2) as ex:
    fq = ex.submit(timed_qdrant)
    fz = ex.submit(timed_zilliz)
    dense_hits, qdrant_ms  = fq.result()
    sparse_hits, zilliz_ms = fz.result()
```

In the FastAPI app we use `asyncio.gather` with `run_in_executor` for the same effect:
```python
loop = asyncio.get_event_loop()
dense_hits, sparse_hits = await asyncio.gather(
    loop.run_in_executor(None, _search_qdrant_dense_sync, dense_vec, fetch_k),
    loop.run_in_executor(None, _search_zilliz_sparse_sync, sparse_dict, fetch_k),
)
```

This is important: total search latency is `max(qdrant_time, zilliz_time)`, not their sum.
The script shows Qdrant ~200ms, Zilliz ~300ms — so parallel wall time is ~300ms, not ~500ms.

---

## Model Loading Strategy

BGE-M3 is ~570MB on disk, takes ~15s to load on CPU. Two strategies:

**Option A — Eager load at startup** (recommended for production)
```python
# in lifespan()
embed_svc.get_model()  # blocks startup until model is loaded
```
Pros: First query is fast. Cons: App takes ~15s to start.

**Option B — Lazy load on first query**
```python
# in encode_query()
model = get_model()  # loads on demand
```
Pros: Fast startup. Cons: First search request after restart takes ~15s — user sees spinner.

For a research tool with infrequent restarts, **Option A is better**.

---

## RRF — Why It Works Without Score Normalisation

RRF's elegance is that it ignores raw scores entirely. Only rank matters:

```
paper X is rank 1 in dense, rank 5 in sparse
  dense score: 1/(60+1) = 0.01639
  sparse score: 1/(60+5) = 0.01538
  total: 0.03177

paper Y is rank 3 in dense, rank 3 in sparse
  both: 1/(60+3) = 0.01587
  total: 0.03175
```

Paper X barely beats Y — consistent top rankings across both channels is better than
excelling in only one. This property means you don't need to worry about Qdrant's cosine
scores being on a different scale than Zilliz's IP scores.

---

## LLM Query Rewriter — What It Does and When to Skip It

The rewriter converts casual queries into dense academic keyword strings:
- `"when the ai makes up fake facts"` → `"LLM hallucination factual errors sycophancy truthfulness survey"`
- `"the llama model by facebook"` → `"LLaMA open efficient foundation language model Meta AI"`

The 15 few-shot examples in the script are carefully chosen — they demonstrate two things:
1. Vague language → academic terminology (helps dense semantic search)
2. Named entities preserved/added (helps sparse keyword search)

**Cost**: ~200-500ms Groq API call per query. The Groq call runs first, before encoding.

**When to skip**: If the user's query already looks academic (contains arXiv-style terms,
author names, or model acronyms), the rewrite adds little. A heuristic: if the query is
>6 words and contains uppercase acronyms, skip the rewrite.

**Fallback**: The script always falls back to the original query on any Groq error. This
is the right pattern — LLM rewriting is an enhancement, not a dependency.

---

## What Phase 2 Leaves Unchanged from Phase 1

These are intentionally NOT changed:

1. **Recommendations** — `qdrant_svc.recommend()` via Qdrant Recommend API. Already works.
   No query encoding needed. No model dependency.

2. **User state** — `user_state.py` deques + SQLite interactions table. Already correct.

3. **Metadata fetching** — `arxiv_svc.fetch_metadata_batch()`. Phase 1's arXiv API + SQLite
   cache is strictly better than the Kaggle CSV approach for a web app.

4. **arXiv ID normalisation** — `arxiv_svc._normalise_id()`. Already handles all formats.

5. **Event logging** — `db.log_interaction()`. Events are source-agnostic. A save from
   hybrid search goes into the same table as a save from arXiv keyword search.

6. **All templates** — `paper_card.html`, `action_buttons.html`, `search_results.html`.
   The templates expect `paper` dicts with `arxiv_id, title, abstract, authors, category,
   published` — exactly what `arxiv_svc.fetch_metadata_batch()` returns.

7. **HTMX patterns** — the save/dismiss flow, lazy-load recommendations, search-as-you-type.

---

## Phase 2 Implementation Order

These steps are ordered to minimise integration risk. Each step leaves the app in a
working state.

### Step 1 — Add BGE-M3 model service (`app/embed_svc.py`)

Load BGE-M3, expose `encode_query(text) → (dense, sparse)` with LRU cache.
No app integration yet — just the service + unit tests.

**Test**: encode "attention is all you need", verify dense shape is (1024,), sparse has >5 keys.

### Step 2 — Add Zilliz client (`app/zilliz_svc.py`)

Connect to `arxiv_bgem3_sparse`. Expose `search_sparse(sparse_dict, k) → list[tuple]`.
Include reconnect logic from the script.

**Test**: live search with the sparse vector from Step 1, verify results come back.

### Step 3 — Add dense search to Qdrant service

Add `search_dense(dense_vec, k) → list[tuple]` to `qdrant_svc.py`.
Keep `recommend()` and `lookup_qdrant_ids()` unchanged.

**Test**: live dense search, verify arxiv_ids in results are valid.

### Step 4 — Add Groq rewriter (`app/groq_svc.py`)

Expose `rewrite(query) → str`. Must fall back gracefully on error.

**Test**: rewrite "attention is all you need", verify output contains "transformer".

### Step 5 — Hybrid search orchestrator (`app/hybrid_search_svc.py`)

Wire: rewrite → encode → parallel(dense, sparse) → RRF → rerank → return arxiv_ids.
No citation data yet — use `0.80 × rrf + 0.20 × recency`.

**Test**: search "hallucination in language models", verify `2309.01219` is in top 5.

### Step 6 — Swap search router

Replace `arxiv_svc.search(q)` in `app/routers/search.py` with
`hybrid_search_svc.search(q)` + `arxiv_svc.fetch_metadata_batch(arxiv_ids)`.

This is a one-function swap. The router's response format stays identical.

**Test**: all 12 integration tests should still pass. Run `python -m pytest`.

### Step 7 — Add model warm-up to lifespan

Add `embed_svc.get_model()` to `app/main.py` lifespan. Update startup log message.

### Step 8 (optional) — Citation data

Decide on data source, add `citationCount` to SQLite `paper_metadata` table,
wire into reranker.

---

## Performance Budget for Phase 2

Assuming BGE-M3 is already warm (loaded at startup):

| Stage | Time | Notes |
|---|---|---|
| LLM rewrite (Groq) | ~300ms | Can be skipped for short/academic queries |
| BGE-M3 encode (CPU) | ~300ms first, ~0ms cached | LRU cache on query text |
| Qdrant dense search | ~200ms | Network + HNSW |
| Zilliz sparse search | ~300ms | Network + sparse IP |
| Both (parallel) | ~300ms | Bottleneck is whichever is slower |
| RRF + rerank | <5ms | Pure Python, pre-built dicts |
| arXiv metadata | ~0ms (cached) / ~500ms (cold) | SQLite → arXiv API |
| **Total (warm cache)** | **~600ms** | With LLM, warm encode, cold Zilliz |
| **Total (fully cached)** | **~300ms** | Encode cached, metadata cached |

Phase 1 search: ~500ms (arXiv API, no local computation).
Phase 2 search: ~600ms warm / ~1.1s cold (first query after restart with LLM rewrite).

The latency is comparable with much better result quality.

---

## New Dependencies

Add to `requirements.txt`:

```
FlagEmbedding>=1.2.9    # BGE-M3
torch>=2.0              # required by FlagEmbedding, CPU-only is fine
pymilvus>=2.4           # Zilliz/Milvus client
groq>=0.9               # Groq LLM API
numpy>=1.24             # already implicit via FlagEmbedding
```

Note: `torch` for CPU-only install:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

BGE-M3 on CPU is ~300ms per encode. Acceptable for a search tool. GPU would bring it to ~30ms.

---

## Summary: The Three Things Phase 2 Is

1. **Replace the search input**: arXiv keyword API → BGE-M3 encode → Qdrant dense + Zilliz sparse → RRF

2. **Add a search quality layer**: LLM query rewriting (Groq) before encoding, citation+recency reranking after RRF

3. **Keep everything else**: recommendations, user state, metadata caching, HTMX frontend, event logging — all unchanged

The script has already proven the core pipeline works (20-query evaluation suite with Recall, Precision, MRR metrics). Phase 2 is productionising that pipeline into the existing FastAPI structure.
