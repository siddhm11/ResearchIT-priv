# PHASE 8 — Search and Recommendation: the exact design

**Status:** Partly shipped · **Updated:** 2026-07-30

The definitive specification of both pipelines: every stage, every constant,
where it lives, and whether it is live or pending. Latency figures are measured
against the deployed Space, not estimated.

Legend: **[LIVE]** merged to main · **[PENDING]** designed, not built

---

## 1. Search

`GET /search?q=…` → `app/routers/search.py` → `app/hybrid_search_svc.py:search()`

```
query
  │
  ├─[1]─► Groq rewrite            concurrent, non-blocking        [LIVE]
  ├─[2]─► BGE-M3 encode           in executor, max_length=512     [LIVE]
  │
  ├─[3a]► Qdrant dense    limit=60, rescore=True, oversampling=1  [LIVE]
  └─[3b]► lexical         Zilliz sparse → FTS5                    [PENDING]
            │
          [4] RRF fusion, k=60                                    [LIVE]
            │
          [5] cross-encoder, top 50, full abstracts               [LIVE / PENDING]
            │
          [6] title-match + citation boost, top 50                [LIVE]
            │
          [7] return top 10
```

### Stage detail

| # | Stage | Constants | Measured |
|---|---|---|---|
| 1 | Groq rewrite | `groq_svc.rewrite`, skipped if ≤2 words or `_looks_academic` | 161–329 ms, **overlapped** |
| 2 | BGE-M3 encode | `max_length=512`, LRU 128, `run_in_executor` | 285–642 ms |
| 3a | Qdrant dense | `limit = 10 × SEARCH_FETCH_K_MULTIPLIER(6) = 60` | **~190 ms** |
| 3b | Zilliz sparse | same limit | ~network |
| 4 | RRF | `SEARCH_RRF_K = 60`, `1/(k+rank)` summed over all lists | ~0 ms |
| 5 | Cross-encoder | `SEARCH_RERANK_TOP_N = 10`, sigmoid → [0,1]; `SEARCH_BGE_RERANK=0` disables | 139–182 ms at n=10 |
| 6 | Boosts | exact 2.0 · substring 1.0 · coverage ≥0.8→1.0 / ≥0.5→0.5 · citation cap 0.2 | ~1 ms |
| 7 | Return | `ARXIV_MAX_RESULTS = 10` | |

### Rules that must not change

- **RRF is correct for search.** Many retrievers, one query — rank-based fusion
  needs no score calibration. Do not replace it with quota (that is the
  recommendation-side answer to a different problem).
- **~~Rerank window must exceed the result count.~~ SUPERSEDED 2026-08-02 by
  commit `082e383`.** This rule said 50-of-60 was current and that a window of
  10 was "worse than useless". The measured sweep says the opposite: recall@10
  rose 66.7% → 73.3% → **85.0%** as the window narrowed 50 → 25 → 10, and the
  50-vs-25 comparison moved 14 targets up and 0 down (sign test p = 0.000122).
  `SEARCH_RERANK_TOP_N` now defaults to **10**.

  Read the numbers carefully, because they do not mean the reranker got better.
  `hybrid_search_svc.py` truncates with `fused[:top_n]`, and `ARXIV_MAX_RESULTS`
  is also 10 — so at a window of 10 the cross-encoder **cannot change which
  papers are returned, only their order**, and 85.0% is simply the recall of the
  RRF ordering itself. The real finding is that at 50 the cross-encoder was
  *destroying* recall, pulling 85% down to 66.7% by promoting near-misses over
  correct papers.

  Open question, not yet measured: whether the stage still earns ~58% of search
  latency for the ordering-only benefit it can still provide. `SEARCH_BGE_RERANK=0`
  turns it off; compare MRR@10, which is the only metric it can now move.
- **Rescore stays on.** `rescore=False` is 615 ms vs 608 ms — no faster — and
  drops recall@10 from 100% to 57%. Binary codes alone cannot rank 1024-dim
  vectors.
- **Title boost uses the ORIGINAL query, never the rewrite.** The user's literal
  text is what should match a title.
- **Stage timings no longer sum.** `groq_time_ms` overlaps `encode_time_ms`;
  `search_meta.groq_overlapped` flags this.

### Pending

1. **Lexical from FTS5, not Zilliz** — same RRF input shape, no network, one
   fewer vendor, ~2 GB less storage. A/B against Zilliz before switching:
   BGE-M3 sparse weights are learned, BM25 is not.
2. **Full abstracts to the cross-encoder** — 90.0% of stored abstracts are
   truncated at 500 chars while the retriever saw 1024. The precision stage
   currently has less information than the stage it refines. Biggest remaining
   search-quality item.

---

## 2. Recommendations

`GET /api/recommendations` → `app/routers/recommendations.py`

Cascading tiers, first non-empty wins.

```
Tier 1  ≥5 saves   clustering + quota fusion      ← the actual product
Tier 2  ≥3 saves   EWMA long-term vector
Tier 3  ≥1 save    Qdrant Recommend (BEST_SCORE)
Tier 0  onboarded  trending by category
```

### Tier 1 — the real pipeline

```
saved papers
  └─[1] fetch vectors            qdrant_svc.get_paper_vectors    1247 ms / 20  ← hot spot
  └─[2] Ward clustering          MIN_PAPERS_FOR_CLUSTERING = 5
  └─[3] Hungarian stabilise      against persisted medoids
  └─[4] quota allocation         total_slots=100, min_slots=3, by importance
  └─[5] per-cluster ANN          limit = quota × _OVERSAMPLE(3), parallel
  └─[6] short-term supplement    _ST_SUPPLEMENT = 20
  └─[7] scoring                  heuristic (default) | LightGBM      [LIVE]
  └─[8] category suppression     ≥3 dismissals / 14 days, arXiv codes [LIVE]
  └─[9] MMR diversity            lambda_param = 0.6, top_k = 10
  └─[10] exploration injection   n_explore = 2  → returns limit + 2
```

### Scoring — why the heuristic is the default

`RERANKER_MODE = "heuristic"` (`app/config.py`).

Parsing `reranker_v1.txt`: features 20–30 have **zero splits across all 141
trees** — every EWMA similarity, both cluster features, the suppression and
onboarding flags, all four interaction counts. A tree only reads features it
splits on, so a user's entire profile provably cannot change the output. Not a
statistical claim; structural.

`candidate_num_cited_by` additionally holds 65.2% of importance and is
hardcoded to 0 at serving time.

`heuristic_score()` reads features 20–22:

```
has_ewma:  relevance = 0.40·lt_sim + 0.25·st_sim
otherwise: relevance = 0.65·qdrant_cosine
```

Demonstrated: opposite `ewma_longterm_similarity` produces rankings
`[0,1,2,3,4,5]` vs `[5,4,3,2,1,0]` — a full reversal from the profile alone.

Set `RERANKER_MODE=lightgbm` to compare once real engagement data exists.
`/healthz/reranker` reports `model_loaded` and `scoring_with` separately.

### Rules that must not change

- **Quota, not RRF, for recommendations.** Many queries (one per interest
  cluster), one user. RRF would let the dominant cluster win on rank alone and
  reintroduce interest collapse — the failure the whole product exists to
  prevent.
- **Hungarian matching stays.** Without it a user's "NLP cluster" becomes
  `cluster_7` after the next recluster and instrumentation loses continuity.
- **Suppression on arXiv codes, not `primary_topic`.** The latter has ~15 coarse
  buckets; `AI/ML` alone is 20.2% of the corpus, so three dismissals used to
  suppress a fifth of everything.
- **MMR λ=0.6** — below ~0.5 relevance degrades visibly; above ~0.8 the feed
  collapses toward one interest.

### Pending

1. **Candidate vectors local.** `get_paper_vectors` is 1,247 ms for 20; Tier 1
   needs 100+. MMR and scoring both need real vectors, not ids. At int8, 1.9M
   vectors is 1.95 GB — same sidecar pattern as metadata. Biggest remaining
   recommendation latency item.
2. **Persistence to Turso.** Profiles, clusters and interactions live in SQLite
   at `/tmp` and are destroyed on every rebuild. Gates everything below.
3. **Reranker retrain** on real interactions, restricted to features that are
   non-zero at serving time. Requires 2.

---

## 3. Cold start

| Tier | Trigger | Source |
|---|---|---|
| 0 | onboarded, 0 saves | `fetch_trending_by_categories` |
| 3 | 1 save | Qdrant Recommend |
| 2 | 3 saves | EWMA long-term |
| 1 | 5 saves | full pipeline |

Trending ranks by citations **within a recency window** measured back from the
newest paper in the corpus, widening 24 → 48 → 96 months for thin categories,
with publication date decoded from the arXiv id (`update_date` is the revision
date, so 2017 classics looked new). `TRENDING_RECENCY_MONTHS = 24`.

**Known gap:** the onboarding wizard targets 5 seed saves and Tier 1 needs 5,
but nothing tells the user that 5 is a threshold. Save 4 and you silently get
Tier 3, the weakest path.

---

## 4. Build order

| # | Task | Depends on | Risk |
|---|---|---|---|
| 1 | Persistence → Turso | — | none, additive |
| 2 | FTS5 in sidecar + A/B vs Zilliz | — | none until switched |
| 3 | Full abstracts (ingest + backfill) | Phase 7 | needs arXiv job |
| 4 | Local candidate vectors | index build | image size |
| 5 | Retire Zilliz | 2 | after A/B |
| 6 | Reranker retrain | 1 + months of data | — |

Shipped already: event-loop fix, Groq overlap, `RERANKER_MODE`, rerank window,
quantization search params, arXiv-code suppression, recency trending, metadata
sidecar.
