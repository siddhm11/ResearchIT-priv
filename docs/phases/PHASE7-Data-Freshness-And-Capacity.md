# PHASE 7 — Data Freshness and Capacity

**Status:** Planning · **Created:** 2026-07-30
**Prerequisite for:** Phase 8 (reranker retrain), Phase 9 (exploration / CF)

Every figure below was measured against the live deployment on 2026-07-29/30,
not estimated. Where something is inferred rather than measured it says so.

---

## 1. The one-paragraph version

The corpus is a static Kaggle snapshot whose newest paper was published
**2025-05**. It is now **2026-07**. Live arXiv holds **290,672** papers in our
43 configured categories that we do not have — for cs.AI alone, arXiv has
published nearly half again as many papers in that window as our entire index
contains. Separately, **90.0%** of stored abstracts are truncated at 500
characters, and that truncated text is what the cross-encoder reranks on. No
amount of ranking work fixes either. This phase closes both.

The blocker is disk on the Qdrant free tier, and there is an unresolved
measurement discrepancy that must be settled before anything is deleted
(§4.2).

---

## 2. Measured current state

### 2.1 Corpus

| Property | Value |
|---|---|
| Papers (Turso, sidecar) | 1,597,097 |
| Qdrant points | 1,596,587 (510 fewer; 0.03%) |
| Qdrant indexed into HNSW | 1,595,264 (1,323 pending) |
| `update_date` range | 2007-05-23 → 2025-05-30 |
| Newest *publication* (from arXiv id) | 2025-05 |
| Zilliz sparse | LoadStateLoaded |

Field completeness is perfect: zero rows missing title, authors, categories or
date. 16.7% (265,928) have zero citations.

Store alignment is **not** a problem. A 400-id sample resolved 400/400 in
Qdrant. The 510-row gap needs no action.

### 2.2 Qdrant configuration

Binary quantization **is** enabled (`quantization_config.binary.always_ram =
true`). An earlier reading of `config.params.quantization_config` returned
null and was wrong — the setting lives at `config.quantization_config`.

```
vectors      float16, 1024-dim, on_disk = true
hnsw         m=32, ef_construct=128, on_disk = true
payload      on_disk = true
segments     35, max_segment_size = 100,000
```

### 2.3 Per-point cost (measured, not estimated)

From segment telemetry: segment 0 holds 1,323 points with
`vectors_size_bytes = 2,709,504` and `payloads_size_bytes = 169,344`.

| Component | Bytes/point | × 1,596,587 |
|---|---|---|
| Vector (float16 × 1024) | 2,048 | 3.27 GB |
| Payload | 128 | 204 MB |
| Binary quantization | 128 | 204 MB |
| HNSW graph (m₀=64 × u32, inferred) | ~270 | ~431 MB |
| **Total** | **~2,574** | **≈ 4.1 GB** |

Only the graph row is inferred. `vectors_size_bytes` is a *logical* figure
(exactly `count × 1024 × 2`), and segment `disk_usage_bytes` reports 0, so
Qdrant does not expose true on-disk bytes through the collection API.

---

## 3. The gaps

### 3.1 Staleness — the primary product limit

Union queries against live arXiv, papers submitted 2025-06-01 → 2026-07-30:

| Scope | Unique papers |
|---|---|
| 8 core CS/ML categories | 170,668 |
| **All 43 configured categories** | **290,672** |

Summing categories individually gives 242,427 for the 8-category set — that
counts a paper once per tag. The overlap factor is 1.42×, so always use union
counts for capacity planning.

Per-category, since 2025-06:

| Category | New on arXiv | Entire corpus |
|---|---|---|
| cs.AI | 62,438 | 128,356 |
| cs.LG | 58,747 | 219,516 |
| cs.CV | 43,269 | 156,798 |
| cs.CL | 29,295 | 85,314 |
| quant-ph | 21,026 | 162,381 |

### 3.2 Abstract truncation

`abstract_preview` is capped at 500 chars and **1,437,656 rows (90.0%)** sit at
that cap. Sampling 200 live arXiv abstracts: mean **1,244** chars, median 1,221,
min 601 — every one exceeds the cap, so roughly **60% of each abstract is
discarded**.

Two consequences:

1. The cross-encoder — the stage whose entire purpose is precision — reranks on
   roughly one or two sentences. Since `SEARCH_RERANK_TOP_N` is now 50, it does
   this for 50 candidates per query.
2. The original embeddings were built from `abstract[:1024]`, i.e. **twice** the
   text the reranker sees. The precision stage has strictly less information
   than the retrieval stage it is supposed to refine.

### 3.3 Dependency risk found while validating

`requirements.txt` allowed `FlagEmbedding>=1.2.9` with
`transformers>=4.44,<5.0`. A fresh resolve picks FlagEmbedding 1.3.x and a
recent transformers, and BGE-M3 encoding dies inside `tokenizer.pad()` with
`AttributeError: 'list' object has no attribute 'keys'`.

This is silent in production: `main.py` catches the load error, logs "BGE-M3 not
loaded", and search degrades to the arXiv keyword API with nothing surfaced to
the user. Pinned to `1.2.10` + `4.44.2` (verified loading) in commit `4b88183`.

---

## 4. Capacity

### 4.1 What the ingest costs

At ~2,574 bytes/point:

| Scope | Papers | Qdrant disk | BQ RAM |
|---|---|---|---|
| All 43 categories | 290,672 | **+0.75 GB** | +37 MB |
| 8 core CS/ML | 170,668 | +0.44 GB | +22 MB |

Other stores: Turso ~465 MB and 290k row writes (free tier is 9 GB / 25M
writes/month — comfortable). Zilliz ~400–450 MB sparse, and its serverless
free-tier cap is **not verifiable through the API** — this is the least
characterised constraint.

### 4.2 Decision gate: the disk reading must be settled first

Two irreconcilable figures:

* This document's arithmetic: **≈ 4.1 GB** used, against a 4 GB free-tier disk.
* The Qdrant Cloud console, as reported: **400–500 MB**.

400–500 MB cannot be the on-disk total. Raw float16 vectors are not compressed,
so 1,596,587 × 2,048 bytes = 3.27 GB is a floor for the vector data alone. The
reported figure does, however, match `memory_resident_bytes = 529,870,848`
(530 MB) almost exactly, which suggests the panel is reporting **RAM, not
disk** — consistent with a 1 GB RAM tier at ~53% utilisation.

**Nothing may be deleted until this is resolved.** The two branches lead to
opposite actions:

| If the true disk usage is… | Then |
|---|---|
| ~500 MB (≈3.5 GB free) | Ingest all 290,672. **Delete nothing.** |
| ~4.1 GB (≈0 GB free) | Ingest cannot proceed as-is; see §5 |

How to resolve: in the Qdrant Cloud console, find the panel labelled *disk* or
*storage* rather than memory. Failing that, upsert 5,000 test points and observe
whether reported usage moves by ~13 MB (RAM) or ~13 MB of disk — then delete
them.

### 4.3 Eviction analysis — dropping pre-2010

Requested on 2026-07-30. Measured consequences:

| | |
|---|---|
| Papers removed | **180,197** (11.3% of corpus) |
| Disk freed | **0.46 GB** |
| Ingest still needs | 0.75 GB |
| **Net after both** | **+0.28 GB — still grows** |
| Of those removed, with >0 citations | 155,312 (**86%**) |
| Of those removed, with ≥100 citations | **17,776** |

Most affected categories:

```
quant-ph            35,763      hep-th               8,477
astro-ph            30,329      math-ph              8,178
cond-mat.stat-mech  10,639      math.MP              8,178
math.CO              9,525      math.PR              8,754
```

**Recommendation: do not do this.** Three reasons:

1. **It does not solve the problem.** 0.46 GB freed against 0.75 GB needed still
   leaves +0.28 GB of growth. If disk is genuinely full, this buys a delay, not
   a fix.
2. **The cost is high and irreversible.** 17,776 removed papers have 100+
   citations. Restoring any of them means re-fetching and re-encoding on a GPU
   we do not currently have working (§6.1).
3. **It lands on the wrong fields.** The damage concentrates in quant-ph,
   astro-ph, hep-th and mathematics — disciplines whose literature ages slowly.
   A 2008 hep-th paper is often still current in a way a 2008 cs.CV paper is
   not. Our onboarding taxonomy offers all four as first-class choices, so those
   users would lose the foundations of their fields while ML users lose nothing.

Cheaper alternatives, in preference order:

| Option | Frees / saves | Cost |
|---|---|---|
| Resolve §4.2 first; likely no action needed | up to 3.5 GB of headroom | none |
| Narrow ingest to 8 core CS/ML categories | needs 0.44 GB not 0.75 GB | no physics/maths freshness |
| Drop pre-2010 papers with **zero** citations only | 0.06 GB (24,885 papers) | negligible |
| Lower HNSW `m` 32 → 16 | ~215 MB | full re-index; small recall cost |

---

## 5. Plan

### Phase 7.0 — Resolve capacity (blocking, minutes)
Settle §4.2. Everything downstream branches on it.

### Phase 7.1 — Get a working GPU (blocking for ingest)
Kaggle assigned a **P100 (sm_60)**; its own `torch 2.10.0+cu128` dropped Pascal
support, so encoding aborts. `"accelerator": "nvidiaTeslaT4"` in
`kernel-metadata.json` is silently ignored — GPU *type* is not settable via the
API. Requires switching the notebook to **GPU T4 x2** in the Kaggle UI.

Why it matters: the arXiv rate limit (~3.5s per 200 papers) floors the job at
~85 minutes. On T4 encoding hides inside that window. On CPU it is ~13.5 hours,
over Kaggle's 12-hour session cap.

### Phase 7.2 — Ingest (scope set by 7.0)
`scripts/ingest_arxiv.py`, checkpointed per batch and idempotent. Run in slices,
checking collection status between them. Validated already: arXiv fetch, parse,
pagination, `totalResults`, and the Qdrant id-collision guard (highest live id
1,596,586 — new points continue above it).

### Phase 7.3 — Abstract backfill
New rows arrive with full abstracts automatically. The existing 1,437,656
truncated rows need re-fetching: ~7,200 requests at the arXiv rate limit ≈ **7
hours**, no GPU required. Turso and the sidecar only; vectors untouched.

### Phase 7.4 — Sidecar rebuild
`scripts/build_metadata_sidecar.py`, then re-upload to
`siddhm11/researchit-metadata`. Not urgent: `local_meta` falls back to Turso on
a miss, so newly ingested papers work immediately without it.

### Phase 7.5 — Re-embed on full abstracts (optional, later)
Only worth doing once 7.3 lands. Would improve retrieval but requires
re-encoding 1.6M papers and doubles peak Qdrant disk during the swap. Defer.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Disk exhaustion mid-ingest | Run in slices; script is checkpointed and idempotent, so a failed write loses nothing |
| Zilliz free-tier cap unknown | Watch after the first slice; it is the least characterised store |
| Segment merges spike disk transiently | `max_segment_size=100,000` → ~3 new segments; merges write before dropping, so peak exceeds steady state |
| New vectors inconsistent with old | Embedding text and `max_length` match the original ingest exactly; verified in `ingest_backends.py` |
| Point-id collision overwriting papers | Upserter binary-searches for the highest live id and continues above it |
| Rollback | New points occupy a contiguous id range above 1,596,586 and are deletable by `arxiv_id`; no existing point is modified |

---

## 7. Open questions

1. **True Qdrant disk usage** (§4.2) — blocks everything.
2. **Zilliz free-tier storage cap** — not queryable via API.
3. **Ingest cadence after backfill** — GitHub Actions (a keepalive cron already
   exists) or HF Jobs. ~500 papers/day steady state.
4. **Citation counts for new papers** — ingested as 0. Both the reranker and the
   citation boost lean on this, so new papers are systematically disadvantaged
   until a Semantic Scholar enrichment pass runs. `S2_API_KEY` already exists
   for offline scripts.
