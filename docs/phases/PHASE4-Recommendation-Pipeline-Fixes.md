# Phase 4 — Recommendation Pipeline Fixes

> **Purpose**: Fix the 3 remaining architectural faults identified by Doc 06 in the
> recommendation pipeline: replace RRF with importance-weighted quota fusion, add
> Hungarian matching for cluster stability, and wire category-level negative suppression.
>
> **Status**: 📋 Not started  
> **Estimated effort**: ~1 week  
> **Predecessor**: Phase 3.5 (complete) — Turso metadata DB  
> **Deployment target**: Same — Hugging Face Spaces (no infra changes)

---

## Why This Matters

The recommendation engine works today — all 3 tiers cascade correctly, EWMA profiles
update, Ward clustering detects interests, and MMR enforces diversity. But Doc 06
identified three concrete faults that degrade quality for multi-interest users:

| # | Fault | Impact | Who gets hurt |
|---|---|---|---|
| **4.1** | RRF fuses interest clusters by consensus, not proportionally | Dominant cluster drowns minority interests | User who likes both NLP (70%) and RL (30%) never sees RL papers |
| **4.3** | Cluster indices shuffle on every recluster | Future analytics and UI labels break | Any user who saves a new paper |
| **4.4** | No category-level negative suppression | Dismissed topics keep reappearing | User who dismisses 5 physics papers still gets physics recs |

**What's already fixed (not Phase 4)**:
- ✅ α_long = 0.03 (was 0.10, fixed Phase 2a — PinnerSage rejected 0.10)
- ✅ L2-normalize before Ward (fixed Phase 2b — Doc 06 fault #4)
- ✅ Negative EWMA penalty in reranker (fixed Phase 2c — Feature 5, weight 0.15)
- ✅ Metadata store pre-populated (Phase 3.5 — Turso, 1.23GB)

---

## Current Architecture vs Target Architecture

### Current Retrieval (Phase 2b — being fixed)

```
Cluster medoids + short-term vector
      │
      ▼
Single Qdrant prefetch+RRF call
  ├── Prefetch: medoid_1 (limit=40)
  ├── Prefetch: medoid_2 (limit=30)
  ├── Prefetch: medoid_3 (limit=25)
  └── Prefetch: short_term (limit=25)
      │
      ▼
FusionQuery(fusion=Fusion.RRF)
      │  ← papers near ALL cluster centroids get boosted
      │  ← minority interests get drowned
      ▼
~100 candidates → rerank → MMR → serve
```

**Problem**: RRF was designed for fusing *different retrievers on the same query*
(BM25 + vector). Here we're fusing *different queries for the same user*. Consensus
means "near the centroid of everything" — the exact failure multi-interest models
exist to prevent.

### Target Retrieval (Phase 4)

```
compute_clusters() → K clusters with importance scores
      │
      ▼
allocate_quotas([imp_1, imp_2, ...], total=100, min=3)
  → [55, 30, 15] (proportional, each ≥ 3)
      │
      ▼
asyncio.gather(                   ← concurrent, ~15ms wall-clock
  search_by_vector(medoid_1, limit=55×3),   # 3× over-fetch for rerank headroom
  search_by_vector(medoid_2, limit=30×3),
  search_by_vector(medoid_3, limit=15×3),
  search_by_vector(short_term, limit=25),   # session boost
)
      │
      ▼
Deduplicate across clusters
  (assign each paper to its highest-ranked cluster)
      │
      ▼
Category suppression: drop papers from suppressed categories
      │
      ▼
Rerank → MMR → exploration → serve
```

**Evidence this is correct**:
- PinnerSage (KDD 2020): samples 3 medoids proportional to importance — no RRF
- Taobao ULIM (RecSys 2025): per-category parallel retrieval with quota — +5.54% clicks
- Pinterest Bucketized-ANN (SIGIR 2023): ensures minority items aren't dropped
- Twitter kNN-Embed: candidates per cluster proportional to mixture weight
- Bruch et al. (SIGIR 2022): RRF optimises Recall not nDCG — quota gives better nDCG

---

## 4.1 — Replace RRF with Importance-Weighted Quota Fusion

### New File: `app/recommend/fusion.py`

Pure-math module with zero I/O dependencies. Contains one function:

```python
def allocate_quotas(
    importances: list[float],
    total_slots: int = 100,
    min_slots: int = 3,
) -> list[int]:
    """
    Importance-weighted quota allocation with a minimum floor.

    Each cluster gets feed slots proportional to its importance,
    with a guaranteed minimum of `min_slots` to protect minority interests.

    Algorithm:
      1. Normalise: w_k = importance_k / sum(importances)
      2. Raw allocation: raw_k = total_slots × w_k
      3. Apply floor: slot_k = max(floor(raw_k), min_slots)
      4. Distribute remainder by largest fractional part
      5. Guarantee: sum(slots) == total_slots

    This is the Doc 06 formula verbatim:
      slot_k = max(⌊F × w_k⌋, F_min=3)

    Reference: PinnerSage (KDD 2020), Taobao ULIM (RecSys 2025),
    Pinterest Bucketized-ANN (SIGIR 2023).
    """
```

**Worked example** (from Doc 06 §"Worked example"):
- 3 clusters with importances [0.55, 0.30, 0.15], total_slots=30
- Raw allocation: [16.5, 9.0, 4.5]
- Floor applied: [16, 9, 4] (all ≥ 3, so floor has no effect)
- Remainder: 30 - 29 = 1 slot → goes to cluster 0 (largest fractional part: 0.5)
- Final: [17, 9, 4] — minority cluster gets 4 slots, not 0

**Edge case — tiny cluster**:
- 4 clusters with importances [0.60, 0.25, 0.10, 0.05], total_slots=30
- Raw allocation: [18.0, 7.5, 3.0, 1.5]
- Without floor: [18, 7, 3, 1] — smallest cluster gets 1 paper
- With floor (min=3): [18, 7, 3, 3] — smallest cluster gets 3 papers

### Modified File: `app/routers/recommendations.py`

The `_multi_interest_recommend()` function changes its retrieval step:

**What gets removed**:
- The `_CLUSTER_LIMITS = [40, 30, 25, 20, 15, 15, 15]` hardcoded list
- The call to `qdrant_svc.multi_interest_search()` (the prefetch+RRF path)
- Building the `interest_vectors` list of `(medoid_embedding, limit)` tuples

**What replaces it**:
```python
import asyncio
from app.recommend.fusion import allocate_quotas

# Step 2: Quota-based parallel retrieval (replaces RRF)
quotas = allocate_quotas(
    importances=[c.importance for c in clusters],
    total_slots=100,   # wide retrieval net
    min_slots=3,       # every cluster gets at least 3 slots
)

# Launch concurrent ANN searches — one per cluster + session
search_coros = []
for cluster, quota in zip(clusters, quotas):
    search_coros.append(
        qdrant_svc.search_by_vector(
            query_vector=cluster.medoid_embedding.tolist(),
            limit=quota * 3,  # 3× over-fetch for rerank headroom
            exclude_ids=seen,
        )
    )
# Add short-term session vector if available
st_vec = await profiles.load_profile(user_id, "short_term")
if st_vec is not None:
    search_coros.append(
        qdrant_svc.search_by_vector(
            query_vector=st_vec.tolist(),
            limit=25,
            exclude_ids=seen,
        )
    )

# Execute all searches concurrently (~15ms wall-clock)
per_cluster_results = await asyncio.gather(*search_coros)

# Deduplicate: first occurrence wins (highest-ranked cluster)
seen_in_results = set()
candidate_ids = []
for result_list in per_cluster_results:
    for arxiv_id in result_list:
        if arxiv_id not in seen_in_results:
            seen_in_results.add(arxiv_id)
            candidate_ids.append(arxiv_id)
```

**Key design decisions**:

1. **`asyncio.gather()` for concurrency** — Each `search_by_vector()` call takes ~5-15ms.
   With `asyncio.gather()`, 3-7 concurrent queries run in ~15-25ms wall-clock — same as
   the old single prefetch call.

2. **3× over-fetch** — We fetch `quota × 3` candidates per cluster, then let the reranker
   pick the best `quota` from each. This gives the heuristic scorer enough headroom to
   find quality papers even if some candidates are poor matches.

3. **First-occurrence deduplication** — Papers appearing in multiple cluster results are
   assigned to whichever cluster ranked them highest (first encounter). This is simple,
   deterministic, and matches the PinnerSage pattern.

4. **`multi_interest_search()` is NOT deleted** — The function stays in `qdrant_svc.py`
   for potential future use. We simply stop calling it from the recommendations router.

### Latency Impact

| Stage | Before (RRF) | After (Quota) |
|---|---|---|
| Qdrant retrieval | ~15-25ms (1 prefetch call) | ~15-25ms (3-7 concurrent calls) |
| Dedup + quota | N/A | <1ms |
| Rerank + MMR | ~12ms | ~12ms (unchanged) |
| **Total pipeline** | ~30ms | ~30ms |

No latency regression. The concurrent gather matches the prefetch parallelism.

---

## 4.3 — Hungarian Matching for Cluster Stability

### Why This Matters

When a user saves a new paper, `compute_clusters()` runs Ward clustering from scratch.
The cluster that was "NLP papers" yesterday might get `cluster_idx=2` today and
`cluster_idx=0` tomorrow. This breaks:

- Future analytics ("which cluster does the user engage with most?")
- Future UI labels ("Your Interest: Natural Language Processing")
- A/B test logs that reference cluster indices
- Doc 06 §"Clustering specifics" calls this "the real operational risk"

### Modified File: `app/recommend/clustering.py`

Add a new function called between `compute_clusters()` and `save_clusters_to_db()`:

```python
from scipy.optimize import linear_sum_assignment

def stabilize_cluster_ids(
    new_clusters: list[InterestCluster],
    old_clusters: list[dict] | None,
    paper_vectors: dict[str, list[float]] | None = None,
) -> list[InterestCluster]:
    """
    Remap new cluster indices to match previous clusters via Hungarian matching.

    1. Compute cost matrix: cost[i][j] = 1 - cosine_sim(new_medoid_i, old_medoid_j)
    2. Solve assignment with scipy.optimize.linear_sum_assignment
    3. Remap new cluster_idx to matched old cluster_idx
    4. Genuinely new clusters (no match) get next available index

    At K ≤ 7 this is trivially fast (7×7 matrix).

    Reference: Doc 06 §"Clustering specifics" — "persist cluster→medoid-paper-id
    mapping across reclusterings and use Hungarian matching against previous medoids."
    """
```

**Algorithm walkthrough**:

1. Load previous clusters from SQLite via `load_clusters_from_db(user_id)`
2. If `old_clusters is None` (first time): no remapping needed, return as-is
3. Build a cost matrix of shape `(K_new, K_old)`:
   - For each pair, fetch the old medoid embedding from `paper_vectors`
   - `cost[i][j] = 1 - cosine_similarity(new_medoid_i, old_medoid_j)`
4. Run `scipy.optimize.linear_sum_assignment(cost_matrix)` — O(K³), trivial at K≤7
5. For matched pairs `(new_i, old_j)` where `cost < 0.5` (cosine sim > 0.5):
   assign `new_clusters[new_i].cluster_idx = old_clusters[old_j]['cluster_idx']`
6. For unmatched new clusters: assign the next available index

**Where it's called** — in `_multi_interest_recommend()` in `recommendations.py`:

```python
# Step 1: Compute interest clusters
clusters = compute_clusters(aligned_ids, aligned_embs)

# Step 1.5: Stabilise cluster IDs against previous run
old_clusters = await load_clusters_from_db(user_id)
clusters = stabilize_cluster_ids(clusters, old_clusters, vectors)

# Step 1.6: Persist (now with stable IDs)
await save_clusters_to_db(user_id, clusters)
```

### What Needs to Change

The old medoid embeddings need to be compared against new medoid embeddings. The old
medoid embeddings aren't stored in SQLite (only the `medoid_paper_id` is). Two options:

**Option A** (recommended): Use the `paper_vectors` dict that's already loaded at the
top of `_multi_interest_recommend()` (line 128: `vectors = await qdrant_svc.get_paper_vectors(positives)`).
Old medoid paper IDs are likely in this set since the medoid IS a saved paper. If not,
do a small `get_paper_vectors([old_medoid_id])` call.

**Option B**: Store medoid embeddings as BLOBs in `user_clusters` table. This adds a
4KB column but avoids any Qdrant call. Overhead is negligible.

**Decision**: Option A — avoids schema migration and the vectors are already in memory.

---

## 4.4 — Category-Level Negative Suppression

### Design Decisions (Per User Input)

1. **Primary category only** — arXiv papers have multiple categories (e.g., `cs.CV`, `cs.AI`).
   Suppression applies to the **primary category only** to avoid suffocating the recommendation
   graph. A paper tagged `[cs.CV, cs.AI]` is only suppressed if `cs.CV` (primary) is
   suppressed, not if `cs.AI` is.

2. **τ_neg = 14 days** — Standard default from the literature. If a user dismisses ≥3 papers
   from the same primary category within 14 days, that category is suppressed for 14 days
   from the last dismissal.

### ⚠️ Critical Implementation Detail: Category Format Mismatch

The arXiv API and Turso store categories in **different formats**:
- **arXiv API** (`arxiv_svc.py`): uses arXiv codes like `cs.CV`, `cs.CL`, `stat.ML`
- **Turso** (`turso_svc.py`): uses `primary_topic` which contains human-readable labels
  like `"AI/ML"`, `"Computer Vision"`, `"NLP/Computational Linguistics"`
- Both write to `paper_metadata.category` via different paths

This means `paper_metadata.category` contains a **mix of both formats** depending on
which service populated it. The suppression logic must handle this:

```python
# In the suppression filter, normalise category comparison:
# - Papers from arXiv have codes: "cs.CV"
# - Papers from Turso have labels: "Computer Vision"
# Both may appear in suppressed_cats, so we suppress on exact match
```

**Resolution**: The `get_suppressed_categories()` query will return whatever format is
in the database. The filter in `recommendations.py` will compare candidate categories
(from Turso metadata) against the suppressed set. Since recommendations primarily use
Turso for metadata, the formats will match. For the rare arXiv-fallback case, we accept
the slight inconsistency — it's a minor gap that self-corrects as more Turso data is used.

### What's Already Done

The EWMA negative profile is already wired as Feature 5 in `reranker.py`:
```python
# Feature 5: cosine_sim_negative (0.15 penalty weight)
neg_penalty = cosine_sim(candidate, neg_profile) * 0.15
final_score -= neg_penalty
```

This gives a "soft" directional signal: papers semantically similar to dismissed papers
get demoted. What's missing is the "hard" category-level suppression.

### What's NOT Being Done (Deferred)

**Per-item temporal decay** (`score -= α × exp(-dt / τ)`) is deferred to Phase 6.
Reasoning:
- Requires per-dismissed-item timestamps matched against candidates
- Most naturally expressed as a LightGBM feature (`days_since_most_recent_similar_dismissal`)
- The EWMA negative penalty already covers the directional signal
- Adding hand-tuned temporal formulas when LightGBM is the next phase would create throwaway code

### Modified File: `app/db.py`

Add one new helper function:

```python
async def get_suppressed_categories(
    user_id: str,
    threshold: int = 3,
    days: int = 14,
) -> set[str]:
    """
    Find primary arXiv categories where the user has dismissed ≥ threshold
    papers within the last `days` days.

    Joins interactions (event_type='not_interested') against paper_metadata
    to get the category of each dismissed paper.

    Returns: set of category strings to suppress (e.g., {'cs.CV', 'physics.optics'})
    """
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """SELECT pm.category, COUNT(*) as cnt
               FROM interactions i
               JOIN paper_metadata pm ON i.paper_id = pm.arxiv_id
               WHERE i.user_id = ?
                 AND i.event_type = 'not_interested'
                 AND i.timestamp >= datetime('now', ?)
               GROUP BY pm.category
               HAVING cnt >= ?""",
            (user_id, f"-{days} days", threshold),
        )
        rows = await cur.fetchall()
        return {row[0] for row in rows if row[0]}
```

**Data dependency**: This requires dismissed papers to have their metadata in
`paper_metadata`. Currently:
- Papers from **arXiv API** (`arxiv_svc.py`) are automatically cached via `db.cache_metadata()`
- Papers from **Turso** (`turso_svc.py`) are **NOT cached** to `paper_metadata`

This is a gap. When a user dismisses a paper whose metadata came from Turso (the common
case since Phase 3.5), the category won't be in `paper_metadata` and the suppression
join will miss it.

**Fix**: Add a `cache_turso_metadata()` helper in the recommendations router that writes
Turso-sourced paper dicts to `paper_metadata` after fetching. This is a small INSERT OR
IGNORE — ~1ms overhead for 20 papers. We should also add this to `search.py` and
`saved.py` so ALL metadata paths feed the cache.

### Modified File: `app/routers/recommendations.py`

In `_multi_interest_recommend()`, after re-ranking but before MMR:

```python
# Step 3.5: Category suppression
suppressed_cats = await db.get_suppressed_categories(user_id)
if suppressed_cats:
    # Filter out candidates whose primary category is suppressed
    reranked_ids_filtered = []
    reranked_scores_filtered = []
    reranked_embs_list = []
    for i, rid in enumerate(reranked_ids):
        cat = cand_meta.get(rid, {}).get("category", "")
        # Extract primary category (first in the list, or the whole string)
        primary_cat = cat.split()[0] if cat else ""
        if primary_cat not in suppressed_cats:
            reranked_ids_filtered.append(rid)
            reranked_scores_filtered.append(reranked_scores[i])
            reranked_embs_list.append(reranked_embs[i])

    if reranked_ids_filtered:
        reranked_ids = reranked_ids_filtered
        reranked_scores = reranked_scores_filtered
        reranked_embs = np.array(reranked_embs_list, dtype=np.float32)
```

---

## What Does NOT Change

These are explicitly out of scope for Phase 4:

| Component | Why it stays |
|---|---|
| **Search pipeline** (`search.py`, `hybrid_search_svc.py`) | RRF is correct for search (different retrievers, same query) |
| **α_long = 0.03** (`profiles.py`) | Already fixed in Phase 2a |
| **L2 normalization** (`clustering.py`) | Already applied before Ward in Phase 2b |
| **Negative EWMA Feature 5** (`reranker.py`) | Already wired in Phase 2c |
| **`qdrant_svc.multi_interest_search()`** | Kept in codebase, just no longer called by recs |
| **Per-item temporal decay** | Deferred to Phase 6 (LightGBM feature) |
| **Templates / UI** | No frontend changes |
| **Infrastructure** | Same deployment, same databases |

---

## Files Changed — Complete Map

| File | Action | Lines Changed (est.) | What Changes |
|---|---|---|---|
| `app/recommend/fusion.py` | **NEW** | ~60 | `allocate_quotas()` function |
| `app/routers/recommendations.py` | **MODIFY** | ~40 | Replace RRF call with quota + parallel search; add category suppression |
| `app/recommend/clustering.py` | **MODIFY** | ~50 | Add `stabilize_cluster_ids()` with Hungarian matching |
| `app/db.py` | **MODIFY** | ~20 | Add `get_suppressed_categories()` |
| `tests/test_fusion.py` | **NEW** | ~80 | Unit tests for quota allocation |
| `tests/test_clustering.py` | **MODIFY** | ~30 | Add test for Hungarian matching stability |
| `tests/test_search_router.py` | **NO CHANGE** | 0 | Search pipeline untouched |
| `tests/test_integration.py` | **NO CHANGE** | 0 | Integration tests use mocks, unaffected |

**Total new/modified production code**: ~170 lines  
**Total new test code**: ~110 lines

---

## Implementation Order

Each step leaves the app in a working state. Tests pass after every step.

### Step 1 — Create `fusion.py` + unit tests (~30 min)

Build `allocate_quotas()` in isolation with thorough unit tests:

- `test_basic_allocation` — 3 clusters, verify proportionality
- `test_floor_enforcement` — tiny cluster still gets `min_slots`
- `test_total_equals_requested` — sum always equals `total_slots`
- `test_single_cluster` — all slots go to the one cluster
- `test_equal_importances` — even split
- `test_many_clusters_with_floor` — 7 clusters, floor forces redistribution

### Step 2 — Refactor `_multi_interest_recommend()` (~1 hour)

Replace the RRF call with quota + `asyncio.gather()`. Key changes:
1. Remove `_CLUSTER_LIMITS` hardcoded list
2. Import `allocate_quotas` from `fusion.py`
3. Replace `multi_interest_search()` with per-cluster `search_by_vector()` calls
4. Add deduplication logic
5. Wire short-term vector as a separate search

**Test**: Run `python -m pytest tests/ -v` — all tests must pass.

### Step 3 — Add Hungarian matching to `clustering.py` (~1 hour)

1. Add `stabilize_cluster_ids()` function
2. Call it in `_multi_interest_recommend()` between `compute_clusters()` and `save_clusters_to_db()`
3. Add test: create clusters, slightly perturb, verify indices preserved

**Test**: Run `python -m pytest tests/test_clustering.py -v`

### Step 4 — Add category suppression (~30 min)

1. Add `get_suppressed_categories()` to `db.py`
2. Add suppression filter in `_multi_interest_recommend()` after reranking
3. Ensure Turso metadata is cached to `paper_metadata` for the join to work

**Test**: Run full `python -m pytest tests/ -v`

### Step 5 — End-to-end verification (~30 min)

1. Run `python test_e2e_recs.py` — verify recommendations generate correctly
2. Verify latency stays comparable (~7-8s end-to-end including network I/O)
3. Run full `python -m pytest tests/ -v` — 125+ tests, zero regressions

---

## Test Plan

### New Unit Tests: `tests/test_fusion.py`

| Test | What it verifies |
|---|---|
| `test_basic_proportional_allocation` | 3 clusters with [0.5, 0.3, 0.2] → ~[50, 30, 20] slots |
| `test_floor_protects_minority` | Tiny importance still gets ≥ `min_slots` |
| `test_sum_always_equals_total` | No slots lost or gained during allocation |
| `test_single_cluster` | One cluster gets all slots |
| `test_equal_importances` | N clusters get total/N each |
| `test_remainder_distribution` | Remainder goes to largest fractional part |

### New Unit Test: `tests/test_clustering.py`

| Test | What it verifies |
|---|---|
| `test_hungarian_preserves_indices` | Slight perturbation doesn't shuffle indices |

### Regression

- All 125 existing tests must pass
- `test_e2e_recs.py` must complete successfully

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **Concurrent searches slower than prefetch** | Higher latency | `asyncio.gather()` runs them truly concurrently. Each is ~5-15ms. Wall-clock ~ max(all), not sum(all). |
| **Floor forces too many slots** | With 7 clusters, floor=3 requires 21 minimum slots. If total<21... | `allocate_quotas()` will clamp: if `K × min_slots > total`, reduce floor proportionally. At `total_slots=100` and `MAX_CLUSTERS=7`, minimum is 21, well within budget. |
| **Hungarian matching with different K** | New clustering produces fewer/more clusters than before | Handle rectangular cost matrices. `linear_sum_assignment` natively supports non-square matrices. Unmatched new clusters get fresh indices. |
| **`paper_metadata` missing for suppression join** | `get_suppressed_categories()` returns empty set | **Real gap found** — Turso metadata is not cached to `paper_metadata`. Fix: add `cache_turso_metadata()` calls in search/rec/saved routers. |
| **Turso categories vs arXiv categories format** | Turso stores human-readable categories ("AI/ML"), arXiv uses codes ("cs.AI") | **Real gap found** — both formats coexist in `paper_metadata.category`. Suppression will work within each format. Cross-format inconsistency is minor and self-corrects as Turso dominates. |
| **`search_by_vector` already does 2× over-fetch internally** | Asking for `quota*3` then `search_by_vector` internally doubles it | **Real gap found** — `search_by_vector()` at line 234 already fetches `limit*2` when `exclude_ids` is set. So asking for `quota*3` will actually fetch `quota*6` from Qdrant. This is fine (more candidates for reranker) but should be noted for tuning. |

---

## Verification Checklist

Before declaring Phase 4 complete:

- [ ] `python -m pytest tests/ -v` — all tests pass (130+ including new tests)
- [ ] `test_fusion.py` — 6+ quota allocation tests pass
- [ ] `test_clustering.py` — Hungarian matching test passes
- [ ] `test_e2e_recs.py` — end-to-end recommendations generate correctly
- [ ] Recommendations include papers from minority clusters (quota working)
- [ ] Cluster indices remain stable across consecutive saves
- [ ] Category suppression activates after ≥3 dismissals of same category
- [ ] Search pipeline is completely unaffected (RRF still used for search)
- [ ] Latency comparable to Phase 3.5 baseline
- [ ] All 3 recommendation tiers still cascade correctly (Tier 1 → 2 → 3)

---

## References

- PinnerSage (Pal et al., KDD 2020) — Ward + medoid + importance sampling, no RRF
- Taobao ULIM (Meng et al., RecSys 2025) — quota allocation, +5.54% clicks
- Pinterest Bucketized-ANN (SIGIR 2023) — minority representation protection
- Twitter kNN-Embed (arXiv:2205.06205) — per-cluster proportional drawing
- Bruch et al. (SIGIR 2022) — RRF optimises Recall not nDCG
- YouTube (Xia et al., 2023) — 3× gain from richer negative treatment
- Doc 06 §"The fusion fault in Doc 03" — full RRF critique
- Doc 06 §"Clustering specifics" — Hungarian matching recommendation
- Doc 06 §"Negative signals" — three-layer negative design

---

*Last updated: 2026-04-23*
