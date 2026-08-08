# Phase 6.5 — Implementation Plan

> **Source:** `docs/phases/PHASE6.5-Instrumentation-Framing.md`
> **Timeline:** 5 days (each day leaves the app in a working state)
> **Prerequisite for:** Phase 7 (Evaluation Framework)

---

## Day 1: Phase 6 Hot-fix (A1 + A2)

### A1: Real Qdrant Cosine Scores (Feature 0 fix)

**Problem:** `recommendations.py:329-339` fakes Qdrant scores with linear rank decay (`1.0 - rank * 0.01`). Feature 0 is the model's #5 most important feature — it should be real cosines from Qdrant.

**Root cause:** The search calls use `search_by_vector()` (returns `list[str]`) instead of `search_by_vector_with_scores()` (returns `list[dict]` with `{"arxiv_id": str, "score": float}`).

---

#### [MODIFY] [recommendations.py](file:///c:/Users/siddh/ResearchIT-Final/app/routers/recommendations.py)

**Change 1 — Per-cluster searches (line 258-266):**
Switch from `search_by_vector()` to `search_by_vector_with_scores()`:

```diff
-        search_tasks = [
-            qdrant_svc.search_by_vector(
-                query_vector=c.medoid_embedding.tolist(),
-                limit=quota * _OVERSAMPLE,
-                exclude_ids=seen,
-            )
-            for c, quota in zip(clusters, quotas)
-        ]
-        per_cluster_results = await asyncio.gather(*search_tasks)
+        search_tasks = [
+            qdrant_svc.search_by_vector_with_scores(
+                query_vector=c.medoid_embedding.tolist(),
+                limit=quota * _OVERSAMPLE,
+                exclude_ids=seen,
+            )
+            for c, quota in zip(clusters, quotas)
+        ]
+        per_cluster_scored = await asyncio.gather(*search_tasks)
```

**Change 2 — Build `paper_cluster_map` AND `qdrant_score_map` in one pass (line 268-277):**

```diff
-        paper_cluster_map: dict[str, int] = {}
-        for cluster, result_ids in zip(clusters, per_cluster_results):
-            for aid in result_ids:
-                if aid not in paper_cluster_map:
-                    paper_cluster_map[aid] = cluster.cluster_idx
-
-        candidate_ids = merge_quota_results(list(per_cluster_results), quotas)
+        paper_cluster_map: dict[str, int] = {}
+        qdrant_score_map: dict[str, float] = {}
+        for cluster, scored_results in zip(clusters, per_cluster_scored):
+            for hit in scored_results:
+                aid = hit["arxiv_id"]
+                if aid not in paper_cluster_map:
+                    paper_cluster_map[aid] = cluster.cluster_idx
+                # Keep highest cosine if paper appears in multiple clusters
+                if aid not in qdrant_score_map or hit["score"] > qdrant_score_map[aid]:
+                    qdrant_score_map[aid] = float(hit["score"])
+
+        # merge_quota_results expects list[list[str]] — extract IDs
+        per_cluster_ids = [[h["arxiv_id"] for h in scored] for scored in per_cluster_scored]
+        candidate_ids = merge_quota_results(per_cluster_ids, quotas)
```

**Change 3 — Short-term supplement search (line 280-290):**
Also switch to scored search:

```diff
-            st_results = await qdrant_svc.search_by_vector(
+            st_scored = await qdrant_svc.search_by_vector_with_scores(
                 query_vector=st_vec.tolist(),
                 limit=_ST_SUPPLEMENT,
                 exclude_ids=seen_so_far,
             )
-            for aid in st_results:
-                if aid not in set(candidate_ids):
-                    candidate_ids.append(aid)
+            for hit in st_scored:
+                aid = hit["arxiv_id"]
+                if aid not in set(candidate_ids):
+                    candidate_ids.append(aid)
+                    if aid not in qdrant_score_map:
+                        qdrant_score_map[aid] = float(hit["score"])
                     paper_cluster_map[aid] = -1  # short-term supplement
```

**Change 4 — Delete fake score block (line 329-339):**
The entire synthetic-decay block becomes dead code. Delete it:

```diff
-        # Build qdrant_score_map from per_cluster_results
-        # per_cluster_results is list[list[str]] — we need scores too.
-        # Use the paper_cluster_map to approximate: score = 1.0 - (rank / total)
-        # for now, as the current retrieval path returns only IDs.
-        # TODO: Phase 6.2+ switch to search_by_vector_with_scores()
-        qdrant_score_map: dict[str, float] = {}
-        for cluster_ids in per_cluster_results:
-            for rank, aid in enumerate(cluster_ids):
-                if aid not in qdrant_score_map:
-                    # Approximate score from rank position (higher rank = higher score)
-                    qdrant_score_map[aid] = max(0.0, 1.0 - rank * 0.01)
```

The existing `qdrant_scores = np.asarray(...)` on line 341-344 stays as-is — it reads from `qdrant_score_map` which now has real cosines.

### A2: Verify `/healthz/reranker` live

> ✅ **Already done.** Verified 2026-05-03: `model_loaded: true, n_trees: 141, fallback_active: false`.

Just need to add the timestamp to `PHASE6-Reranker-Framing.md`.

---

## Day 2: B1 — `query_id` Linkage

### What it enables
Per-feed CTR: "out of 30 papers shown in this request, how many got saved?"

### Current state verified
- `interactions` table already has a `query_id TEXT` column ✅ (line 31 in DDL)
- `db.log_interaction()` already accepts `query_id` ✅ (line 135)
- `events.py` already accepts and forwards `query_id` via `Form(default="")` ✅ (line 26)
- **Missing:** `recommendations.py` never generates or passes `query_id`. Search router never generates one either. Templates don't carry it.

---

#### [MODIFY] [recommendations.py](file:///c:/Users/siddh/ResearchIT-Final/app/routers/recommendations.py)

**1. Generate `query_id` at the top of `get_recommendations()` (line 59):**

```python
query_id = str(uuid.uuid4())
```

**2. Thread `query_id` into `paper_tags` in all 3 tiers:**

- Tier 1: In `_multi_interest_recommend()` return value, add `"query_id": query_id` to each tag dict (line 455-458)
- Tier 2: EWMA fallback tags (line 116-120) — add `"query_id": query_id`
- Tier 3: Qdrant recommend tags (line 131-135) — add `"query_id": query_id`
- Trending fallback (line 85-87) — add `"query_id": query_id`

**3. Embed `query_id` + `position` into paper dicts (line 153-166):**

```python
for idx, aid in enumerate(rec_arxiv_ids):
    ...
    papers.append({
        **meta[aid],
        "saved": False,
        "dismissed": False,
        "ranker_version": tags.get("ranker_version", _RANKER_VERSION),
        "candidate_source": tags.get("candidate_source", ""),
        "cluster_id": tags.get("cluster_id", ""),
        "query_id": tags.get("query_id", ""),       # NEW
        "position": idx,                              # NEW
    })
```

> [!IMPORTANT]
> The `_multi_interest_recommend` signature needs updating to accept `query_id` as a parameter, since it's where the Tier 1 paper_tags are built. Alternatively, we generate `query_id` inside it and return it alongside the tags. I'll use the approach of passing it as a param.

---

#### [MODIFY] [search.py](file:///c:/Users/siddh/ResearchIT-Final/app/routers/search.py)

**Generate `query_id` per search and embed in paper dicts (line 70-77):**

```python
query_id = str(uuid.uuid4())  # generated once per /search request

for idx, p in enumerate(papers):
    p["saved"] = p["arxiv_id"] in saved_ids
    p["dismissed"] = p["arxiv_id"] in dismissed_ids
    p["query_id"] = query_id        # NEW
    p["position"] = idx             # NEW
```

---

#### [MODIFY] [action_buttons.html](file:///c:/Users/siddh/ResearchIT-Final/app/templates/partials/action_buttons.html)

**Add `query_id` and `position` to ALL three `hx-vals` JSON blobs:**

Add to template header:
```jinja2
{% set _query_id = paper.query_id | default("") if paper is defined else "" %}
{% set _position = paper.position | default(0) if paper is defined else 0 %}
```

Add to each `hx-vals`:
```
"query_id": "{{ _query_id }}", "position": "{{ _position }}"
```

The save button (line 37) already has `position` — update to use `_position`. The not-interested buttons (line 26, 45) need `query_id` and `position` added.

---

## Day 3: B2 — Propensity Logging

### What it enables
Counterfactual evaluation (SNIPS estimator) — "what would have happened with ranker B?"

---

#### [MODIFY] [db.py](file:///c:/Users/siddh/ResearchIT-Final/app/db.py)

**1. Migration (after `_MIGRATION_6_3`):**
```python
_MIGRATION_6_5 = [
    "ALTER TABLE interactions ADD COLUMN propensity REAL",
    "ALTER TABLE interactions ADD COLUMN policy_id TEXT",
]
```

**2. Run in `init_db()`.**

**3. Extend `log_interaction()` signature (line 129-149):**
Add `propensity: float | None = None` and `policy_id: str | None = None` kwargs. Extend the INSERT.

---

#### [MODIFY] [recommendations.py](file:///c:/Users/siddh/ResearchIT-Final/app/routers/recommendations.py)

**Compute propensity after `inject_exploration()` (line 443):**

```python
# Exploration papers: uniformly sampled from pool
explore_pool_size = max(1, len(reranked_ids) - len(mmr_selected))
explore_propensity = len(exploration_set) / explore_pool_size if explore_pool_size > 0 else 0.0

# Exploitation (MMR-selected): deterministic → propensity = 1.0
for aid in final:
    paper_tags[aid]["propensity"] = (
        explore_propensity if aid in exploration_set else 1.0
    )
    paper_tags[aid]["policy_id"] = _RANKER_VERSION
```

Thread `propensity` and `policy_id` into template context the same way as `query_id`.

---

#### [MODIFY] [search.py](file:///c:/Users/siddh/ResearchIT-Final/app/routers/search.py)

Search is fully deterministic → `propensity = 1.0` for all results.

---

#### [MODIFY] [action_buttons.html](file:///c:/Users/siddh/ResearchIT-Final/app/templates/partials/action_buttons.html)

Add `propensity` and `policy_id` to `hx-vals`.

---

#### [MODIFY] [events.py](file:///c:/Users/siddh/ResearchIT-Final/app/routers/events.py)

Add `propensity: float = Form(default=0.0)` and `policy_id: str = Form(default="")` to both endpoints. Forward to `db.log_interaction()`.

---

## Day 4: B3 — Cluster Snapshot Versioning

### What it enables
Cluster history, debugging "why did recs shift?", content-addressed key for Phase 8a LLM summary cache.

---

#### [MODIFY] [db.py](file:///c:/Users/siddh/ResearchIT-Final/app/db.py)

**1. Add `cluster_snapshots` DDL to `_SCHEMA`:**
```sql
CREATE TABLE IF NOT EXISTS cluster_snapshots (
    user_id              TEXT NOT NULL,
    snapshot_id          TEXT NOT NULL,
    cluster_idx          INTEGER NOT NULL,
    medoid_paper_id      TEXT NOT NULL,
    importance           REAL NOT NULL,
    paper_ids            TEXT NOT NULL,
    medoid_embedding_blob BLOB,
    snapshot_date        TEXT NOT NULL DEFAULT (datetime('now')),
    paper_ids_hash       TEXT NOT NULL,
    PRIMARY KEY (user_id, snapshot_id, cluster_idx)
);
CREATE INDEX IF NOT EXISTS idx_snap_user_date ON cluster_snapshots(user_id, snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_snap_hash ON cluster_snapshots(paper_ids_hash);
```

**2. Add `save_cluster_snapshot()` and `prune_old_snapshots()` functions.**

---

#### [MODIFY] [recommendations.py](file:///c:/Users/siddh/ResearchIT-Final/app/routers/recommendations.py)

After `save_clusters_to_db(user_id, clusters)` (line ~253), call `db.save_cluster_snapshot()`.

---

#### [MODIFY] [main.py](file:///c:/Users/siddh/ResearchIT-Final/app/main.py)

Call `db.prune_old_snapshots(retention_days=30)` in the lifespan handler after `init_db()`.

---

## Day 5: B4 — Semantic Scholar Author Import

### What it enables
"Paste S2 URL → 20 implicit saves" — replaces manual seed search friction.

---

#### [NEW] [s2_svc.py](file:///c:/Users/siddh/ResearchIT-Final/app/s2_svc.py)

Functions:
- `parse_author_input(text) → str | None` — accepts S2 URL, raw S2 ID, or ORCID
- `resolve_orcid(orcid) → str | None` — resolves ORCID via S2 author search
- `fetch_author_arxiv_papers(author_id, limit=50) → list[str]` — returns arXiv IDs

---

#### [MODIFY] [config.py](file:///c:/Users/siddh/ResearchIT-Final/app/config.py)

Add `S2_API_KEY = os.getenv("S2_API_KEY", "")` — key already in `.env`.

---

#### [MODIFY] [onboarding.py](file:///c:/Users/siddh/ResearchIT-Final/app/routers/onboarding.py)

Add `POST /api/onboarding/import-author` endpoint.

---

#### [NEW] Template partials for import step

- `partials/import_author.html` — the import form step
- `partials/import_success.html` — success confirmation
- `partials/import_error.html` — error message

---

## Verification Plan

### Automated Tests

After each day:

```bash
python -m pytest tests/ -v --tb=short
```

**New test files:**
- Day 1: Add `test_qdrant_scores_are_real_cosines` to `tests/test_phase6_feature_wiring.py`
- Day 2: Create `tests/test_instrumentation.py` — `test_query_id_round_trips`
- Day 3: Add `test_propensity_sums_correctly` to instrumentation tests
- Day 4: Add `test_snapshot_appended_on_each_recluster`, `test_prune_respects_retention`
- Day 5: Add `test_s2_import_saves_papers_with_correct_source_tag`

### Manual Verification

- Day 1: `curl -s https://siddhm11-researchit.hf.space/healthz/reranker` — confirm model still loaded after code change
- Day 5: Test author import with real S2 profile URL

---

## Documentation Updates (after all days)

- [ ] CLAUDE.md: Add Rule 3.11 — "Every interaction must carry `query_id`, `propensity`, and `policy_id`"
- [ ] TASK-TRACKER.md: Add Phase 6.5 section with checklist
- [ ] README.md: Update test count
- [ ] PHASE6-Reranker-Framing.md: Add live verification timestamp

---

## Open Questions

> [!IMPORTANT]
> **Q1:** The framing doc proposes `_RANKER_VERSION` as the `policy_id`. Currently it's `"v4.1_quota_hungarian_suppression"`. Should we also bump this to `"v6.5_lightgbm_real_cosines"` when Day 1 lands? It would make A/B-style log analysis cleaner.

> [!IMPORTANT]
> **Q2:** Day 5 (S2 author import) requires `httpx` as a dependency. It's already used by `turso_svc.py`, so no new install needed — just confirming.

> [!NOTE]
> **Q3:** The framing doc suggests cluster snapshot pruning at startup. For a simple MVP this is fine. Phase 7 can upgrade to APScheduler if needed.
