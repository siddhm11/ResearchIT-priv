# PHASE 6.5 — Instrumentation Framing

> **Status:** 📋 Proposed (not started)
> **Scope:** Phase 6 hot-fix (Day 1) + Phase 6.5 instrumentation (Days 2–4) + Phase 5.1 cold-start completion (Day 5, parallel)
> **Prerequisite for:** Phase 7 (Evaluation Framework)
> **Supersedes:** Open items at the end of `PHASE6-Reranker-Framing.md` (Section E.1.a, E.2 verification, ADR A1/A4 deferrals)
> **Owner:** Amin
> **Authoring date:** 2026-05-03

---

## TL;DR

Phase 6 is **substantively complete** but has two open flags. Phase 7 (evaluation framework) cannot be built cleanly on top of the current schema — three pieces of telemetry are missing. This doc bundles three coherent units of work:

| Bucket                    | Identity                          | Days | Why it's separate                                                                                                  |
| ------------------------- | --------------------------------- | ---- | ------------------------------------------------------------------------------------------------------------------ |
| **Phase 6 Hot-fix**       | Close out Phase 6 cleanly         | 1    | Two correctness/verification items left over from PHASE6-Reranker-Framing. Belongs to Phase 6, not later.          |
| **Phase 6.5**             | Telemetry foundation              | 3    | Mirrors the Phase 4.5 precedent: a small instrumentation phase that exists *because* the next phase needs it.      |
| **Phase 5.1 (side-quest)** | Cold-start completion             | 1    | Author-import was the deferred Layer 2 from Phase 5's three-layer onboarding plan. Sits beside, not inside, 6.5.   |

Total: **5 working days**. After this, Phase 7 starts on a clean substrate where all the prerequisite plumbing is already in production.

---

## 1. Why this doc exists (the reasoning)

The instinct is to fold all five days of work into Phase 7 — it's all "stuff that helps evaluation," after all. That instinct is wrong, and the reason matters.

**Phases in this project have always had one identity.** Look at the existing pattern:

- Phase 4 = quota fusion
- Phase 4.5 = instrumentation only (`ranker_version`, `candidate_source`, `cluster_id`)
- Phase 5 = onboarding
- Phase 6 = LightGBM reranker integration
- Phase 7 = **evaluation framework** (per master roadmap: nDCG@10, Recall@50, HR@10, ILS, category entropy, time-split eval, regression CI)

Phase 4.5 is the precedent. When instrumentation needed to land between Phase 4 and Phase 5, it didn't get folded into either — it got its own micro-phase precisely because it was load-bearing for everything downstream and had a single identity. The framing doc for Phase 6 (Part H) was also explicit about what Phase 6 is NOT — and "evaluation harness" was carved into Phase 7 deliberately.

**What happens if we fold everything into Phase 7?** Phase 7's master-roadmap budget is ~1 week. Adding ~3 days of prerequisite infrastructure either:

1. Bloats Phase 7 to 2+ weeks, or
2. Forces shortcuts on the actual harness work (offline regression, time-split eval, frozen `eval/eval_set_v1.0.parquet`, CI gates on >3% nDCG@10 drops) — which is a meaty deliverable in its own right.

**What happens if we leave the Phase 6 closeout for later?** The biggest item in the closeout is the `qdrant_cosine_score` fix — and that's a model-correctness bug. Feature 0 is the reranker's #5 most-important feature by training importance, and right now it's being fed synthetic linear decay (`1.0 - rank * 0.01`) instead of actual cosines. Every day it sits unfixed, the model is performing below its training-distribution capability. This belongs to Phase 6, full stop.

**What happens if the cold-start work waits?** B4 (S2 author import) is the single biggest cold-start lift available — replacing "manually save 5 papers" with "paste your S2 URL → 20 saves." It's a Phase 5 completion, not a Phase 7 input. It can run in parallel with Phase 6.5 work because it touches a different code path (onboarding router, no schema changes to `interactions`).

**The structural answer:** three identities → three buckets. This doc unifies them under one plan with one timeline.

---

## 2. Phase 6 Audit — current status

Cross-checked against `PHASE6-Reranker-Framing.md` (Parts A–G) and current code. Audit was performed 2026-05-03.

### ✅ Phase 6.1 — Simplification Pass: DONE

In `app/routers/recommendations.py`:

- `suppressed` and `onboarding_categories` loaded **before** the rerank call
- `qdrant_score_map` built from `per_cluster_results`
- `user_total_saves` / `user_total_dismissals` computed and passed
- `is_suppressed_arr` and `onboarding_match_arr` computed per-candidate
- `rerank_candidates` called with the full Phase 6 kwarg signature

### ✅ Phase 6.2 — Per-Candidate Plumbing: DONE

- `paper_cluster_map` is built before the merge — first-occurrence wins, exactly per spec
- `per_candidate_importance` is a `(N,)` array, not a scalar
- `per_candidate_medoids` is a `(N, 1024)` stack, not broadcast
- `app/recommend/reranker.py:287–298` slot 24 correctly handles both 1D (broadcast) and 2D (per-candidate) medoid shapes
- `test_phase6_feature_wiring.py::test_per_candidate_cluster_importance` and `test_per_candidate_medoid_distance` exist

### ✅ Phase 6.3 — Deployment Verification: DONE (code), ⚠️ UNVERIFIED (live)

- `/healthz/reranker` endpoint exists in `app/routers/health.py`
- `is_model_loaded()`, `get_loaded_model_path()`, `get_num_trees()` accessors exist in `reranker.py`
- Per-request feature activation logging at `reranker.py:432–438`
- Bug B fix: `medoid_embedding_blob BLOB` column added via migration in `db.py:128`
- Hungarian fallback now prefers live vector → persisted blob → skip with warning

### ⚠️ Two flags from Phase 6 (handled in §3 below)

1. **`qdrant_scores` are still rank-approximated, not real cosines.** `recommendations.py:316–325` uses synthetic linear decay because the call site is still on `search_by_vector()` (returns `list[str]`) instead of `search_by_vector_with_scores()` (returns `[{"arxiv_id": ..., "score": ...}]`). The scored function already exists in `qdrant_svc.py:265` — the swap is mechanical.
2. **`/healthz/reranker` not curl-verified live.** The endpoint exists in code. Production status is unknown — could be silently running heuristic fallback if the model file isn't being copied into the Docker image.

### ✅ Phase 6.4 — Retraining: correctly deferred

Documented in `PHASE6-Reranker-Framing.md` Section F.4, gated on synthetic simulator OR 100 real users with ≥10 saves each.

### Verdict

Phase 6 is substantively complete. The two flags above are polish, not blockers — but the qdrant-scores fix is feeding the model wrong data for one of its top-importance features and should ship as part of Phase 6 closeout, not deferred.

---

## 3. Bucket 1 — Phase 6 Hot-fix (Day 1)

### 3.1 — A1: Real Qdrant Scores (the lying-feature-0 fix)

**The problem.** In `recommendations.py:248`, the per-cluster search calls `qdrant_svc.search_by_vector()` which returns `list[str]` — arXiv IDs only, no scores. Then around line 316, scores are faked by linear decay from rank position:

```python
qdrant_score_map[aid] = max(0.0, 1.0 - rank * 0.01)
```

A paper at rank 0 gets score 1.0, rank 50 gets 0.50, rank 100 gets 0.0. This bears almost no relationship to actual cosine similarity, where a top result might be 0.85 and rank 50 might be 0.78 — a much tighter band. Feature 0 (`qdrant_cosine_score`) is the model's #5 most-important feature by training importance. Feeding it a synthetic linear sequence caps how much the model can help.

**The fix.** Switch to `search_by_vector_with_scores()` (already exists at `qdrant_svc.py:265`), and build `qdrant_score_map` from actual cosines as part of the same loop that builds `paper_cluster_map`.

**Code change** in `app/routers/recommendations.py`, the `_multi_interest_recommend()` flow around line 245:

```python
# OLD
search_tasks = [
    qdrant_svc.search_by_vector(
        query_vector=c.medoid_embedding.tolist(),
        limit=quota * _OVERSAMPLE,
        exclude_ids=seen,
    )
    for c, quota in zip(clusters, quotas)
]
per_cluster_results = await asyncio.gather(*search_tasks)

# Phase 4.5: Build paper → cluster mapping BEFORE merge
paper_cluster_map: dict[str, int] = {}
for cluster, result_ids in zip(clusters, per_cluster_results):
    for aid in result_ids:
        if aid not in paper_cluster_map:
            paper_cluster_map[aid] = cluster.cluster_idx

# Apply quota merge
candidate_ids = merge_quota_results(list(per_cluster_results), quotas)
```

becomes:

```python
# NEW — fetch scores alongside IDs
search_tasks = [
    qdrant_svc.search_by_vector_with_scores(
        query_vector=c.medoid_embedding.tolist(),
        limit=quota * _OVERSAMPLE,
        exclude_ids=seen,
    )
    for c, quota in zip(clusters, quotas)
]
per_cluster_scored = await asyncio.gather(*search_tasks)

# Build paper → cluster map AND real qdrant_score_map in one pass
paper_cluster_map: dict[str, int] = {}
qdrant_score_map: dict[str, float] = {}
for cluster, scored_results in zip(clusters, per_cluster_scored):
    for hit in scored_results:
        aid = hit["arxiv_id"]
        if aid not in paper_cluster_map:
            paper_cluster_map[aid] = cluster.cluster_idx
        # Keep highest cosine if paper appears in multiple clusters
        if aid not in qdrant_score_map or hit["score"] > qdrant_score_map[aid]:
            qdrant_score_map[aid] = float(hit["score"])

# merge_quota_results expects list[list[str]] — extract IDs
per_cluster_ids = [[hit["arxiv_id"] for hit in scored] for scored in per_cluster_scored]
candidate_ids = merge_quota_results(per_cluster_ids, quotas)
```

Then **delete** the synthetic-score block (current `recommendations.py:313–325`):

```python
# DELETE — qdrant_score_map is now built from real cosines above
# qdrant_score_map: dict[str, float] = {}
# for cluster_ids in per_cluster_results:
#     for rank, aid in enumerate(cluster_ids):
#         if aid not in qdrant_score_map:
#             qdrant_score_map[aid] = max(0.0, 1.0 - rank * 0.01)
```

**Don't forget the short-term supplement search.** Around line 263 (the path that pulls extra papers from `state.short_term_centroid` to fill the feed) does the same synthetic-decay trick. Same swap applies, with `paper_cluster_map[aid] = -1` (signalling "not from a long-term cluster") and `qdrant_score_map` populated from real scores.

**Test** (add to `tests/test_phase6_feature_wiring.py`):

```python
def test_qdrant_scores_are_real_cosines_not_rank_proxies():
    """Feature 0 should be actual cosine similarities — not a perfect linear
    sequence from rank 0 → N."""
    # Mock search_by_vector_with_scores to return realistic clustered scores:
    # e.g. [0.91, 0.89, 0.87, 0.86, 0.84, 0.83, ...] not [1.0, 0.99, 0.98, ...]
    fake_hits = [
        {"arxiv_id": f"24{i:02d}.{i:05d}", "score": 0.92 - 0.005 * i + (0.01 if i % 3 == 0 else 0)}
        for i in range(20)
    ]
    # ... call _multi_interest_recommend, capture qdrant_score_map
    # ... assert all values in [0.5, 1.0] (realistic cosine band, not 0.0–1.0 sweep)
    # ... assert NOT a perfect linear sequence (variance > 0 in successive diffs)
    diffs = [s2 - s1 for s1, s2 in zip(scores[:-1], scores[1:])]
    assert max(diffs) - min(diffs) > 0.001, "scores look synthetically linear"
```

**Estimated effort:** 2 hours (including the test).

---

### 3.2 — A2: Verify `/healthz/reranker` Live

**Not a code change** — a 5-minute verification command:

```bash
curl -s https://siddhm11-researchit.hf.space/healthz/reranker | python -m json.tool
```

**Three possible outcomes:**

| Response                                                              | Meaning                                              | Action                                                                                                |
| --------------------------------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `model_loaded: true, n_trees: 141, fallback_active: false`            | ✅ Production is using LightGBM                       | Tick the box in TASK-TRACKER. Add timestamp to PHASE6-Reranker-Framing.md.                            |
| `model_loaded: false, fallback_active: true`                          | ⚠️ Space is silently using the heuristic              | Debug per checklist below.                                                                            |
| 404 or 500                                                            | Endpoint isn't deployed yet                          | Push the latest commit; HF Spaces will rebuild.                                                       |

**If the model isn't loading, debug in this order:**

1. **Is the model file in the Git repo?**
   ```bash
   git ls-files | grep reranker_v1.txt
   ```
   If empty: check `.gitignore` for any pattern that might catch it (e.g. `*.txt` in a subtree, or a too-broad `models/` rule). The current `.gitignore` looks safe but worth double-checking — the file is `models/reranker-phase6/production_model/reranker_v1.txt`.

2. **Is the model file being copied into the Docker image?**
   Check `Dockerfile` for `COPY models/ models/` or `COPY . .`. Check `.dockerignore` for any pattern that excludes `models/` or `*.txt`.

3. **Does the path search in `reranker.py:35–44` find it from HF Spaces' working directory?** If HF Spaces runs from `/app` instead of the repo root, the relative paths might miss. Set `RERANKER_MODEL_PATH` explicitly in HF Secrets:
   ```
   RERANKER_MODEL_PATH=/app/models/reranker-phase6/production_model/reranker_v1.txt
   ```

4. **Check the build logs** for the line `[reranker] LightGBM model loaded from <path> (n_trees=141)`. If that line is missing, the loader is silently failing — turn on DEBUG logging in `reranker.py` to see why.

**If it's working**, update `PHASE6-Reranker-Framing.md` with a one-liner under Section E:

> *Verified live at 2026-MM-DD: `model_loaded=true, n_trees=141, fallback_active=false`.*

**Estimated effort:** 30 minutes including any Docker fixes.

---

## 4. Bucket 2 — Phase 6.5: Instrumentation Foundation (Days 2–4)

This is the new phase. Single identity: **telemetry schema and storage foundations that Phase 7 will sit on top of.** Three pieces of work, each a day, each independently shippable, each leaves the app in a working state.

### 4.1 — B1: query_id Linkage (Day 2)

**Why this matters more than it sounds.** Right now, interaction logs look like this:

```
user_id=u1, paper_id=2401.001, event=save, source=recommendation, candidate_source=cluster_0
user_id=u1, paper_id=2401.002, event=save, source=recommendation, candidate_source=cluster_1
```

You can count saves but you cannot answer:

- *"Out of the 30 papers we showed in this single feed request, how many got saved?"* (CTR per query)
- *"Did this user save the paper from the same feed they saw it in, or come back 3 days later?"* (intra-session vs return)
- *"When ranker version changed, did CTR for the same user change?"* (ranker A/B comparison)

Without `query_id`, every interaction floats free of the request that generated it. Phase 7 evaluation cannot compute even the most basic feed-level metric.

**The fix in 4 steps:**

#### Step 1: Generate `query_id` in `recommendations.py`

At the top of `get_recommendations()`:

```python
import uuid
query_id = str(uuid.uuid4())
```

When building `paper_tags` (the per-paper instrumentation dict already used by Phase 4.5):

```python
paper_tags[aid] = {
    "ranker_version": _RANKER_VERSION,
    "candidate_source": source,
    "cluster_id": str(cluster_idx) if cluster_idx is not None and cluster_idx >= 0 else "",
    "query_id": query_id,            # NEW
    "position": str(position),       # NEW — index in final ranked list (0-based)
}
```

#### Step 2: Same plumbing in `search.py`

Generate one `query_id` per `/search` request, attach to every paper card. Same shape as recommendations — different `source` value (`"search"` not `"recommendation"`) but same fields.

#### Step 3: Template plumbing

In `app/templates/partials/action_buttons.html`, extend the `hx-vals` JSON:

```html
hx-vals='{
  "source": "{{ _source }}",
  "position": "{{ position | default(0) }}",
  "ranker_version": "{{ _ranker_version }}",
  "candidate_source": "{{ _candidate_source }}",
  "cluster_id": "{{ _cluster_id }}",
  "query_id": "{{ paper.query_id | default('') }}"
}'
```

(The Jinja templates that currently render paper cards need the per-card `query_id` and `position` available in their context — pass them in via the loop variable when rendering the feed.)

#### Step 4: events.py forwards the field

`db.log_interaction()` already accepts a `query_id` parameter. Just ensure `events.py` forwards the Form field:

```python
@router.post("/api/events")
async def log_event(
    paper_id: str = Form(...),
    event_type: str = Form(...),
    source: str = Form(default=""),
    position: int = Form(default=0),
    ranker_version: str = Form(default=""),
    candidate_source: str = Form(default=""),
    cluster_id: str = Form(default=""),
    query_id: str = Form(default=""),  # NEW
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    await db.log_interaction(
        user_id=user_id,
        paper_id=paper_id,
        event_type=event_type,
        source=source,
        position=position,
        ranker_version=ranker_version,
        candidate_source=candidate_source,
        cluster_id=cluster_id or None,
        query_id=query_id or None,  # NEW
    )
```

**What this enables in Phase 7.** A single SQL query gives per-feed CTR by ranker version:

```sql
SELECT
    query_id,
    ranker_version,
    COUNT(*) FILTER (WHERE event_type = 'save') * 1.0 / COUNT(DISTINCT paper_id) AS save_rate
FROM interactions
WHERE source = 'recommendation'
GROUP BY query_id, ranker_version;
```

**Test** (add to `tests/test_instrumentation.py`):

```python
async def test_query_id_round_trips_from_request_to_db():
    """A single /api/recommendations call should generate one query_id;
    every paper card returned should carry it; saving any paper should
    persist that exact query_id in interactions."""
    resp = await client.get("/api/recommendations", cookies={"uid": "test-user"})
    # Parse out query_id values from the rendered cards
    query_ids = re.findall(r'"query_id":\s*"([0-9a-f-]{36})"', resp.text)
    assert len(set(query_ids)) == 1, "all cards should share one query_id"
    qid = query_ids[0]

    # Save the first paper
    paper_id = re.search(r'data-paper-id="([^"]+)"', resp.text).group(1)
    await client.post("/api/events", data={
        "paper_id": paper_id, "event_type": "save",
        "source": "recommendation", "query_id": qid,
    })
    rows = await db.fetch_all("SELECT query_id FROM interactions WHERE paper_id = ?", paper_id)
    assert rows[0]["query_id"] == qid
```

**Estimated effort:** 3 hours.

---

### 4.2 — B2: Propensity Logging (Day 3)

**Why this is non-negotiable per the project's own framing doc.** ADR A4 in `PHASE6-Reranker-Framing.md` says verbatim:

> *Telemetry gaps bite hardest in Phase 5 (IPS impossible without propensities): freeze schema before any logging (A4); include policy_id, propensity, shown_position, ranker_version*

You already have `policy_id` in spirit (`ranker_version`) and `shown_position` (`position`). What's missing is `propensity` — the probability that the active policy chose to show this paper to this user in this slot.

Without propensity, **counterfactual evaluation is mathematically impossible**. You can never retrospectively answer "what would have happened if we'd used a different ranker?" because you cannot reweight observed clicks correctly. Adding the column to a table with 50K rows is a multi-week migration project; adding it to an empty table is 4 hours.

#### Schema migration

Add to `app/db.py`:

```python
_MIGRATION_B2 = [
    "ALTER TABLE interactions ADD COLUMN propensity REAL",
    "ALTER TABLE interactions ADD COLUMN policy_id TEXT",
]
```

(`policy_id` is a synonym for `ranker_version` but more honest about what it represents — the identifier of the *full pipeline configuration* that chose to show this paper, including MMR λ, exploration rate ε, and any feature-flag state. Some systems keep both: `ranker_version` for the model file hash, `policy_id` for the pipeline hash. For now they can be the same value, but the column is there when you need to differentiate.)

Run the migration via the existing migration runner pattern in `db.py:128`:

```python
async def _apply_migrations(conn):
    # ... existing migrations ...
    for sql in _MIGRATION_B2:
        try:
            await conn.execute(sql)
        except aiosqlite.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise
    await conn.commit()
```

Update `db.log_interaction()`:

```python
async def log_interaction(
    user_id: str,
    paper_id: str,
    event_type: str,
    *,
    source: str = "",
    position: int = 0,
    ranker_version: str | None = None,
    candidate_source: str | None = None,
    cluster_id: str | None = None,
    query_id: str | None = None,
    propensity: float | None = None,   # NEW
    policy_id: str | None = None,      # NEW
):
    # ... INSERT statement extended with propensity, policy_id ...
```

#### The propensity computation

In `recommendations.py`, after the final feed is built but before tags are returned, compute per-paper propensity. The math depends on which slot the paper occupies:

```python
# Phase 6.5+B2: compute per-paper propensity
N_FINAL = len(final)
N_EXPLORE = len(exploration_set)  # the ε papers MMR didn't pick
N_EXPLOIT = N_FINAL - N_EXPLORE

# Exploration papers: uniformly sampled from `reranked_ids` not in mmr_selected
explore_pool_size = max(1, len(reranked_ids) - len(mmr_selected))
explore_propensity = N_EXPLORE / explore_pool_size if explore_pool_size > 0 else 0.0

# Exploitation papers: deterministically selected by MMR → propensity = 1.0
# (this is the "logging policy = serving policy" case — IPS weight will be 1)

for aid in final:
    paper_tags[aid]["propensity"] = (
        explore_propensity if aid in exploration_set else 1.0
    )
    paper_tags[aid]["policy_id"] = _RANKER_VERSION  # or compute pipeline hash
```

Plumb through templates (add `propensity` and `policy_id` to `hx-vals` like with `query_id`), and store in `events.py`.

**For search**, propensity is `1.0` for every result (search is fully deterministic — no exploration). Set it explicitly so the column is always populated:

```python
# search.py
paper_tags[aid]["propensity"] = 1.0
paper_tags[aid]["policy_id"] = _SEARCH_POLICY_ID
```

#### Why this earns its day

Phase 7 evaluation will eventually want to test "ranker B vs ranker A" without a full A/B test (you don't have user volume for that). With propensity logging, you can use **SNIPS** (Self-Normalized Inverse Propensity Scoring) on existing logs to estimate "what would CTR have been if we'd used ranker B?" — purely from data ranker A already collected. The estimator is:

```
                   Σ_i (r_i × π_B(a_i | x_i) / π_A(a_i | x_i))
SNIPS(π_B) = ─────────────────────────────────────────────────
                       Σ_i (π_B(a_i | x_i) / π_A(a_i | x_i))
```

where `π_A` is the logging policy (your current ranker, propensity stored at log time) and `π_B` is the candidate policy you want to evaluate. Without `π_A` stored at log time, this formula has a missing denominator and the estimator collapses.

**Test:**

```python
async def test_propensity_sums_correctly_across_exploration_and_exploitation():
    """For a feed of N papers with K exploration slots, the sum of propensities
    over ALL candidates in the explore pool should equal K (each paper had K/|pool|
    chance, summed over |pool| papers = K)."""
    # Mock a recommendation flow with N=30, K_explore=2, pool_size=50
    # Capture propensity values
    explore_props = [p["propensity"] for p in tagged if p["aid"] in exploration_set]
    assert all(0 < p <= 1 for p in explore_props)
    # Each exploration paper has propensity = K/pool = 2/50 = 0.04
    assert all(abs(p - 0.04) < 1e-6 for p in explore_props)
    # Exploitation papers all have propensity = 1.0
    exploit_props = [p["propensity"] for p in tagged if p["aid"] not in exploration_set]
    assert all(p == 1.0 for p in exploit_props)
```

**Estimated effort:** 4 hours.

---

### 4.3 — B3: Cluster Snapshot Versioning (Day 4)

**The current problem.** `db.save_user_clusters()` (around `db.py:235`) does:

```python
await conn.execute("DELETE FROM user_clusters WHERE user_id = ?", (user_id,))
for c in clusters:
    await conn.execute("INSERT INTO user_clusters ...")
```

Every recluster, the previous cluster state is **destroyed**. You cannot answer:

- *"What clusters did this user have a week ago?"* — for debugging "why did the recs suddenly shift?"
- *"When did cluster 2 form?"* — for cluster lifecycle analytics
- Phase 8a's content-addressed LLM-summary cache key needs `(cluster_stable_id, snapshot_date)` per ADR A1 — and the snapshot_date doesn't exist as a concept yet

This implements **ADR A1** from `PHASE6-Reranker-Framing.md`.

#### Schema

```sql
CREATE TABLE IF NOT EXISTS cluster_snapshots (
    user_id              TEXT NOT NULL,
    snapshot_id          TEXT NOT NULL,           -- UUID, one per recluster event
    cluster_idx          INTEGER NOT NULL,        -- stable index after Hungarian
    medoid_paper_id      TEXT NOT NULL,
    importance           REAL NOT NULL,
    paper_ids            TEXT NOT NULL,           -- JSON array
    medoid_embedding_blob BLOB,
    snapshot_date        TEXT NOT NULL DEFAULT (datetime('now')),
    paper_ids_hash       TEXT NOT NULL,           -- sha256(sorted(paper_ids))[:16]
    PRIMARY KEY (user_id, snapshot_id, cluster_idx)
);
CREATE INDEX IF NOT EXISTS idx_snap_user_date ON cluster_snapshots(user_id, snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_snap_hash ON cluster_snapshots(paper_ids_hash);
```

`paper_ids_hash` is the content-addressing key — Phase 8a will use this to dedupe LLM-summary generation across users. If two different users have a cluster with identical paper sets, they share one cached summary. The 16-character truncation is enough entropy at our scale (low birthday-collision risk for <100M clusters).

#### Write side

Add a new function in `db.py`:

```python
import json
import hashlib
import uuid

async def save_cluster_snapshot(user_id: str, clusters: list[dict]) -> str:
    """Append a new snapshot. Returns the snapshot_id (one per recluster event)."""
    snapshot_id = str(uuid.uuid4())
    async with aiosqlite.connect(DB_PATH) as conn:
        for c in clusters:
            paper_ids = json.loads(c["paper_ids"]) if isinstance(c["paper_ids"], str) else c["paper_ids"]
            paper_ids_hash = hashlib.sha256(
                json.dumps(sorted(paper_ids)).encode()
            ).hexdigest()[:16]
            await conn.execute(
                """INSERT INTO cluster_snapshots
                   (user_id, snapshot_id, cluster_idx, medoid_paper_id,
                    importance, paper_ids, medoid_embedding_blob, paper_ids_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, snapshot_id, c["cluster_idx"], c["medoid_paper_id"],
                 c["importance"], json.dumps(paper_ids),
                 c.get("medoid_embedding_blob"), paper_ids_hash),
            )
        await conn.commit()
    return snapshot_id
```

In `recommendations.py`, **after** `save_clusters_to_db(user_id, clusters)` (the existing call that maintains the "current state" view), add:

```python
snapshot_id = await db.save_cluster_snapshot(user_id, [
    {
        "cluster_idx": c.cluster_idx,
        "medoid_paper_id": c.medoid_paper_id,
        "importance": c.importance,
        "paper_ids": json.dumps(c.paper_ids),
        "medoid_embedding_blob": c.medoid_embedding.astype(np.float32).tobytes(),
    }
    for c in clusters
])
```

Crucially: keep `save_clusters_to_db` doing exactly what it does today. `cluster_snapshots` is **purely additive history** — current-state queries still hit `user_clusters`, retrospective queries hit `cluster_snapshots`. No existing code path changes behaviour.

#### Retention policy

A nightly cleanup keeps the last 30 days per user (anything older is unlikely to be useful for debugging and bloats the snapshots table without bound):

```python
async def prune_old_snapshots(retention_days: int = 30):
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(
            "DELETE FROM cluster_snapshots WHERE snapshot_date < datetime('now', ?)",
            (f"-{retention_days} days",),
        )
        await conn.commit()
```

For now, call it on startup (FastAPI lifespan handler). In Phase 7 you'll add a proper APScheduler cron.

**Tests:**

```python
async def test_snapshot_appended_on_each_recluster():
    """Two reclusters of the same user should produce two distinct snapshot_ids
    and 2N rows in cluster_snapshots (where N = number of clusters)."""
    user_id = "test-user"
    clusters_v1 = [_make_cluster(idx=0, papers=["a", "b"])]
    clusters_v2 = [_make_cluster(idx=0, papers=["a", "b", "c"])]
    sid1 = await db.save_cluster_snapshot(user_id, clusters_v1)
    sid2 = await db.save_cluster_snapshot(user_id, clusters_v2)
    assert sid1 != sid2
    rows = await db.fetch_all(
        "SELECT snapshot_id, paper_ids_hash FROM cluster_snapshots WHERE user_id = ? ORDER BY snapshot_date",
        user_id,
    )
    assert len(rows) == 2
    assert rows[0]["paper_ids_hash"] != rows[1]["paper_ids_hash"]  # content-addressed

async def test_prune_respects_retention():
    """Snapshots older than retention_days should be deleted; newer ones kept."""
    # Insert one snapshot dated 45 days ago, one dated 5 days ago
    # Run prune_old_snapshots(retention_days=30)
    # Assert only the recent one remains
```

**Estimated effort:** 6 hours.

---

## 5. Bucket 3 — Phase 5.1: Cold-Start Completion (Day 5, parallel)

This sits **outside Phase 6.5** but ships as part of the same 5-day push. Single identity: **complete the Layer 2 of Phase 5's three-layer onboarding plan that was deferred at the time.** Original Phase 5 plan called for: (Layer 1) category selection, (Layer 2) author-paper import, (Layer 3) seed paper search. Layer 2 was cut for time. This is it.

### 5.1 — B4: Semantic Scholar Author Import

**The user-visible win.** Before B4: a new user lands on `/onboarding`, picks 3 categories, then has to manually search for and save 5 seed papers — friction that bleeds users at the conversion step. After B4: paste your S2 author URL, the system pulls your authored papers, and you have 20 implicit "saves" instantly. First feed is genuinely personalized within seconds of arrival.

This is also the only piece of work in the 5-day push that touches user experience directly. The other four days are all infrastructure. It's worth shipping in the same window so the user-facing improvement masks the otherwise-invisible plumbing changes.

#### S2 API endpoint

```
GET https://api.semanticscholar.org/graph/v1/author/{author_id}/papers
?fields=externalIds,title,year,citationCount
&limit=100
```

`externalIds.ArXiv` gives you the arXiv ID directly — no DOI translation needed. `S2_API_KEY` env var already exists (it's used in Phase 6 reranker training scripts).

#### The flow

**1. New onboarding step** (insert between "categories" and "seed papers" in the existing onboarding wizard):

```
Step 2 of 3: Import your work (optional)

[ Paste your Semantic Scholar profile URL or ORCID ]
                  [ Import ]

[ Skip — I'll search for seed papers manually ]
```

**2. New service file** `app/s2_svc.py`:

```python
"""Semantic Scholar API client for author paper import."""
import re
import httpx
from app import config

S2_BASE = "https://api.semanticscholar.org/graph/v1"


def parse_author_input(text: str) -> str | None:
    """Accept S2 URL, raw S2 ID, or ORCID. Return S2 author ID or None."""
    text = text.strip()
    # S2 URL: https://www.semanticscholar.org/author/Name/12345678
    m = re.search(r"semanticscholar\.org/author/[^/]+/(\d+)", text)
    if m:
        return m.group(1)
    # Raw S2 ID
    if text.isdigit():
        return text
    # ORCID: 0000-0002-1825-0097
    if re.match(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$", text):
        # Resolve ORCID → S2 ID via S2's author search
        return None  # caller should call resolve_orcid()
    return None


async def resolve_orcid(orcid: str) -> str | None:
    """Resolve ORCID → S2 author ID via S2's author search."""
    headers = {"x-api-key": config.S2_API_KEY} if config.S2_API_KEY else {}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{S2_BASE}/author/search",
            params={"query": f"ORCID:{orcid}", "limit": 1, "fields": "authorId"},
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return data[0]["authorId"] if data else None


async def fetch_author_arxiv_papers(author_id: str, limit: int = 50) -> list[str]:
    """Return arxiv_ids of papers authored by this S2 author, most-recent first."""
    headers = {"x-api-key": config.S2_API_KEY} if config.S2_API_KEY else {}
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{S2_BASE}/author/{author_id}/papers",
            params={"fields": "externalIds,year", "limit": limit},
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
    arxiv_ids = []
    # Sort by year descending so we keep most-recent papers if we hit limit
    papers = sorted(
        data.get("data", []),
        key=lambda p: p.get("year") or 0,
        reverse=True,
    )
    for paper in papers:
        ext = paper.get("externalIds") or {}
        if arxiv_id := ext.get("ArXiv"):
            arxiv_ids.append(str(arxiv_id))  # CLAUDE.md rule: arxiv_ids always strings
    return arxiv_ids
```

**3. New router endpoint** in `app/routers/onboarding.py`:

```python
@router.post("/api/onboarding/import-author", response_class=HTMLResponse)
async def import_author(
    request: Request,
    author_input: str = Form(...),
    user_id: str | None = Cookie(default=None, alias=COOKIE_NAME),
):
    user_id = user_id or str(uuid.uuid4())

    # Parse: accept S2 URL, S2 ID, or ORCID
    s2_author_id = s2_svc.parse_author_input(author_input)
    if not s2_author_id:
        # Try ORCID resolution
        if re.match(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$", author_input.strip()):
            s2_author_id = await s2_svc.resolve_orcid(author_input.strip())
    if not s2_author_id:
        return templates.TemplateResponse(
            request, "partials/import_error.html",
            {"error": "Could not parse input. Try a Semantic Scholar URL or ORCID."},
            status_code=400,
        )

    # Fetch from S2 with timeout + graceful fallback
    try:
        arxiv_ids = await s2_svc.fetch_author_arxiv_papers(s2_author_id, limit=50)
    except httpx.HTTPError as e:
        log.warning("s2 author fetch failed: %s", e)
        return templates.TemplateResponse(
            request, "partials/import_error.html",
            {"error": "Semantic Scholar is temporarily unavailable. Try seed search instead."},
            status_code=503,
        )

    if not arxiv_ids:
        return templates.TemplateResponse(
            request, "partials/import_error.html",
            {"error": "No arXiv papers found for this author. Try seed search instead."},
        )

    # Save each as a seed (triggers EWMA, clustering on next request)
    saved_count = 0
    for aid in arxiv_ids:
        await db.log_interaction(
            user_id=user_id,
            paper_id=aid,
            event_type="save",
            source="onboarding_author_import",
        )
        us.record_positive(user_id, aid)
        # Background: fetch vector + update EWMA (don't block the response)
        asyncio.create_task(_update_profile_on_save(user_id, aid))
        saved_count += 1

    response = templates.TemplateResponse(
        request, "partials/import_success.html",
        {"saved_count": saved_count, "next_step": "seed_search"},
    )
    response.set_cookie(COOKIE_NAME, user_id, max_age=COOKIE_MAX_AGE)
    return response
```

**4. Tag the imports specially** — `source="onboarding_author_import"` distinguishes these from normal saves and from `source="onboarding_seed_search"`. Phase 7 evaluation can then ask: *"Do users who used author-import have higher week-1 retention than users who used only seed search?"*

#### Edge cases

| Case                                                         | Solution                                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| Author has 200 papers                                        | Cap at 50 most-recent (50 is plenty for clustering; year-sorted before cap)           |
| Author has 0 arXiv papers (e.g. pure CS-conference profile)  | Show "No arXiv papers found — try seed search instead"                                |
| User pastes ORCID instead of S2 URL                          | Resolved via S2's author search by ORCID                                              |
| User pastes a paper URL by mistake                           | `parse_author_input` returns None → friendly error                                    |
| S2 API rate limit hit                                        | Graceful 503 → fall back to manual seed search                                        |
| User imports, then dislikes everything                       | Negative EWMA self-corrects within 5–10 dismissals                                    |
| User has S2 ID but multiple disambiguated profiles           | Out of scope — they pick the right one when copying their URL                         |

**Test:**

```python
async def test_s2_import_saves_papers_with_correct_source_tag():
    # Mock fetch_author_arxiv_papers to return ["2401.001", "2401.002"]
    # POST /api/onboarding/import-author with a fake S2 URL
    rows = await db.fetch_all(
        "SELECT paper_id, source FROM interactions WHERE user_id = ?", user_id,
    )
    assert {r["paper_id"] for r in rows} == {"2401.001", "2401.002"}
    assert all(r["source"] == "onboarding_author_import" for r in rows)
```

**Estimated effort:** 5 hours.

---

## 6. What Phase 7 inherits

After these 5 days, Phase 7 starts on a substrate where every prerequisite is already in production:

| Capability                              | Before this push                          | After this push                                  |
| --------------------------------------- | ----------------------------------------- | ------------------------------------------------ |
| Feature 0 in LightGBM                   | ❌ rank-proxy lie                          | ✅ actual cosine                                  |
| Production model verified live           | ❓ unverified                              | ✅ green checkmark with timestamp                 |
| Per-feed CTR measurable                 | ❌ no `query_id`                           | ✅ one SQL query away                             |
| Counterfactual eval (SNIPS) possible    | ❌ no propensity                           | ✅ schema ready, propensities flowing             |
| Cluster history queryable               | ❌ destroyed on each recluster             | ✅ 30 days kept, content-addressed                |
| Cold-start onboarding                   | ❌ manual 5-paper search only              | ✅ paste S2 URL → 20 implicit saves               |

Phase 7's evaluation framework now has a real substrate. Without these, Phase 7 would have to spend its first week building this infrastructure anyway — better to do it deliberately as a pre-Phase-7 push than under deadline pressure.

---

## 7. Acceptance criteria

### Bucket 1 — Phase 6 Hot-fix done when:

- [ ] `qdrant_score_map` is populated from `search_by_vector_with_scores()` in both the per-cluster path and the short-term supplement path
- [ ] Synthetic-decay block (current `recommendations.py:313–325`) is deleted
- [ ] `test_qdrant_scores_are_real_cosines_not_rank_proxies` passes
- [ ] `curl https://siddhm11-researchit.hf.space/healthz/reranker` returns `model_loaded: true, n_trees: 141, fallback_active: false`
- [ ] PHASE6-Reranker-Framing.md updated with verification timestamp

### Bucket 2 — Phase 6.5 done when:

- [ ] `query_id` is generated per request in `recommendations.py` and `search.py` and round-trips through templates → events → DB
- [ ] `interactions` table has `propensity REAL` and `policy_id TEXT` columns
- [ ] Every interaction logged from a recommendation/search request has non-null `propensity` and `policy_id`
- [ ] `cluster_snapshots` table exists with the schema in §4.3
- [ ] Every recluster appends a new snapshot (verified by `test_snapshot_appended_on_each_recluster`)
- [ ] `prune_old_snapshots(retention_days=30)` is registered in the FastAPI lifespan handler
- [ ] All new tests pass; total test count in `README.md` updated

### Bucket 3 — Phase 5.1 done when:

- [ ] `app/s2_svc.py` exists and `fetch_author_arxiv_papers` returns arxiv IDs (verified against a real S2 author profile)
- [ ] `/api/onboarding/import-author` accepts S2 URL, S2 ID, and ORCID input forms
- [ ] Imported papers are saved with `source="onboarding_author_import"`
- [ ] Background EWMA update fires for each imported paper
- [ ] All 6 edge cases in §5.1 are handled with graceful UX

---

## 8. Sequencing & timeline

### Recommended order

```
Day 1  (~3h)  Bucket 1: A1 (real Qdrant scores) + A2 (curl /healthz)
Day 2  (~3h)  Bucket 2.B1: query_id linkage
Day 3  (~4h)  Bucket 2.B2: propensity logging
Day 4  (~6h)  Bucket 2.B3: cluster snapshot versioning
Day 5  (~5h)  Bucket 3.B4: S2 author import
```

Each day leaves the app in a working state. No big-bang refactors. No day depends on a later day's work.

### Parallelization options

If you have stretches where you want to context-switch:

- **Day 5 (B4) can run anytime** — it's onboarding code, doesn't touch the recommendation pipeline or schema. Could ship before Day 1 if you want a user-visible win first.
- **Day 1 should land before Day 2–4** — once `query_id` and `propensity` start flowing, you want feature 0 to already be real cosines so your first logged interactions are clean training data for any future retrain.
- **Days 2–4 should ship as a block** — the three pieces compound. Shipping B1 without B2 means logs have feed identity but no eval lever; shipping B2 without B1 means propensities can't be grouped by feed; shipping B3 without either means snapshots exist but you can't correlate them to actions.

### What this defers (intentionally)

| Item                                                  | Why deferred                                                                                  |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Track C: Full ORCID/Scholar import with disambiguation | B4 captures ~80% of the value. Full version waits until there's user-data evidence it's needed. |
| Track D: Cluster summary cards (Phase 8a preview)     | Needs Phase 7 evaluation infrastructure to measure whether it actually helps users.            |
| Phase 6.4 reranker retraining                         | Already gated on synthetic simulator OR 100 real users with ≥10 saves each. Unchanged.        |

---

## 9. Documentation updates needed

After this push lands:

- [ ] Add line to `CLAUDE.md` non-negotiable rules: *"Rule 9: Every interaction logged from a recommendation/search request must carry `query_id`, `propensity`, and `policy_id`. These are load-bearing for Phase 7 evaluation."*
- [ ] Update `PHASE6-Reranker-Framing.md` Section E with the live verification timestamp
- [ ] Update `TASK-TRACKER.md`:
  - Tick `[x] [reranker] LightGBM model loaded (verified live YYYY-MM-DD)`
  - Tick `[x] [reranker] qdrant_cosine_score uses real cosines`
  - Add new section `Phase 6.5 — Instrumentation Foundation` with checklist from §7
- [ ] Update `README.md` test count
- [ ] Update `docs/walkthroughs/04-Next-Steps-and-Phase-Plan.md`: insert Phase 6.5 between Phase 6 and Phase 7 in the master roadmap; note Phase 5.1 as a parallel side-quest
- [ ] Mark ADR A1 (cluster snapshot versioning) and ADR A4 (telemetry schema) as **Decided + Implemented** in the Phase 6 framing doc's ADR table

---

## 10. Out of scope (explicit)

To keep this doc focused, the following are **not** part of this push:

- Building the actual evaluation harness (offline regression, time-split eval, frozen `eval/eval_set_v1.0.parquet`, CI gates) — that's Phase 7 itself.
- LLM cluster summaries (Phase 8a) — depends on `paper_ids_hash` from B3, but the LLM call path itself is Phase 8.
- Reranker retraining (Phase 6.4) — gated on user-volume thresholds, unchanged.
- Google Scholar import — no public API, would need scraping. Defer until S2 import shows real adoption.
- Per-paper relevance dial in author import (not all of someone's authored papers represent current interest) — out of scope; let the EWMA negative path handle it organically.

---

*End of framing doc.*
