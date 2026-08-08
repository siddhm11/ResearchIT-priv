# Integration Guide — LightGBM Reranker into ResearchIT

> **For:** Whoever integrates the reranker into `app/recommend/reranker.py`  
> **Covers:** Steps 5-8 from the Phase 6 roadmap  
> **Prerequisites:** The production model is trained and in `production_model/reranker_v1.txt`

---

## Overview

You need to do 4 things:
1. **Expand `compute_features()` from 5 → 37 features** (biggest change)
2. **Wire model loading + heuristic fallback** at startup
3. **Add `lightgbm` to `requirements.txt`** and model file to Docker image
4. **Integration testing**

---

## Step 5: Expand `compute_features()` to 37 Features

The current heuristic uses 5 features. The LightGBM model expects 37 features in a **specific order** defined in `production_model/feature_schema.json`.

### Feature Schema (order matters!)

```python
FEATURE_SCHEMA = [
    # Content/Retrieval (0-19)
    "qdrant_cosine_score",           # 0  - from Qdrant ANN search
    "candidate_position",            # 1  - rank in ANN results
    "candidate_citation_count",      # 2  - from Turso papers table
    "candidate_log_citations",       # 3  - log(citation_count + 1)
    "candidate_influential_citations",  # 4  - from Turso papers table
    "candidate_age_days",            # 5  - (now - update_date).days
    "candidate_recency_score",       # 6  - exp(-0.002 * age_days)
    "query_citation_count",          # 7  - user's profile paper citations (or 0)
    "query_age_days",                # 8  - user's profile paper age (or 0)
    "year_diff",                     # 9  - |query_year - candidate_year|
    "same_primary_category",         # 10 - 1 if same primary_topic
    "co_citation_count",             # 11 - shared citers (expensive; can be 0)
    "shared_author_count",           # 12 - shared authors between query & candidate
    "candidate_is_newer",            # 13 - 1 if candidate.year > query.year
    "query_log_citations",           # 14 - log(query_citation_count + 1)
    "citation_count_ratio",          # 15 - cand_citations / (query_citations + 1)
    "age_ratio",                     # 16 - cand_age / (query_age + 1)
    "candidate_citations_per_year",  # 17 - citations / max(age_years, 0.5)
    "query_num_references",          # 18 - 0 for now (needs citation graph in prod)
    "candidate_num_cited_by",        # 19 - 0 for now (needs citation graph in prod)

    # User Behavior (20-30) — from EWMA profiles, clusters, interactions
    "ewma_longterm_similarity",      # 20 - cos(candidate_emb, user.lt_profile)
    "ewma_shortterm_similarity",     # 21 - cos(candidate_emb, user.st_profile)
    "ewma_negative_similarity",      # 22 - cos(candidate_emb, user.neg_profile)
    "cluster_importance",            # 23 - cluster weight from Ward clustering
    "cluster_distance_to_medoid",    # 24 - cos(candidate_emb, cluster_medoid)
    "is_suppressed_category",        # 25 - 1 if suppressed category
    "onboarding_category_match",     # 26 - 1 if matches onboarding prefs
    "user_total_saves",              # 27 - total saves from interactions table
    "user_total_dismissals",         # 28 - total dismissals
    "user_days_since_last_save",     # 29 - days since last save
    "user_session_save_count",       # 30 - saves this session

    # Cross Features (31-36) — computed from above
    "cosine_x_recency",             # 31 - feat[0] * feat[6]
    "cosine_x_citations",           # 32 - feat[0] * feat[3]
    "category_x_recency",           # 33 - feat[10] * feat[6]
    "cosine_x_cocitation",          # 34 - feat[0] * log(feat[11] + 1)
    "position_inverse",             # 35 - 1 / (feat[1] + 1)
    "citations_x_recency",          # 36 - feat[3] * feat[6]
]
```

### Implementation Sketch

```python
import numpy as np
from datetime import datetime, timezone

def compute_features_v2(
    user_state: dict,        # EWMA profiles, cluster info, interaction counts
    candidate: dict,         # paper metadata from Turso
    qdrant_score: float,     # cosine score from ANN search
    candidate_position: int, # rank position (0-indexed)
    candidate_embedding: np.ndarray,  # 1024-dim BGE-M3 embedding
) -> np.ndarray:
    """
    Compute 37-feature vector for LightGBM reranker.
    
    Args:
        user_state: {
            "lt_profile": np.ndarray,       # long-term EWMA (1024-dim or None)
            "st_profile": np.ndarray,       # short-term EWMA (1024-dim or None)
            "neg_profile": np.ndarray,      # negative EWMA (1024-dim or None)
            "cluster_importance": float,     # from Ward clustering
            "cluster_medoid": np.ndarray,   # cluster medoid embedding (or None)
            "suppressed_categories": set,    # suppressed arXiv categories
            "onboarding_categories": set,    # onboarding selections
            "total_saves": int,
            "total_dismissals": int,
            "days_since_last_save": float,
            "session_save_count": int,
            "query_paper": dict | None,     # the "seed" paper if applicable
        }
        candidate: {
            "arxiv_id": str,
            "primary_topic": str,
            "update_date": str,              # "YYYY-MM-DD"
            "citation_count": int,
            "influential_citations": int,
            "authors": list[str],
        }
        qdrant_score: cosine similarity from ANN search
        candidate_position: rank in ANN results (0-indexed)
        candidate_embedding: paper's BGE-M3 embedding vector
        
    Returns:
        np.ndarray of shape (37,) — feature vector in schema order
    """
    features = np.zeros(37, dtype=np.float32)
    now = datetime.now(timezone.utc)
    
    # --- Content/Retrieval features (0-19) ---
    
    # 0: qdrant_cosine_score
    features[0] = qdrant_score
    
    # 1: candidate_position
    features[1] = float(candidate_position)
    
    # 2: candidate_citation_count
    cand_citations = candidate.get("citation_count", 0) or 0
    features[2] = float(cand_citations)
    
    # 3: candidate_log_citations
    features[3] = np.log(cand_citations + 1)
    
    # 4: candidate_influential_citations
    features[4] = float(candidate.get("influential_citations", 0) or 0)
    
    # 5: candidate_age_days
    try:
        pub_date = datetime.strptime(candidate.get("update_date", "")[:10], "%Y-%m-%d")
        pub_date = pub_date.replace(tzinfo=timezone.utc)
        cand_age = max(0, (now - pub_date).days)
    except (ValueError, TypeError):
        cand_age = 365  # default 1 year
    features[5] = float(cand_age)
    
    # 6: candidate_recency_score
    features[6] = np.exp(-0.002 * cand_age)
    
    # 7-9: Query paper features (from user's seed paper, or defaults)
    query_paper = user_state.get("query_paper") or {}
    query_citations = query_paper.get("citation_count", 0) or 0
    features[7] = float(query_citations)
    
    try:
        q_pub = datetime.strptime(query_paper.get("update_date", "")[:10], "%Y-%m-%d")
        q_pub = q_pub.replace(tzinfo=timezone.utc)
        query_age = max(0, (now - q_pub).days)
    except (ValueError, TypeError):
        query_age = 0
    features[8] = float(query_age)
    
    cand_year = _parse_year(candidate.get("update_date", ""))
    query_year = _parse_year(query_paper.get("update_date", "")) if query_paper else cand_year
    features[9] = abs(query_year - cand_year)
    
    # 10: same_primary_category
    q_cat = query_paper.get("primary_topic", "") if query_paper else ""
    c_cat = candidate.get("primary_topic", "")
    features[10] = 1.0 if (q_cat and c_cat and q_cat == c_cat) else 0.0
    
    # 11: co_citation_count (0 unless you have citation graph loaded)
    features[11] = 0.0  # TODO: populate if citation graph is loaded
    
    # 12: shared_author_count
    if query_paper and query_paper.get("authors"):
        q_authors = {a.lower().strip() for a in query_paper["authors"] if a}
        c_authors = {a.lower().strip() for a in (candidate.get("authors") or []) if a}
        features[12] = float(len(q_authors & c_authors))
    
    # 13: candidate_is_newer
    features[13] = 1.0 if cand_year > query_year else 0.0
    
    # 14: query_log_citations
    features[14] = np.log(query_citations + 1)
    
    # 15: citation_count_ratio
    features[15] = cand_citations / (query_citations + 1)
    
    # 16: age_ratio
    features[16] = cand_age / (query_age + 1) if query_age > 0 else 0.0
    
    # 17: candidate_citations_per_year
    cand_age_years = max(cand_age / 365.0, 0.5)
    features[17] = cand_citations / cand_age_years
    
    # 18-19: Graph features (0 unless citation graph loaded in prod)
    features[18] = 0.0  # query_num_references
    features[19] = 0.0  # candidate_num_cited_by
    
    # --- User Behavior features (20-30) ---
    
    # 20: ewma_longterm_similarity
    lt_prof = user_state.get("lt_profile")
    if lt_prof is not None and candidate_embedding is not None:
        features[20] = _cosine_sim(candidate_embedding, lt_prof)
    
    # 21: ewma_shortterm_similarity
    st_prof = user_state.get("st_profile")
    if st_prof is not None and candidate_embedding is not None:
        features[21] = _cosine_sim(candidate_embedding, st_prof)
    
    # 22: ewma_negative_similarity
    neg_prof = user_state.get("neg_profile")
    if neg_prof is not None and candidate_embedding is not None:
        features[22] = _cosine_sim(candidate_embedding, neg_prof)
    
    # 23: cluster_importance
    features[23] = float(user_state.get("cluster_importance", 0.0))
    
    # 24: cluster_distance_to_medoid
    medoid = user_state.get("cluster_medoid")
    if medoid is not None and candidate_embedding is not None:
        features[24] = _cosine_sim(candidate_embedding, medoid)
    
    # 25: is_suppressed_category
    suppressed = user_state.get("suppressed_categories", set())
    features[25] = 1.0 if c_cat in suppressed else 0.0
    
    # 26: onboarding_category_match
    onboarding = user_state.get("onboarding_categories", set())
    features[26] = 1.0 if c_cat in onboarding else 0.0
    
    # 27-30: Interaction counts
    features[27] = float(user_state.get("total_saves", 0))
    features[28] = float(user_state.get("total_dismissals", 0))
    features[29] = float(user_state.get("days_since_last_save", 0.0))
    features[30] = float(user_state.get("session_save_count", 0))
    
    # --- Cross Features (31-36) ---
    
    features[31] = features[0] * features[6]   # cosine × recency
    features[32] = features[0] * features[3]   # cosine × log_citations
    features[33] = features[10] * features[6]  # category × recency
    features[34] = features[0] * np.log(features[11] + 1)  # cosine × log_cocitation
    features[35] = 1.0 / (features[1] + 1)     # position_inverse
    features[36] = features[3] * features[6]   # log_citations × recency
    
    return features


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _parse_year(date_str: str) -> int:
    try:
        return int(date_str[:4])
    except (ValueError, TypeError, IndexError):
        return 2020
```

### Vectorized Version (for batch scoring)

For production use, compute features for ALL candidates at once:

```python
def compute_features_batch(
    user_state: dict,
    candidates: list[dict],
    qdrant_scores: list[float],
    candidate_embeddings: np.ndarray,  # (N, 1024)
) -> np.ndarray:
    """
    Compute features for all candidates at once.
    Returns (N, 37) feature matrix.
    """
    N = len(candidates)
    features = np.zeros((N, 37), dtype=np.float32)
    
    for i, (cand, score) in enumerate(zip(candidates, qdrant_scores)):
        features[i] = compute_features_v2(
            user_state=user_state,
            candidate=cand,
            qdrant_score=score,
            candidate_position=i,
            candidate_embedding=candidate_embeddings[i] if candidate_embeddings is not None else None,
        )
    
    return features
```

> **Performance note:** The bottleneck is NOT feature computation or LightGBM prediction (0.4ms). It's fetching candidate metadata from Turso. Batch your Turso queries.

---

## Step 6: Wire Model Loading + Heuristic Fallback

In `app/recommend/reranker.py`:

```python
import os
import lightgbm as lgb
import numpy as np

# ── Model Loading ────────────────────────────────────────────────────────────

_lgb_model = None
_model_path = os.environ.get("RERANKER_MODEL_PATH", "production_model/reranker_v1.txt")

try:
    _lgb_model = lgb.Booster(model_file=_model_path)
    print(f"[reranker] LightGBM model loaded from {_model_path}")
    print(f"[reranker]   num_features: {_lgb_model.num_feature()}")
    print(f"[reranker]   num_trees: {_lgb_model.num_trees()}")
except FileNotFoundError:
    print(f"[reranker] Model file not found: {_model_path} — using heuristic")
except Exception as e:
    print(f"[reranker] Model load failed: {e} — using heuristic")


# ── Main Reranking Function ──────────────────────────────────────────────────

def rerank_candidates(
    user_state: dict,
    candidates: list[dict],
    qdrant_scores: list[float],
    candidate_embeddings: np.ndarray | None = None,
) -> list[dict]:
    """
    Rerank candidates using LightGBM (or heuristic fallback).
    
    Returns candidates sorted by score (best first).
    """
    if not candidates:
        return []
    
    if _lgb_model is not None:
        # LightGBM path
        features = compute_features_batch(user_state, candidates, qdrant_scores, candidate_embeddings)
        scores = _lgb_model.predict(features)
    else:
        # Heuristic fallback (always works, no model needed)
        scores = np.array([
            heuristic_score(user_state, cand, score)
            for cand, score in zip(candidates, qdrant_scores)
        ])
    
    # Sort by score descending
    order = np.argsort(-scores)
    return [candidates[i] for i in order]
```

### Key Design Decisions

1. **The heuristic fallback is PERMANENT.** Don't remove it. It's your safety net if:
   - The model file is missing (fresh deploy)
   - LightGBM import fails (dependency issue)
   - The model produces garbage (bad retrain)

2. **Model path is configurable** via `RERANKER_MODEL_PATH` env var. This lets you A/B test different models without code changes.

3. **No model versioning yet.** For v1, just replace the file. When you have v2, add version tracking.

---

## Step 7: Update `requirements.txt`

Add to your `requirements.txt`:
```
lightgbm>=4.0,<5.0
```

And in your `Dockerfile`, ensure the model file is copied:
```dockerfile
COPY production_model/reranker_v1.txt /app/production_model/reranker_v1.txt
```

Or download from this repo at startup:
```python
# In app startup
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="siddhm11/researchit-reranker-phase6",
    filename="production_model/reranker_v1.txt",
)
```

---

## Step 8: Integration Testing

### Smoke Test
```python
import lightgbm as lgb
import numpy as np

# Load model
model = lgb.Booster(model_file="production_model/reranker_v1.txt")
assert model.num_feature() == 37

# Predict on dummy input
dummy = np.zeros((5, 37), dtype=np.float32)
scores = model.predict(dummy)
assert scores.shape == (5,)
assert not np.any(np.isnan(scores))
print("✅ Smoke test passed")
```

### End-to-End Test
```python
# Verify the full pipeline: ANN → feature computation → LightGBM → ranked output
def test_e2e():
    # 1. Simulate a user with EWMA profiles
    user_state = {
        "lt_profile": np.random.randn(1024).astype(np.float32),
        "st_profile": np.random.randn(1024).astype(np.float32),
        "neg_profile": np.random.randn(1024).astype(np.float32),
        "cluster_importance": 0.8,
        "cluster_medoid": np.random.randn(1024).astype(np.float32),
        "suppressed_categories": {"cs.CR"},
        "onboarding_categories": {"cs.CL", "cs.LG"},
        "total_saves": 42,
        "total_dismissals": 10,
        "days_since_last_save": 0.5,
        "session_save_count": 3,
        "query_paper": None,
    }
    
    # 2. Simulate candidates from Qdrant
    candidates = [
        {"arxiv_id": f"2024.{i:05d}", "primary_topic": "cs.CL",
         "update_date": "2024-01-15", "citation_count": i*10,
         "influential_citations": i, "authors": ["Alice", "Bob"]}
        for i in range(50)
    ]
    qdrant_scores = [0.9 - i*0.01 for i in range(50)]
    candidate_embeddings = np.random.randn(50, 1024).astype(np.float32)
    
    # 3. Rerank
    ranked = rerank_candidates(user_state, candidates, qdrant_scores, candidate_embeddings)
    
    assert len(ranked) == 50
    # The order should differ from the ANN order (LightGBM reranks)
    original_ids = [c["arxiv_id"] for c in candidates]
    reranked_ids = [c["arxiv_id"] for c in ranked]
    assert original_ids != reranked_ids, "LightGBM should change the order"
    print("✅ E2E test passed")
```

### Latency Test
```python
import time

features = np.random.randn(100, 37).astype(np.float32)

# Warmup
for _ in range(100):
    model.predict(features)

# Benchmark
t0 = time.time()
for _ in range(1000):
    model.predict(features)
elapsed = (time.time() - t0) / 1000 * 1000  # ms per call

assert elapsed < 1.0, f"Too slow: {elapsed:.3f}ms (target: <1ms)"
print(f"✅ Latency: {elapsed:.3f}ms per 100 candidates")
```

---

## Notes for Future Retraining

When you have 500+ real user interactions:

1. Export interactions from Turso:
   ```sql
   SELECT user_id, arxiv_id, action, created_at FROM interactions
   ```

2. Generate new training triples with **real labels**:
   - `action = 'save'` → label 2
   - `action = 'click'` → label 1
   - `action = 'dismiss'` → label 0

3. The 37-feature schema is **stable** — features 20-30 will now be populated with real EWMA profiles, cluster data, and interaction counts.

4. Retrain with the same `03_train_lightgbm.py` script on the new data.

5. The user behavior features (20-30) should gain significant importance in the new model.
