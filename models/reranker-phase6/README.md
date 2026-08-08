# ResearchIT Phase 6 — LightGBM Reranker

> **Status:** ✅ **Production model trained and evaluated on real citation data.**  
> **Parent project:** [siddhm11/ResearchIT](https://huggingface.co/spaces/siddhm11/ResearchIT)  
> **Replaces:** Hand-tuned heuristic scorer in `app/recommend/reranker.py`  
> **Architecture position:** LightGBM-1 in the Doc 07 multi-stage pipeline

---

## 🎯 TL;DR

A LightGBM lambdarank model that reranks arXiv paper recommendations. Trained on **242K real citation edges** from Semantic Scholar across **1.6M arXiv papers** in the ResearchIT corpus.

| Metric | Heuristic | LightGBM | Improvement |
|--------|:---------:|:--------:|:-----------:|
| **nDCG@5** | 0.1819 | **0.8250** | **+353.6%** |
| **nDCG@10** | 0.2641 | **0.8791** | **+232.8%** |
| **nDCG@20** | 0.3296 | **0.8857** | **+168.7%** |
| Recall@10 | 0.4384 | **0.9825** | +124.1% |
| HR@10 | 0.6638 | **1.0000** | +50.6% |
| MRR | 0.2906 | **0.8795** | +202.7% |

**Latency:** 0.371ms per 100 candidates (budget: <1ms) ✅  
**Model size:** 948 KB  
**Verdict:** ✅ DEPLOY — massive improvement across all metrics.

---

## 📦 Repository Contents

```
researchit-reranker-phase6/
│
├── production_model/                  ← PRODUCTION ARTIFACTS
│   ├── reranker_v1.txt               ← THE MODEL (948 KB, LightGBM text format)
│   ├── eval_metrics.json             ← Full benchmark results + training metadata
│   ├── baseline_comparison.json      ← LightGBM vs heuristic comparison
│   ├── feature_importance.csv        ← All 37 features ranked by gain
│   └── feature_schema.json           ← 37-feature schema definition (ordered)
│
├── scripts/                           ← REPRODUCIBLE PIPELINE (3 scripts)
│   ├── 01_fetch_citation_edges.py    ← S2 API → citations.parquet
│   ├── 02_generate_training_triples.py ← Qdrant ANN + Turso → train/eval.parquet
│   └── 03_train_lightgbm.py          ← LightGBM lambdarank training + eval
│
├── synthetic_model/                   ← PROOF OF CONCEPT (synthetic data)
│   ├── reranker_v1_synthetic.txt     ← Model trained on synthetic data (286 KB)
│   └── test_results.json             ← Synthetic eval results
│
├── tests/
│   └── test_full_pipeline.py         ← Comprehensive test suite (6 categories)
│
├── INTEGRATION_GUIDE.md              ← Step-by-step integration into ResearchIT
├── CHANGELOG.md                       ← Version history
└── README.md                          ← This file
```

---

## 🧠 How It Works

### The Problem

ResearchIT recommends arXiv papers using a multi-stage pipeline:
```
Qdrant ANN retrieval → Quota fusion → Reranking → MMR diversity → Feed
```

The **reranking** step uses a hand-tuned heuristic scorer with 5 features and fixed weights:
```python
score = 0.40 × cos(paper, long_term_profile)
      + 0.25 × cos(paper, short_term_profile)
      + 0.15 × recency_decay
      + 0.10 × retrieval_rank_confidence
      - 0.15 × cos(paper, negative_profile)
```

This heuristic can't use citation count, co-citation networks, category match, or any feature interactions. The weights are guesses.

### The Solution

Train a **LightGBM lambdarank model** on **citation-graph pseudo-labels**:

1. **Each arXiv paper acts as a "pseudo-user"** — its bibliography simulates what that researcher would "save"
2. **Direct citations → label 2** (strong positive — this paper was important enough to cite)
3. **Co-citations → label 1** (weak positive — papers sharing community context)
4. **ANN-retrieved but not cited → label 0** (negative — topically related but not worth citing)

The model learns: given 37 features about a (user, paper) pair, which papers should rank higher?

### Where It Fits in Doc 07

This is **LightGBM-1** in the multi-stage architecture:
```
Qdrant ANN (Phase 2) → LightGBM-1 (THIS MODEL) → [TinyBERT → LightGBM-2] (Phase 8b, future)
```

---

## 📊 Production Results

### Training Data (Real Citation Graph)

| Metric | Value |
|--------|-------|
| Corpus size | 1,597,097 arXiv papers |
| Papers sampled for S2 API | 50,000 |
| In-corpus citation edges | 242,179 |
| Training rows | 90,993 (1,857 queries, pre-2023) |
| Eval rows | 7,007 (143 queries, 2023+) |
| Label distribution | 4.6% direct citation, 0.2% co-citation, 95.1% negative |
| Time split | Train: pre-2023, Eval: 2023+ (verified: no temporal leakage) |

### Model Training

| Parameter | Value |
|-----------|-------|
| Objective | lambdarank |
| Num boost rounds | 500 (early stopped at 141) |
| Learning rate | 0.05 |
| Num leaves | 63 |
| Min data in leaf | 50 |
| Feature fraction | 0.8 |
| Bagging fraction | 0.8 |
| Training time | ~7 minutes |

### Evaluation: LightGBM vs Heuristic Baseline

The heuristic baseline uses `qdrant_cosine_score` as proxy for `ewma_longterm_similarity` (since EWMA profiles don't exist for pseudo-users). This is a **fair comparison** — both models see the same zero-filled user features.

| Metric | Heuristic | LightGBM | Delta | % Improvement |
|--------|:---------:|:--------:|:-----:|:-------------:|
| nDCG@5 | 0.1819 | **0.8250** | +0.6432 | **+353.6%** |
| nDCG@10 | 0.2641 | **0.8791** | +0.6150 | **+232.8%** |
| nDCG@20 | 0.3296 | **0.8857** | +0.5561 | **+168.7%** |
| Recall@10 | 0.4384 | **0.9825** | +0.5442 | **+124.1%** |
| Recall@50 | 1.0000 | 1.0000 | 0.0000 | 0.0% |
| HR@10 | 0.6638 | **1.0000** | +0.3362 | **+50.6%** |
| MRR | 0.2906 | **0.8795** | +0.5889 | **+202.7%** |

### Production Readiness

| Check | Result | Target | Status |
|-------|--------|--------|--------|
| Latency (100 candidates) | 0.371ms | <1ms | ✅ 2.7× under budget |
| Model size | 948 KB | <2 MB | ✅ |
| Model reload | Identical predictions | — | ✅ |
| Handles NaN input | Graceful | — | ✅ |
| Handles extreme values | No crash | — | ✅ |
| Best iteration | 141/500 | — | ✅ Early stopping healthy |

---

## 🏆 Feature Importance (Top 15)

| Rank | Feature | Importance | Description |
|------|---------|:----------:|-------------|
| 1 | `candidate_num_cited_by` | 75,203 | How many corpus papers cite this candidate |
| 2 | `age_ratio` | 7,597 | candidate_age / (query_age + 1) |
| 3 | `candidate_position` | 6,765 | Rank position in ANN results |
| 4 | `cosine_x_citations` | 2,383 | cosine × log(citations) interaction |
| 5 | `qdrant_cosine_score` | 2,353 | BGE-M3 cosine similarity |
| 6 | `candidate_citation_count` | 2,042 | Raw citation count |
| 7 | `citation_count_ratio` | 2,001 | candidate/query citation ratio |
| 8 | `query_age_days` | 1,749 | Age of the query paper |
| 9 | `query_num_references` | 1,726 | How many papers the query cites |
| 10 | `candidate_citations_per_year` | 1,633 | Citation velocity |
| 11 | `candidate_influential_citations` | 1,564 | S2 influential citation count |
| 12 | `query_citation_count` | 1,290 | Query paper's citation count |
| 13 | `category_x_recency` | 1,188 | category_match × recency interaction |
| 14 | `citations_x_recency` | 1,143 | log_citations × recency interaction |
| 15 | `position_inverse` | 1,108 | 1 / (position + 1) |

**Key insight:** `candidate_num_cited_by` (how many corpus papers cite this candidate) is the dominant signal — 10× more important than any other feature. This is a "corpus-wide popularity" signal that the heuristic cannot access.

**User behavior features (20-30):** All 11 have zero importance (correctly — they're zero-filled for pseudo-labels). When real user data arrives (500+ interactions), retrain and these features will activate.

---

## 🔬 The 37-Feature Schema

### Content/Retrieval Features (0-19) — Active in pseudo-label training

| # | Feature | Description | Source |
|---|---------|-------------|--------|
| 0 | `qdrant_cosine_score` | BGE-M3 cosine similarity from ANN search | Qdrant |
| 1 | `candidate_position` | Rank position in ANN results (0-indexed) | Qdrant |
| 2 | `candidate_citation_count` | Total citation count | Turso |
| 3 | `candidate_log_citations` | log(citation_count + 1) | Computed |
| 4 | `candidate_influential_citations` | Influential citation count (S2) | Turso |
| 5 | `candidate_age_days` | Days since publication | Turso |
| 6 | `candidate_recency_score` | exp(-0.002 × age_days) — matches heuristic | Computed |
| 7 | `query_citation_count` | Citation count of the query/user paper | Turso |
| 8 | `query_age_days` | Days since query paper published | Turso |
| 9 | `year_diff` | |query_year - candidate_year| | Computed |
| 10 | `same_primary_category` | 1 if same primary arXiv category | Turso |
| 11 | `co_citation_count` | Papers citing BOTH query and candidate | Citation graph |
| 12 | `shared_author_count` | Shared authors (case-insensitive) | Turso |
| 13 | `candidate_is_newer` | 1 if candidate published after query | Computed |
| 14 | `query_log_citations` | log(query_citation_count + 1) | Computed |
| 15 | `citation_count_ratio` | candidate_citations / (query_citations + 1) | Computed |
| 16 | `age_ratio` | candidate_age / (query_age + 1) | Computed |
| 17 | `candidate_citations_per_year` | citation_count / max(age_years, 0.5) | Computed |
| 18 | `query_num_references` | Papers the query cites (in-corpus) | Citation graph |
| 19 | `candidate_num_cited_by` | Corpus papers that cite the candidate | Citation graph |

### User Behavior Features (20-30) — Zero-filled for pseudo-labels, active for real users

| # | Feature | Description | Source in ResearchIT |
|---|---------|-------------|---------------------|
| 20 | `ewma_longterm_similarity` | cos(candidate, long-term EWMA profile) | `profiles.py` α=0.03 |
| 21 | `ewma_shortterm_similarity` | cos(candidate, short-term EWMA profile) | `profiles.py` α=0.40 |
| 22 | `ewma_negative_similarity` | cos(candidate, negative EWMA profile) | `profiles.py` α=0.15 |
| 23 | `cluster_importance` | Importance weight of serving cluster | `clustering.py` |
| 24 | `cluster_distance_to_medoid` | cos(candidate, cluster medoid) | `clustering.py` |
| 25 | `is_suppressed_category` | 1 if category suppressed (≥3 dismissals in 14d) | `db.py` |
| 26 | `onboarding_category_match` | 1 if matches onboarding selections | `db.py` |
| 27 | `user_total_saves` | Total papers saved | `interactions` table |
| 28 | `user_total_dismissals` | Total papers dismissed | `interactions` table |
| 29 | `user_days_since_last_save` | Days since last save | `interactions` table |
| 30 | `user_session_save_count` | Saves in current session | In-memory state |

### Cross Features (31-36) — Interaction terms

| # | Feature | Formula |
|---|---------|---------|
| 31 | `cosine_x_recency` | qdrant_cosine_score × candidate_recency_score |
| 32 | `cosine_x_citations` | qdrant_cosine_score × candidate_log_citations |
| 33 | `category_x_recency` | same_primary_category × candidate_recency_score |
| 34 | `cosine_x_cocitation` | qdrant_cosine_score × log(co_citation_count + 1) |
| 35 | `position_inverse` | 1 / (candidate_position + 1) |
| 36 | `citations_x_recency` | candidate_log_citations × candidate_recency_score |

---

## 🔄 Reproducing the Pipeline

### Prerequisites
```bash
pip install httpx pyarrow tqdm numpy qdrant-client lightgbm
```

### Step 1: Export Corpus IDs
Export arXiv IDs from Turso:
```sql
SELECT arxiv_id FROM papers;
```
Save as `arxiv_ids.txt` (one ID per line). Our corpus: 1,597,097 IDs.

### Step 2: Fetch Citation Edges (~30 min for 50K papers)
```bash
python scripts/01_fetch_citation_edges.py \
  --corpus-file arxiv_ids.txt \
  --output citations.parquet \
  --max-papers 50000  # sample for rate limits; remove for full corpus
```

- Supports checkpoint/resume (safe to interrupt)
- S2 API key optional but recommended (faster rate limit)
- Filters to in-corpus edges (both papers must be in Qdrant)
- Our run: 50K papers → 242,179 in-corpus edges

### Step 3: Generate Training Triples (~80 min)
```bash
python scripts/02_generate_training_triples.py \
  --citations citations.parquet \
  --corpus-file arxiv_ids.txt \
  --qdrant-url "$QDRANT_URL" \
  --qdrant-api-key "$QDRANT_API_KEY" \
  --qdrant-collection arxiv_bgem3_dense \
  --turso-url "$TURSO_URL" \
  --turso-token "$TURSO_DB_TOKEN" \
  --output-dir ./ltr_dataset \
  --num-queries 2000 \
  --candidates-per-query 50
```

- Enforces time-split: train on pre-2023, eval on 2023+
- Asserts no temporal leakage: `max(train.year) < min(eval.year)`
- Uses `scroll()` + `query_points()` for qdrant-client 1.17+
- Our run: 90,993 train rows + 7,007 eval rows

### Step 4: Train Model (~5 min)
```bash
python scripts/03_train_lightgbm.py \
  --train-file ltr_dataset/train.parquet \
  --eval-file ltr_dataset/eval.parquet \
  --output-dir ./model_output \
  --num-boost-round 500 \
  --learning-rate 0.05
```

- Evaluates nDCG@5/10/20, Recall@10/50, HR@10, MRR
- Compares LightGBM vs exact heuristic baseline
- Reports feature importance, latency benchmark, per-query win rates
- Our run: early stopped at iteration 141, 948 KB model

---

## 🔌 Integration Into ResearchIT

See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for the complete step-by-step guide.

**Quick summary** — add to `app/recommend/reranker.py`:

```python
import lightgbm as lgb

# Load once at startup
_lgb_model = None
try:
    _lgb_model = lgb.Booster(model_file="production_model/reranker_v1.txt")
    print("[reranker] LightGBM model loaded (948 KB)")
except Exception:
    print("[reranker] LightGBM unavailable — using heuristic fallback")

# In rerank_candidates():
if _lgb_model is not None:
    features = compute_features_v2(user, candidates)  # 37-dim feature vector
    scores = _lgb_model.predict(features)
else:
    scores = heuristic_score(candidates)  # existing fallback
```

The heuristic scorer remains as a permanent fallback. If the model file is missing or fails to load, the system silently uses the heuristic. No user-facing impact.

---

## ⚠️ Known Limitations

### Citation ≠ User Interest
Citation pseudo-labels ("cited in bibliography") ≠ real user signals ("saved in feed"). A foundational paper like "Attention Is All You Need" gets label=2 in citation data but might be dismissed by users who've already read it.

**Mitigation:** The `candidate_log_citations` and `candidate_citations_per_year` features help learn a popularity curve. When 500+ real user interactions accumulate, retrain on actual save/dismiss data — the 11 user behavior features (20-30) activate and the model learns real preferences.

### Sampled Corpus (50K of 1.6M)
We sampled 50K papers for S2 API calls due to rate limiting, yielding 242K in-corpus edges. The full corpus would produce ~8-10M edges with a valid API key or S2 bulk download. More edges → more training data → better model.

### S2 API Key
The provided API key returned 403 Forbidden. We ran unauthenticated at ~1.5s delay per request. A working key or the S2 bulk dataset download would be significantly faster.

### Pseudo-Label Heuristic Baseline
The heuristic baseline uses `qdrant_cosine_score` as proxy for `ewma_longterm_similarity` (feature 20) since real EWMA profiles don't exist for pseudo-users. This is fair but means the heuristic baseline nDCG (0.264) is lower than what the real heuristic achieves in production with actual user profiles.

---

## 🗺️ Roadmap

| Step | Description | Status |
|------|-------------|--------|
| ~~1. Citation edges~~ | S2 API scraping | ✅ Done (242K edges) |
| ~~2. Training triples~~ | Qdrant ANN + Turso → labeled data | ✅ Done (98K rows) |
| ~~3. LightGBM training~~ | lambdarank + eval | ✅ Done (nDCG@10: 0.879) |
| ~~4. Synthetic testing~~ | Test suite on synthetic data | ✅ Done (6 categories pass) |
| 5. `compute_features()` expansion | 5→37 features in reranker.py | 🔜 Next (Opus) |
| 6. Model loading + fallback | Wire `lgb.Booster` into reranker | 🔜 Next (Opus) |
| 7. `requirements.txt` update | Add `lightgbm>=4.0` | 🔜 Next (Opus) |
| 8. Integration testing + deploy | End-to-end verification | 🔜 Next (Opus) |
| 9. Real user data retrain | 500+ interactions → retrain with features 20-30 | Future |
| 10. Phase 8b: TinyBERT + LightGBM-2 | Cross-encoder reranker stage | Future |

---

## 📚 References

- **ResearchIT Doc 06 §3.1** — LightGBM lambdarank architecture decision
- **ResearchIT Doc 07 §A6** — Time-split evaluation protocol
- **ResearchIT Doc 07 §B.4** — Multi-stage reranker architecture (LightGBM-1 → TinyBERT → LightGBM-2)
- **PinnerSage** (Pal et al., KDD 2020) — Ward clustering + importance-weighted retrieval
- **Taobao ULIM** (Meng et al., RecSys 2025) — Quota allocation, +5.54% CTR
- **YouTube DNN** (Xia et al., 2023) — 3× gain from negative signals in reranking
- **RRF Analysis** (Bruch et al., SIGIR 2022) — RRF optimizes Recall not nDCG

---

## 🛠️ Dependencies

```
lightgbm>=4.0
httpx>=0.24
pyarrow>=12.0
numpy>=1.24
qdrant-client>=1.17
tqdm>=4.65
```

---

## 📄 License

This model and pipeline are part of the ResearchIT project by [@siddhm11](https://huggingface.co/siddhm11).
