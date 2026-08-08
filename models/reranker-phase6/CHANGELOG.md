# Changelog

## v1.0.0 — Production Model (2025-04-27)

### Trained on Real Data
- **242,179 citation edges** from Semantic Scholar API (50K sampled papers from 1.6M corpus)
- **90,993 training rows** (1,857 queries, pre-2023 papers)
- **7,007 eval rows** (143 queries, 2023+ papers)
- Strict time-split with verified no temporal leakage

### Results
- nDCG@10: **0.8791** (vs heuristic 0.2641, +232.8%)
- nDCG@5: **0.8250** (vs heuristic 0.1819, +353.6%)
- MRR: **0.8795** (vs heuristic 0.2906, +202.7%)
- HR@10: **1.0000** (vs heuristic 0.6638, +50.6%)
- Latency: **0.371ms** per 100 candidates
- Model size: **948 KB**

### Top Features
1. `candidate_num_cited_by` (75,203)
2. `age_ratio` (7,597)
3. `candidate_position` (6,765)

---

## v0.1.0 — Synthetic Proof of Concept (2025-04-27)

- Full pipeline tested on synthetic data
- nDCG@10: 0.9985 (vs heuristic 0.9111)
- 6-category test suite passing
- 0.088ms latency, 286 KB model

---

## v0.0.1 — Pipeline Design (2025-04-27)

- 3-script pipeline created
- 37-feature schema designed
- Test suite written
