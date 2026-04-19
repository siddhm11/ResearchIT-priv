# ResearchIT Documentation

All project documentation organized by purpose. Each document has a specific role in the project lifecycle.

---

## 📁 Folder Structure

```
docs/
├── README.md                     ← you are here
│
├── TASK-TRACKER.md               ← master checklist (all phases)
│
├── research/                     ← deep research & strategic thinking
│   ├── 01-Vision-Instagram-for-Research.md
│   ├── 02-Recommendation-System-Blueprint.md
│   ├── 03-MultiInterest-Recommender-Architecture.md
│   ├── 04-Technical-Roadmap-Legacy.md
│   ├── 05-Evolution-Of-Onboarding-And-Interests.md
│   └── 06-Deep-Research-Verdict.md
│
├── phases/                       ← what we built & what we plan to build
│   ├── PHASE1-Zero-ML-Recommender.md
│   ├── PHASE2-Hybrid-Search-Plan.md     (prototype reference)
│   └── PHASE3-Hybrid-Semantic-Search.md (ACTIVE PHASE 3 PLAN)
│
├── walkthroughs/                 ← detailed implementation records
│   ├── 01-Phase1-Code-Tour.md
│   ├── 02-Phase2-MultiInterest-Recommender.md
│   ├── 03-Code-Summary-and-Test-Plan.md
│   └── 04-Next-Steps-and-Phase-Plan.md
│
notebooks/                        ← Kaggle reference notebooks (not in docs/)
├── README.md
├── 01-bme-upload.ipynb             (BGE-M3 encode + upload 1.6M papers)
├── 02-bme-arxiv-test.ipynb         (search quality + encoding tests)
└── 03-check-search-bq-prm.ipynb    (BQ vs PRM benchmark)
```

---

## 📚 Reading Order

If you're new to this project, read these in order:

### 1. Understand the Vision
**[01-Vision-Instagram-for-Research.md](research/01-Vision-Instagram-for-Research.md)**
The strategic blueprint. Covers competitive landscape, UX patterns from TikTok/Spotify/Pinterest, social dynamics, differentiation features, and business model. This is "why we're building this."

### 2. Understand the Technical Foundation
**[02-Recommendation-System-Blueprint.md](research/02-Recommendation-System-Blueprint.md)**
The initial deep research on recommendation architectures. Covers user modeling, content-based vs collaborative filtering, cold start strategies, and evaluation metrics. This is "how recommendation systems work in general."

### 3. Understand the Chosen Architecture
**[03-MultiInterest-Recommender-Architecture.md](research/03-MultiInterest-Recommender-Architecture.md)**
The definitive architecture RFC. EWMA temporal decay, Ward hierarchical clustering, LightGBM re-ranking, MMR diversity. Validated by Twitter, Pinterest, and Alibaba production systems. **This is the blueprint we implemented.**

### 4. See the Architectural Evolution
**[05-Evolution-Of-Onboarding-And-Interests.md](research/05-Evolution-Of-Onboarding-And-Interests.md)**
Documents the founder's pivot from explicit onboarding subject vectors to implicit behavioral tracking. Captures the original vision vs. the current approach and why the change was made.

**[06-Deep-Research-Verdict.md](research/06-Deep-Research-Verdict.md)** ⭐ *Latest Research*
The comprehensive verdict that resolves contradictions across all prior documents. Proposes a **three-layer hybrid** (coarse categories + seed papers + behavioral clustering). Identifies faults in Doc 03 (RRF→quota, α correction). The definitive architectural reference going forward.

### 5. See What Phase 1 Built
**[PHASE1-Zero-ML-Recommender.md](phases/PHASE1-Zero-ML-Recommender.md)**
What was built first: zero-ML-inference recommender using Qdrant's BEST_SCORE Recommend API, SQLite event logging, and arXiv metadata caching. The working foundation.

**[01-Phase1-Code-Tour.md](walkthroughs/01-Phase1-Code-Tour.md)**
A file-by-file walkthrough of every piece of the Phase 1 codebase: entry points, routers, services, database, templates, and tests.

### 6. See What Phase 2 Built
**[02-Phase2-MultiInterest-Recommender.md](walkthroughs/02-Phase2-MultiInterest-Recommender.md)**
What was just built: PinnerSage-style multi-interest engine with EWMA profiles, Ward clustering, prefetch+RRF, heuristic re-ranking, and MMR diversity. 88 tests passing.

### 7. Review Core Code & Automation
**[03-Code-Summary-and-Test-Plan.md](walkthroughs/03-Code-Summary-and-Test-Plan.md)**
Summarizes all structural backend modules, frontend files, and breaks down our three-layered ongoing testing strategies (Automated, Manual, and Analytic Evaluation).

### 8. What's Next — The Revised Phase Plan
**[04-Next-Steps-and-Phase-Plan.md](walkthroughs/04-Next-Steps-and-Phase-Plan.md)** ⭐ *Start Here for Next Steps*
The master roadmap synthesizing all 6 research documents. Resolves contradictions between docs, captures the founder's thinking evolution, and lays out Phases 3-9 in priority order. Includes the three highest-impact next actions.

### 9. Phase 3 Plan (Current Focus)
**[PHASE3-Hybrid-Semantic-Search.md](phases/PHASE3-Hybrid-Semantic-Search.md)** ⭐ *Active Implementation Plan*
The detailed implementation plan for hybrid semantic search. Covers architecture, all new/modified files, Zilliz schema, BGE-M3 encoding, RRF fusion, HF Spaces deployment, latency budget, and 8-step implementation order.

### 10. Data Preparation Notebooks
**[notebooks/README.md](../notebooks/README.md)** — Index + extracted schema details.
- `01-bme-upload.ipynb` — How 1.6M papers were encoded and uploaded to Qdrant + Zilliz
- `02-bme-arxiv-test.ipynb` — BGE-M3 encoding + search quality prototype
- `03-check-search-bq-prm.ipynb` — BQ vs PRM quantization benchmark

---

## 📄 Document Status

| Document | Status | Notes |
|---|---|---|
| 01 — Vision (Instagram for Research) | ✅ Complete | Strategic north star |
| 02 — Recommendation Blueprint | ✅ Complete | Initial research, still relevant |
| 03 — Multi-Interest Architecture | ✅ Implemented | **The RFC we implemented** — has 4 known faults identified in Doc 06 |
| 04 — Technical Roadmap | ⚠️ Legacy | Superseded. Kept for reference only |
| 05 — Evolution of Onboarding | ✅ Complete | Documents the subject-vector → behavioral pivot |
| 06 — Deep Research Verdict | ✅ Complete | **The definitive architectural reference** — resolves all contradictions |
| Phase 1 Walkthrough | ✅ Complete | Still accurate for Phase 1 code |
| Phase 1 Code Tour | ✅ Complete | File-by-file walkthrough |
| Phase 2 Recommender Walkthrough | ✅ Complete | Multi-interest engine |
| Codebase Summary & Test Plan | ✅ Complete | Summarizes codebase & testing |
| Next Steps & Phase Plan | ✅ Complete | **Master roadmap for Phases 3-9** |
| Phase 2 Hybrid Search Plan | 📋 Prototype reference | Superseded by PHASE3-Hybrid-Semantic-Search.md as the active plan |
| **Phase 3 Hybrid Semantic Search** | **📋 Active Plan** | **The current implementation guide for Phase 3** |
| Task Tracker | ✅ Active | Master checklist for all phases |

---

## 🏗️ Architecture Evolution

```
Phase 1 (completed)
  └── Qdrant BEST_SCORE with raw paper IDs
       ├── Works from 1 save
       └── No temporal awareness, no diversity

Phase 2a (completed)
  └── EWMA profile embeddings
       ├── Long-term (α=0.03) + Short-term (α=0.40) + Negative (α=0.15)
       └── Activates at 3+ saves

Phase 2b (completed)
  └── Ward clustering + Qdrant prefetch+RRF
       ├── Auto-detects K interests per user (1-7)
       ├── Single API call, server-side parallel ANN
       └── Activates at 5+ saves

Phase 2c (completed)
  └── Heuristic re-ranking + MMR diversity
       ├── 5-feature scorer (40% relevance, 25% session, 15% recency, 10% rank, -15% negative)
       ├── MMR diversity (λ=0.6) + exploration injection (2 papers)
       └── Upgrade path: swap heuristic for LightGBM at ≥500 interactions

Phase 3 (NEXT — hybrid semantic search)
  └── Replace arXiv keyword API with vector-based search
       ├── BGE-M3 query encoding (loaded at startup)
       ├── Dense (Qdrant) + Sparse (Zilliz) parallel retrieval
       ├── RRF fusion (correct for search: same query, different retrievers)
       └── Deployment: Hugging Face Spaces (Docker SDK, 16GB RAM, 2 vCPUs)

Phase 4 (planned — recommendation pipeline fixes)
  └── RRF → quota fusion, α_long 0.10 → 0.03, negative profile wiring,
       pre-populate metadata store

Phase 5 (planned — cold-start onboarding)
  └── arXiv category multiselect + seed paper import + ORCID

Phase 6+ (future)
  └── LightGBM lambdarank, evaluation framework, LLM summaries,
       collaborative filtering, exploration
```
