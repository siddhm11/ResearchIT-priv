---
name: researchit-codebase-overview
description: "Explain the ResearchIT codebase architecture and current state. Use for onboarding, project overviews, and accurate summaries of how the system works. Triggers: codebase overview, architecture summary, explain this project, how this works, system map."
argument-hint: "Specify audience (dev/stakeholder), depth (brief/standard/deep), and focus (search/recs/data)."
---

# ResearchIT Codebase Overview

## When to Use
- The user asks for a full understanding of the codebase or architecture.
- You need to produce a top-level system map or explain how components interact.
- You need a concise but accurate "what is happening here" summary.

## Inputs to Ask For (if missing)
- Audience: developer vs stakeholder.
- Depth: brief, standard, or deep.
- Focus areas: search, recommendations, data layer, evaluation.

## Required Sources (read in this order)
1. CLAUDE.md (rules and source-of-truth doc map).
2. docs/research/06-Deep-Research-Verdict.md (architecture decisions).
3. README.md (current system summary).
4. docs/walkthroughs/03-Code-Summary-and-Test-Plan.md (module map).
5. docs/walkthroughs/04-Next-Steps-and-Phase-Plan.md (current phase).

## Procedure
1. State the product goal in one sentence and the system constraints (CPU-only, latency budget).
2. Describe the high-level architecture (frontend, backend, vector stores, metadata DB, SQLite).
3. Summarize the two main pipelines:
   - Search: rewrite -> encode -> dense+sparse -> RRF -> title/citation boost.
   - Recommendations: clustering -> quota -> rerank -> MMR -> exploration.
4. Call out invariants from doc 06 (quota for recs, RRF for search, alpha values, MMR lambda).
5. Explain data flow and caching (Turso LRU, Qdrant vector cache, SQLite metadata cache).
6. State current phase status and what is out of scope.

## Output Format
- 6 to 10 bullet points, ordered by importance.
- Short "where to look" section with key files.
- If stakeholder audience: avoid implementation detail and emphasize outcomes.

## Key Files to Cite
- app/main.py
- app/routers/recommendations.py
- app/routers/search.py
- app/hybrid_search_svc.py
- app/recommend/*
- app/qdrant_svc.py, app/zilliz_svc.py, app/turso_svc.py
- app/db.py
