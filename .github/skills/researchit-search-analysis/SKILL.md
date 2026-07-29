---
name: researchit-search-analysis
description: "Explain or analyze the hybrid semantic search pipeline (rewrite, encode, dense+sparse, RRF, title/citation boost). Use for search quality, latency, and correctness reviews. Triggers: search pipeline, hybrid search, RRF, BGE-M3 search." 
argument-hint: "Specify: explain vs debug, and whether to include latency hotspots."
---

# Search Pipeline Analysis

## When to Use
- The user wants to understand or debug search results.
- You need to review hybrid search correctness.
- You are asked about RRF usage or query rewriting.

## Required Sources
1. app/routers/search.py
2. app/hybrid_search_svc.py
3. app/embed_svc.py
4. app/qdrant_svc.py
5. app/zilliz_svc.py
6. app/groq_svc.py
7. app/turso_svc.py and app/arxiv_svc.py

## Procedure
1. Trace the full pipeline from query to results.
2. Call out the dual-encode design (original + rewrite) and why it exists.
3. Verify RRF usage is limited to search fusion (correct per doc 06).
4. Explain title/citation boosts and their intended effect.
5. Document fallback behavior when any component fails.
6. Summarize latency hotspots and caching layers.

## Output Format
- Step-by-step pipeline description.
- Fallbacks and failure handling.
- Notes on ranking behavior and edge cases.
