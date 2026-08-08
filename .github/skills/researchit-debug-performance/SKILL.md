---
name: researchit-debug-performance
description: "Debug performance and quality issues in search or recommendations. Use for latency spikes, slow retrievals, or degraded relevance. Triggers: performance issue, slow search, slow recs, latency debug." 
argument-hint: "Specify area (search/recs/data), symptoms, and whether to propose fixes." 
---

# Debugging and Performance Profiling

## When to Use
- Latency regressions or slow responses appear.
- Search or recommendation quality drops unexpectedly.
- External services time out or return empty results.

## Required Sources
1. app/qdrant_svc.py (vector cache, retrieve latency)
2. app/turso_svc.py (metadata cache, trending cache)
3. app/hybrid_search_svc.py (RRF pipeline)
4. app/routers/recommendations.py (candidate flow + oversample)
5. app/recommend/reranker.py (model load, feature cost)

## Procedure
1. Identify the failing pipeline (search vs recommendations).
2. Check cache hit rates conceptually (vector and metadata caches).
3. Inspect candidate fetch sizes and oversampling factors.
4. Review service fallbacks (Zilliz, Turso, arXiv).
5. Isolate latency contributors and propose focused fixes.

## Output Format
- Symptom -> probable cause mapping.
- Targeted checks in code.
- Minimal, low-risk fix options.
