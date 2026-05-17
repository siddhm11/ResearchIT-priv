---
name: researchit-recs-analysis
description: "Analyze and explain the recommendation pipeline. Use for recs debugging, feature reviews, pipeline changes, or explaining multi-interest behavior. Triggers: recommendation pipeline, recs analysis, multi-interest, quota fusion, reranker." 
argument-hint: "Specify the task (explain/debug/change), expected output (summary/findings), and whether to include tests."
---

# Recommendation Pipeline Analysis

## When to Use
- The user wants a deep explanation of recommendations or changes.
- You need to verify rules like quota fusion, EWMA alphas, or MMR usage.
- You are asked to debug rec quality or performance.

## Required Sources
1. CLAUDE.md and docs/research/06-Deep-Research-Verdict.md (non-negotiables).
2. app/routers/recommendations.py (pipeline and instrumentation).
3. app/recommend/profiles.py (EWMA parameters).
4. app/recommend/clustering.py (Ward + medoids + stabilization).
5. app/recommend/fusion.py (quota logic).
6. app/recommend/reranker.py (LightGBM + features).
7. app/recommend/diversity.py (MMR + exploration).

## Procedure
1. Identify which tier is active and the fallback sequence.
2. Validate invariant rules:
   - Search uses RRF, recommendations do not.
   - Quota fusion with floor; MMR lambda is 0.6.
   - alpha_long=0.03, alpha_short=0.40, alpha_neg=0.15.
3. Trace candidate flow:
   - Medoids -> per-cluster search -> dedup -> rerank -> MMR -> exploration.
4. Check instrumentation fields: query_id, propensity, policy_id.
5. Summarize likely failure modes: missing vectors, empty clusters, cache misses.
6. Recommend targeted tests or metrics to verify changes.

## Output Format
- Pipeline summary with stages and main functions.
- Invariants checklist (pass/fail).
- Risks and suggested tests.

## Notes
- Never propose RRF for multi-medoid recommendations.
- Do not introduce cross-encoders into the hot path.
