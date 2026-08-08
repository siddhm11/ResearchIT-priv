---
name: researchit-reranker-explainer
description: "Explain the LightGBM reranker, feature schema, and fallback behavior. Use for model integration checks, feature debugging, or deployment validation. Triggers: reranker, LightGBM, feature schema, model loading." 
argument-hint: "Specify: explain, validate, or troubleshoot."
---

# Reranker and Feature Schema Explainer

## When to Use
- The user asks how the reranker works or which features are used.
- You need to validate model loading and fallback behavior.
- You are reviewing feature wiring or scoring behavior.

## Required Sources
1. app/recommend/reranker.py
2. models/reranker-phase6/production_model/feature_schema.json
3. app/routers/health.py
4. app/routers/recommendations.py (feature wiring)

## Procedure
1. Confirm model load paths and fallback logic.
2. Verify the 37-feature ordering matches the schema.
3. Explain which features are active in recommendations and how they are computed.
4. Confirm health endpoint expectations (/healthz/reranker).
5. Provide a concise explanation of latency and why cross-encoders are excluded.

## Output Format
- Model load status + fallback behavior.
- Feature group summary (content, behavior, cross features).
- Integration checklist.
