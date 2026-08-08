---
name: researchit-data-layer
description: "Explain the data/storage layer (SQLite, Turso metadata, Qdrant dense vectors, Zilliz sparse vectors). Use for data integrity, schema questions, caching behavior, and ID handling. Triggers: database schema, metadata cache, Qdrant mapping, Zilliz schema." 
argument-hint: "Specify the component(s) and whether you want schema details or runtime behavior."
---

# Data and Storage Layer Analysis

## When to Use
- The user asks about storage, caching, or schemas.
- You need to validate data integrity or ID handling.
- You need to explain how metadata or vector mappings work.

## Required Sources
1. app/db.py (SQLite schema + migrations)
2. app/turso_svc.py (metadata + caches)
3. app/qdrant_svc.py (ID mapping + vector cache)
4. app/zilliz_svc.py (sparse schema + search)
5. app/arxiv_svc.py (API fallback + ID normalization)

## Procedure
1. Summarize each store and its responsibility (SQLite, Turso, Qdrant, Zilliz).
2. Explain arXiv ID handling (always string; never integer coercion).
3. Document caches (vector cache, metadata LRU, trending cache).
4. Note schema migrations and instrumentation columns.
5. Identify data consistency boundaries and fallbacks.

## Output Format
- Component-by-component description.
- Tables/fields summary for SQLite.
- Integrity rules and common pitfalls.
