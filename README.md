---
title: ResearchIT
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
short_description: A multi-interest arXiv recommender over 1.8M papers
models:
  - BAAI/bge-m3
  - cross-encoder/ms-marco-MiniLM-L6-v2
  - siddhm11/researchit-reranker-phase6
datasets:
  - siddhm11/researchit-metadata
---

# ResearchIT — personalized paper discovery

ResearchIT is a FastAPI + HTMX + Jinja2 application for searching arXiv and
building a reading feed around several interests. This description reflects the
**local implementation on 2026-09-24**, not a verification of the live deployment.

## What works

- Semantic and lexical search, optional query rewriting, and search reranking.
- Category onboarding with starter-paper suggestions and an editable interests page.
- Saves, unsaves, dismissals, paper pages, related papers, and optional explanations.
- A multi-interest feed that preserves smaller interests through quota allocation.
- Mostly fresh discovery on refresh, plus recently opened/discovered history.
- Curated collections and an API for a separate 3D map client.
- Local user storage, partial remote replication, health probes, rate limiting, and CI.

## Current architecture

```text
Browser: HTMX + Jinja2
          |
      FastAPI
          |-- Search: original query + optional Groq rewrite
          |     |-- BGE-M3 -> Qdrant dense shard fanout
          |     |-- FTS5 over local metadata; Zilliz fallback
          |     `-- RRF -> MiniLM top-10 rerank -> title/citation adjustments
          |
          |-- Discovery: preferences / saved-paper interests
          |     `-- retrieval -> quota + heuristic scoring -> within-cluster MMR
          |
          |-- SQLite: interactions, profiles, clusters, history, caches
          |     `-- periodic Turso backup of four core user tables
          |
          `-- Metadata: pinned local sidecar -> Turso -> arXiv fallback
```

**Search:** FTS5 is the preferred lexical backend. It works even when embedding
generation fails. If the sidecar/FTS index is unavailable, learned sparse retrieval
uses Zilliz and requires a successful BGE encoding. Original and rewritten query
forms remain distinct. RRF combines available lists; MiniLM reranks 10 candidates
by default. Search does not apply a general recency boost that buries classic
papers. The HTTP route can fall back to the arXiv keyword API.

**Vector stores:** the code supports the primary Qdrant collection, optional shard
B, and an optional recent-papers shard. Fanout requires configuration. Repository
records describe roughly 1.8M indexed papers; verify actual counts and dates with
the deployed health endpoints rather than treating historical counts as live data.

**Recommendation scorer:** `RERANKER_MODE=heuristic` is the default. The optional
LightGBM model has 141 trees and 37 inputs, but no splits on personalization
features 20–30. Its citation-based offline evaluation is not evidence of a better
personalized feed. Candidate retrieval and quotas personalize the overall system
independently of the scorer.

## Discovery and reading behavior

| Reader state | Main serving path |
|---|---|
| No saves | Popularity/recency candidates in chosen categories, or a broad fallback |
| 1–2 saves | Qdrant recommendation from saved examples |
| 3–4 saves | Long-term EWMA profile retrieval, when the profile is ready |
| 5+ saves | Ward interest clusters, quota retrieval, scoring, within-cluster MMR |

These are eligibility thresholds, not guarantees; unavailable vectors/profiles
can lead to a simpler tier.

Personalized retrieval excludes papers served in the last seven days. A fresh
request rebuilds the pool; pagination continues a cached ordering owned by that
reader. If no fresh ranked candidates can be found, one retry permits repeats
and the UI labels them. Saved and dismissed papers remain excluded. A small pool
may return fewer papers. Refresh novelty does not mean newly published research.

Cold start uses impression memory and epsilon-greedy ordering. Fresh candidates
precede an oldest-first repeat block when supply runs low. The indexed sidecar
allows starter retrieval to interleave categories; without it, one cached remote
query avoids multiplying expensive remote scans. No history is deleted to recycle.

`/interests` edits category priors; saving specific papers is the strongest explicit
way to teach detailed interests. `/history` shows opened pages, and
`/history?view=discovered` shows recently served feed papers. Neither claims the
paper was read. Opens never become saves, dislikes, or EWMA updates. Direct visits
are `view` events with no fabricated ranking attribution.

## Run locally

Use Python 3.12 and a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
# Configure only the services you intend to use.
python run.py
```

The app serves port 7860. `.env.local` is loaded before `.env`; process environment
variables take precedence over both. Set `TURSO_SYNC_DISABLED=1` locally unless you
are deliberately testing replication against a scratch database. Metadata reads
can use Turso independently of replication.

The large metadata sidecar is not committed. Docker downloads a pinned dataset
revision, while local runs need `METADATA_SIDECAR_PATH` pointing at a suitable
SQLite file. Without models or remote credentials, the full search/feed pipeline
is not available; graceful fallback is not a substitute for a configured corpus.

## Configuration

See [.env.example](.env.example) and [app/config.py](app/config.py).

| Setting | Purpose |
|---|---|
| `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION` | Primary dense store |
| `QDRANT_B_*`, `SEARCH_FANOUT_B` | Optional second dense shard |
| `QDRANT_RECENT_*`, `SEARCH_FANOUT_RECENT` | Optional recent-papers shard |
| `METADATA_SIDECAR_PATH`, `SPARSE_BACKEND` | Local metadata and lexical search |
| `TURSO_URL`, `TURSO_DB_TOKEN` | Metadata fallback and core user-data backup |
| `TURSO_SYNC_DISABLED`, `DB_PATH` | Isolate local development and scratch storage |
| `ZILLIZ_URI`, `ZILLIZ_TOKEN` | Learned sparse fallback |
| `GROQ_API_KEY` | Optional rewriting, summaries, and explanations |
| `RERANKER_MODE` | `heuristic` (default), `lightgbm`, or `auto` |
| `SEARCH_BGE_RERANK`, `SEARCH_RERANK_TOP_N` | Search cross-encoder flag and depth |
| `MAP_QDRANT_*`, `SPACE_APP_URL`, `SPACE_SERVICE_TOKEN` | Optional companion map |

## Tests and evaluation

```bash
python -m pytest tests/ -m "not live and not browser" -q
python -m compileall -q app scripts tests
```

Tests use temporary user databases and disable Turso replication. The focused
new coverage is in `test_discovery_refresh.py`, `test_discovery_journey.py`, and
`test_search_resilience.py`. Keep test counts in dated validation records, rather
than a permanently stale badge here.

`test_e2e_browser.py` requires Playwright and a running app. Browser tests can
create interactions: use scratch storage and disable replication. Live tests,
real model inference, browser behavior, and local unit tests are separate checks.
Existing `scripts/eval_search_quality.py` and `scripts/eval_recs_quality.py` provide
quality evaluation starting points; they do not establish a measured engagement
lift or a finished held-out evaluation framework.

## Operations and known limits

Docker is configured for Hugging Face Spaces, CPU inference, and local SQLite at
`/tmp/interactions.db`. Core replication covers interactions, user profiles,
clusters, and onboarding. It does **not** currently cover collection follows,
feed exposure logs, impression history, or explanation caches. Replication is
periodic, not synchronous durability. Timestamp cursors and deletion semantics
need additional recovery validation. Browser-cookie identity is not an account
or cross-device recovery system.

Health routes include `/healthz/deep`, `/healthz/shards`, and `/healthz/reranker`.
The keepalive workflow probes services and one stored map position; it does not
prove relevance or validate the companion map's streamed tiles. CI exercises a
lightweight dependency set, not a complete fresh production image build.

Ingestion and abstract-backfill scripts exist, but no scheduled ingestion workflow
is checked in. Confirm any external schedule and actual publication coverage.
“Popular in the indexed corpus” is not “trending now.” YouTube links, learned
collaborative recommendations, and verified real-time trends remain future work.

## Documentation

- [Current technical contract](docs/CURRENT-STATE.md): behavior, limits, and source anchors.
- [Discovery plan](docs/DISCOVERY-PLAN.md): product decisions, tests, and release gates.
- [Documentation index](docs/README.md): current guides versus historical plans.
- [Architecture decisions](docs/research/06-Deep-Research-Verdict.md): rationale and dated amendments.
- [Agent guidance](CLAUDE.md): invariants contributors should preserve.
