# Discovery improvement plan

Date: 2026-09-24. Implemented and validated locally; deployed 2026-09-28 (see the
deployment record below).

## Product contract

The user wants papers related to their interests on the first visit, mostly
fresh choices on refresh, and a way back to recent papers. Search should work
for an unfamiliar concept as well as for a known paper. An opened page is not
proof of reading, and an impression is not a dislike.

## Findings driving this change

1. Tier 0 remembers impressions, but personalized tiers exclude only saves and
   dismissals. Their otherwise-correct relevance ranking repeats on refresh.
2. Completed onboarding redirects to the feed, leaving no preference editor.
   The seed step starts blank and requires knowing what to search for.
3. FTS5 is wired to successful BGE encodings. An embedding failure therefore
   disables a keyword index that does not need embeddings. If only the rewrite
   encodes, the existing zip can also pair it with the wrong query text.
4. Clicks already exist, but readers cannot revisit their recent journey.
5. Docs disagree about the active scorer, sparse backend, persistence, and
   completion status. Historical experiment numbers are not current telemetry.

## Implementation boundaries

- Preserve EWMA parameters, Ward clustering, recommendation quotas, and
  within-cluster MMR. Do not replace relevant recommendations with a shuffle.
- Exclude recently served papers before personalized retrieval (7 days), so
  quotas are computed over eligible candidates. If all tiers produce no fresh
  candidates, retry once without that exclusion and label repeats honestly.
  Never clear impression history just to fill a page.
- Serialize feed generation/recording per user within the single process and
  bind cached feed cursors to their owner. This is not a distributed lock.
- Make category preferences editable, suggest initial seeds from those areas,
  and keep seed search explicit. Categories guide cold start; saved items drive
  behavioral interests. Do not claim category changes instantly retrain a model.
- Run lexical retrieval independently of embedding success; preserve original
  and rewritten query identities. Keep search fusion and rerank constants.
- Expose recently opened papers and recently served discovery papers, with
  honest labels. Keep passive history out of positive/negative EWMA updates.
- Use existing tables and dependencies. Do not ingest a corpus, train a model,
  connect YouTube, or modify live data in this pass.

## Validation gates

- Multiple refreshes in tiers 0–3 advance through eligible papers without
  repeats while supply lasts; saves/dismissals never return via exhaustion.
- Multi-interest guarantees remain intact; no duplicate IDs in a page.
- One user's cursor/history cannot reveal another user's papers. Concurrent
  refresh requests do not race the impression write in the single process.
- Thin or exhausted pools and missing metadata degrade without false promises.
- Keyword retrieval survives failed original/rewrite encodings; partial dense
  or sparse outages preserve usable results and their correct query identities.
- Preference validation, completed-user edits, starter-paper suggestions,
  history ordering, and read-versus-open semantics have regression coverage.
- Run the whole non-live test suite. Browser/live/model checks must be reported
  separately and must not write synthetic interactions to the production DB.

## Local validation record — 2026-09-24

Command: `.venv/bin/python -m pytest -q -m 'not live and not browser'`.
Result: **583 passed, 15 skipped, 15 deselected**. One dependency deprecation
warning concerns Starlette/httpx. `git diff --check` passed.

The new regression suites exercise HTTP routes with real isolated SQLite and
synthetic retrieval candidates. They cover all four feed tiers, concurrent
refresh, exhaustion, cooldown, cursor ownership, pagination, durable decisions,
interest editing, starter suggestions, full saved-library size, history,
unsave reload, and independent lexical retrieval during embedding failures.
Existing clustering, quota, and ranking tests also pass. Tests use temporary
user storage and disable production user-state replication.

A local preview was prepared with synthetic papers and isolated storage, but
starting Uvicorn failed with `operation not permitted` while binding
`127.0.0.1:7865`. Browser interaction and visual checks therefore remain
unverified. External-service/model integration tests were excluded; passing
these tests establishes local behavior, not live relevance or latency.
No deployment was performed.

## Live-service validation record — 2026-09-28

A local server (scratch `DB_PATH`, `TURSO_SYNC_DISABLED=1`) ran against live
Qdrant/Turso and the pinned 2.7 GB sidecar. A scripted journey passed:
onboarding, starter suggestions, seed saves, six tier-1 refreshes with two
clusters and zero repeats, paging, dismiss/unsave, history, interest editing
and keyword-only search. Two defects found and fixed before commit:

1. `starter_papers` ran four concurrent sidecar queries on one shared sqlite3
   connection: 9/10 categories returned empty (once, another thread's rows),
   each fell back to a Turso scan that timed out at 30 s, and failures were not
   cached, so every new visitor waited ~30 s. `local_meta.connection()` now
   gives each thread its own read-only handle; FTS uses it too.
2. Direct `/p/` visits log `view` events, and the home route counted any event
   as history, so a shared-link visitor skipped onboarding permanently. Only
   explicit decisions count now, and cookie-less visits record nothing.

Not covered locally: shards `b`/`recent` (credentials exist only as Space
secrets), BGE-M3 and cross-encoder search, and the arXiv fallback (HTTP 406
from the test machine). Offline suite: 668 passed, 15 skipped, 15 deselected.

## Deployment record — 2026-09-28

Merged as PR #9 and deployed to the Space at `67cba21` (previously `1c7dfd8`).
The build kept the old container serving until the new one was ready. Read-only
checks against the live Space afterwards:

- `/healthz/deep` and `/healthz/shards` healthy: shards a/b/recent at
  899,456 / 697,131 / 202,251 points, sidecar and Turso ok.
- First feed for a new visitor: 7.3 s once (pool build), then ~0.9 s. Before
  the deploy the same request took 42.8 s.
- Feed policy `v10_fresh_discovery`; three refreshes for one visitor, 0 repeats.
- Search uses BGE-M3 (no keyword-only notice), policy
  `search_v2_independent_lexical`, 1.9–4.6 s for three reference queries.
- `/history`, `/interests`, `/saved`, `/onboarding`, `/healthz/discovery`
  (`scheduler: disabled`) all return 200.
- HF's edge returned intermittent 502s (without reaching the app) for about five
  minutes after the switch, then none in 20 consecutive requests.

Known after deploy: feed impressions are not replicated, so this deploy reset
every reader's 7-day refresh memory and "Discovered" history (see Operational
boundaries in CURRENT-STATE.md).

## Next release gates and later work

Before deployment, test a built image against scratch user storage, verify
fresh-container restore, sidecar/FTS coverage and real model availability, and
measure request latency. Existing recovery gaps for follows and exposure logs
remain separate work; this change does not certify persistence completeness.

Before calling a section “trending now,” establish scheduled ingestion,
publication-date coverage, and a defined time-windowed trend signal. Citations
inside a corpus-relative window mean “popular in the indexed corpus.” Add
human-judged emerging-topic and multi-interest cases to the existing evaluation
scripts; a term such as the user's example “Jev AI” may be ambiguous or absent.
Ask readers to refine unfamiliar terms rather than inventing matches.

YouTube is a later comprehension feature: verified paper-to-video links using
arXiv IDs/DOIs in descriptions, explicit provenance, creator/title/date, and
user choice. Do not equate video popularity with paper quality or infer paper
claims from an unverified video match. No YouTube integration is built here.
