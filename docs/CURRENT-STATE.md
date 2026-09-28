# Current technical contract

Updated: 2026-09-28. Deployed to the Hugging Face Space at `67cba21` on
2026-09-28; the live checks are in DISCOVERY-PLAN.md (deployment record).
Re-verify with `/healthz/*` rather than trusting this date.

## Source of truth and maintenance

Use this document and the code to answer what exists. Use Doc 06's dated
architecture decisions for why. Historical phase plans are not release status.
When a feature changes, update its behavior, configuration, failure path, source
anchor, and validation record in the same change. Separate implemented, tested,
and deployed. Never copy old corpus counts or benchmark results into current claims.

## Contracts and implementation anchors

| Concern | Current behavior | Source |
|---|---|---|
| Search | Original + optional rewritten query; dense shard fanout and independent FTS5; Zilliz without FTS; RRF; optional top-10 MiniLM; title/citation boost | `app/hybrid_search_svc.py`, `app/config.py` |
| Missing encodings | Available FTS query forms still run; actual contributing arms determine `retrieval_mode` | `app/hybrid_search_svc.py` |
| Search total miss | arXiv keyword fallback; no invented hits for unfamiliar concepts | `app/routers/search.py` |
| Detailed interests | Saved papers -> long/short/negative EWMA + Ward medoids; parameters unchanged | `app/recommend/profiles.py`, `clustering.py` |
| Multi-interest balance | Importance-weighted quota, within-cluster MMR, quota-bound served order | `fusion.py`, `app/routers/recommendations.py` |
| Scoring | Personalized heuristic default; optional citation-trained LightGBM | `reranker.py`, `config.py` |
| Cold start | Category popularity candidates; local per-category interleaving when sidecar exists; epsilon-greedy fresh block and deterministic recycled block | `app/discovery_svc.py`, recommendation router |
| Refresh | Personalized candidate retrieval excludes last seven days of served papers; one exhaustion retry allows labelled repeats | recommendation router, `app/db.py` |
| Paging | Cached ordered pool per query ID, bound to user; per-user lock includes serving/impression write in one process | recommendation router |
| Preferences | Editable `/interests`; zero to eight known categories; no deletion of existing saves | `app/routers/onboarding.py` |
| First seeds | Suggestions from selected categories; explicit search/save; count advances on successful response | onboarding router and seed templates |
| Reading history | Latest 50 distinct opens or served papers; metadata failure retains a navigable ID | `app/routers/saved.py`, `app/db.py` |
| Feedback semantics | Latest save/unsave/dismissal sets explicit state; click/view never overrides it; unsave survives cache reload | `get_current_feedback`, `get_save_history`, `user_state.py` |
| Library | All current saves; the 20-example retrieval deque no longer limits the library view | saved router |
| Attribution | Ranked visits are clicks with origin fields; direct visits are views without fabricated policy/propensity | `app/routers/paper.py` |
| Persistence | Local SQLite; periodic Turso replication of four core tables only | `app/turso_sync.py:TABLES` |

## Constants worth protecting

- EWMA alpha: long 0.03, short 0.40, negative 0.15.
- Ward: normalized embeddings, Euclidean distance, adaptive clusters, maximum 7.
- Recommendation fusion: quota, never cross-interest RRF; per-cluster MMR lambda 0.6.
- Search fusion: RRF k=60; cross-encoder depth 10 by default.
- Feed: 10 core cards, up to 2 exploration cards; ranked pool 60.
- Refresh suppression: 7 days for personalized retrieval. Cold start remembers the
  available local impression history and recycles only when fresh supply is thin.
- Policy version: `v10_fresh_discovery`; search `search_v2_independent_lexical`.

No new model, training procedure, EWMA weight, or clustering threshold was added.
Category selection guides the initial feed and existing features; it does not
replace a reader's saved-paper profile. The existing heuristic does not consume
all 37 model inputs, including the onboarding-match feature.

## Measurement caveats

An exposure is server delivery, not proven viewport visibility. An open is not a
completed read. Existing propensity fields mix conditional cold-start slot
probabilities and exploration-selection probabilities; they are not a validated
universal counterfactual estimator. Audit their conditioning/support before IPS,
SNIPS, or DR evaluation. Do not turn raw clicks into positive profile updates
without evidence that those signals improve relevance.

## Operational boundaries

- Cookie identity, in-process locks/caches, and local SQLite assume one serving
  process. Multiple replicas need explicit state ownership and concurrency design.
- Follows and exposure logs are not replicated. Discovery history can reset on
  ephemeral storage replacement. Open history is in the replicated interactions
  table, subject to the existing sync interval and recovery limitations.
- The pinned sidecar is not a live corpus. Full abstracts and fresh publication
  coverage must be measured after ingestion and sidecar publication.
- Missing sidecar triggers one remote popularity query; this fallback does not
  provide the local per-category balancing guarantee.
- Exhaustion can return familiar papers. Partial pools may be short. Dependency
  outages must be distinguished from an empty library.
- Candidate quality, real model latency, user satisfaction, and production
  recovery are not established by passing mocked/local tests.

See DISCOVERY-PLAN.md for validation and remaining release gates.

## Recommendation audit — 2026-09-25

A read-only Hugging Face Daily Papers adapter now exists in
`app/hf_papers_svc.py` for shadow evaluation. It is not called by the live feed.
`scripts/run_recommendation_audit.py` produces a standalone HTML/JSON/JUnit report
with real-code structural stress tests, source contract tests, optional live
vector/HTTP probes, exact-vs-ANN agreement and optional human relevance metrics.
See `reports/recommendations/README.md`. Missing model/service prerequisites
are reported as blocked, never as successful semantic evaluations.

### Local source pipeline implementation

`discovery_store.py`, `discovery_worker.py`, `discovery_shadow.py` and
`scripts/hf_discovery.py` implement persistent dated source observations, a
single-host periodic worker, bounded metadata/embedding preparation, freshness
eligibility, scratch-profile baseline capture and source-capped offline comparison.
See HF-DISCOVERY-OPERATIONS.md. Readiness is local/shadow only; no cloud indexing
or serving integration is enabled. The attempted live collection failed with
ConnectError. Synthetic demonstrations never count as real relevance evidence.

### Shadow worker lifecycle follow-up (2026-09-25)

The FastAPI lifespan now starts/stops an opt-in shadow scheduler via
`HF_DISCOVERY_MODE=shadow`, with the same store lock as the CLI. The new
`/healthz/discovery` endpoint reports cached scheduler/collection/preparation
status and explicitly reports `serving_enabled=false`. Shared model-loading
failures no longer consume candidate retry budgets. Local network/runtime
restrictions still block live validation; production settings were not changed.
See `HF-DISCOVERY-OPERATIONS.md` for configuration and remaining serving gates.
