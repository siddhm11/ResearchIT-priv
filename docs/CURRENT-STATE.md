# Current technical contract

Updated: 2026-09-30 (feed changes not yet deployed). Deployed to the Hugging Face Space at `67cba21` on
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
| Search | Original + optional rewritten query; dense shard fanout and independent FTS5; Zilliz without FTS; RRF; optional mxbai-rerank-xsmall cross-encoder (top 10; branch, not deployed); title/citation boost | `app/hybrid_search_svc.py`, `app/config.py` |
| LLM features | Query rewrite, search overview and paper explanations call Groq through an ordered failover chain, `GROQ_MODELS` (default `qwen/qwen3.8-27b`, `openai/gpt-oss-20b`, `openai/gpt-oss-120b`). Retired models are benched for 1 h and rate-limited ones for their Retry-After. Truncated output is trimmed or discarded, and every failure degrades to no LLM output. `/healthz/deep` `services.llm` reports availability, and the keepalive fails once the primary is gone. Llama 3.3 70B, the previous single model, was shut down by Groq on 2026-08-16, and the three features were off until this change (deployed 2026-10-04) | `app/groq_svc.py`, `app/config.py`, `scripts/eval_groq_models.py` |
| AI usage | Every model call is one `llm_calls` row (replicated), for all five features (`search_rewrite`, `search_overview`, `explain`, `briefing`, `ask`): tokens in and out from Groq's usage, the answering model and its place in `GROQ_MODELS` (0 = primary; higher = a fallback answered), whether the reply hit the token cap, and the requests left today and tokens left this minute from Groq's headers. `ai_daily` (replicated) keeps per-day counts of cache hits and misses, outcomes (declined, busy, unanswerable...), points and quotes dropped by the checks, and reader reports. `/healthz/ai` shows today per feature and model, the live windows, the briefing budget and 14 days of history; counts only. `/admin/ai` is the same data as a dashboard for the owner (preview token only, 404 otherwise): today's totals, what stands out, model meters, a per-feature table and 14 days of tokens by feature, with a table view. Groq's limits per model (30 requests and 8,000 tokens a minute, 1,000 requests and 200,000 tokens a day) are `GROQ_REQUESTS_PER_DAY`, `GROQ_TOKENS_PER_DAY`, `GROQ_TOKENS_PER_MINUTE`; `LLM_DAILY_CALLS` covers only the briefing and Ask | `app/ai_usage.py`, `app/groq_svc.py`, `app/llm.py`, `app/routers/admin.py` |
| Missing encodings | Available FTS query forms still run; actual contributing arms determine `retrieval_mode` | `app/hybrid_search_svc.py` |
| Search total miss | arXiv keyword fallback; no invented hits for unfamiliar concepts | `app/routers/search.py` |
| Detailed interests | Saved papers -> long/short/negative EWMA. 2–4 saves: one interest per save; 5+: Ward medoids, K ≤ min(7, ceil(N/2)), finest singleton-free cut (2026-09-30) | `app/recommend/profiles.py`, `clustering.py` |
| Multi-interest balance | Importance-weighted quota, within-cluster MMR, quota-bound served order | `fusion.py`, `app/routers/recommendations.py` |
| Scoring | Personalized heuristic default; optional citation-trained LightGBM | `reranker.py`, `config.py` |
| Cold start | Balanced per ticked interest; citations per month over 24 months, every third slot from the corpus's last 3 months (needs refreshed citations to rank those); epsilon-greedy fresh block and deterministic recycled block | `app/discovery_svc.py`, `local_meta.py`, recommendation router |
| Exploration | 2 per page; from ticked interests no medoid covers, else the far half of own candidates | recommendation router |
| Refresh | Personalized candidate retrieval excludes last seven days of served papers; one exhaustion retry allows labelled repeats | recommendation router, `app/db.py` |
| Paging | Cached ordered pool per query ID, bound to user; per-user lock includes serving/impression write in one process | recommendation router |
| Preferences | Editable `/interests`; zero to eight known categories; no deletion of existing saves | `app/routers/onboarding.py` |
| First seeds | Suggestions from selected categories; explicit search/save; count advances on successful response | onboarding router and seed templates |
| Reading history | Latest 50 distinct opens or served papers; metadata failure retains a navigable ID | `app/routers/saved.py`, `app/db.py` |
| Feedback semantics | Latest save/unsave/dismissal sets explicit state; click/view never overrides it; unsave survives cache reload | `get_current_feedback`, `get_save_history`, `user_state.py` |
| Library | All current saves; the 20-example retrieval deque no longer limits the library view | saved router |
| Attribution | Ranked visits are clicks with origin fields; direct visits are views without fabricated policy/propensity. Card links carry their surface (`sf=search`, `recommendation`, `saved`) since the card-link fix (branch `fix/click-surface-baseline`); click rows before its deploy say `recommendation` whatever page they came from. Paper-page timings and explanation cache hits/misses appear in `/healthz/runtime`; `scripts/baseline_metrics.py` turns an interactions export into weekly reading metrics | `app/routers/paper.py`, `partials/paper_card.html`, `scripts/baseline_metrics.py` |
| Persistence | Local SQLite; periodic Turso replication of four core user tables plus the redesign's paper caches and logs (`paper_explanations`, `paper_abstracts`, `paper_sources`, `paper_briefings`, `paper_answers`, `llm_calls`, `content_reports`, `ai_daily`), which use an AUTOINCREMENT `id` watermark and restore in pages of 2,000 | `app/turso_sync.py:TABLES` |
| Full abstracts | Behind `FULL_ABSTRACTS` (off by default). A paper whose stored abstract is cut at 500 characters is queued on first open; the arXiv API is called with up to 20 ids at least 3.1 s apart, honouring Retry-After, up to `FULL_ABSTRACTS_DAILY_CAP` ids a day. Saved abstracts replace stumps on the paper page and in `/explain`; queue state is in `/healthz/runtime` | `app/abstract_svc.py`, `app/routers/paper.py` |
| Paper sources | Behind `PAPER_SOURCES` (off, shadow, on). Opening a paper schedules one background fetch of `arxiv.org/html/<id>`; the page is parsed in a thread (standard-library parser) into licence, version, sections, paragraphs and figures. `paper_sources` (replicated, at most 24 figures) and local `paper_fulltext`/`paper_paragraphs` (pruned to `FULLTEXT_MAX_PAPERS`). Pacer: 1 request per 2 s, bursts of 3, for fetches a reader triggers; background work (the warm-up) also waits 15 s between its own requests, arXiv's robots.txt `Crawl-delay`. 3 soft blocks in 10 min open a 15-min breaker. Figures show for every licence unless `FIGURE_LICENCES` lists codes. `/healthz/sources` reports coverage | `app/paper_source_svc.py`, `app/arxiv_html.py`, `app/licences.py` |
| Figure thumbnails | Behind `FIGURES`. `/fig/<id>/<n>-<hash>.webp` serves a saved picture (PNG, JPEG, GIF) as a WebP at most 800 px wide, made with Pillow in a thread; JPEGs decode at reduced scale, so print-resolution exports work. `/fig/<id>/<n>-<hash>.svg` serves a vector figure, an SVG file or a drawing inline in the arXiv page (one page fetch per paper), rebuilt from drawing elements only by `app/svg_clean.py` and served with `Content-Security-Policy: default-src 'none'; …; sandbox`. Up to 24 figures per paper. Everything is fetched once from arXiv through the paper-source pacer and cached in `FIG_CACHE_DIR` (wiped on restart, swept to `FIG_CACHE_MAX_MB`). Rate limit 600 a minute per IP. A figure that can't be made is hidden by `app.js`. Every figure shows a credit and licence line | `app/routers/figures.py`, `app/thumbnails.py`, `app/svg_clean.py` |
| Paper page v2 | Behind `PAPER_PAGE_V2` (off, preview, on); `paper.html` serves when off. Outline and links, reading column (summary box with In plain terms / Abstract tabs, key figure, all figures, nearby papers), Ask slot. Reads only saved data; the plain summary (`/explain?v=2`), Code chip (`/api/papers/<id>/code`, Hugging Face, cached weekly) and briefing load after paint. KaTeX 0.18.11 vendored under `/static/vendor/`. Static URLs carry `?v=<content hash>`. Browser tests: `tests/test_e2e_paper.py` on the world from `scripts/seed_e2e.py` | `app/templates/paper_v2.html`, `app/static/paper.js`, `app/routers/paper.py` |
| Briefing | Behind `BRIEFING` (off, shadow, preview, on). `/api/papers/<id>/briefing` returns three cited points, steps and a use line written by the `GROQ_MODELS` chain from up to 16,000 characters of the paper's passages; points citing an unsent passage or holding a number not in it are dropped. Cached in `paper_briefings` per version, parser, prompt and model. One call at a time, 8 s apart; readers wait at most 12 s. Model calls are logged to `llm_calls` with daily caps. Report a problem goes to `content_reports` | `app/briefing_svc.py`, `app/llm.py`, `scripts/eval_briefings.py` |
| Ask this paper | Behind `ASK` (off, preview, on). `POST /api/papers/<id>/ask` (cookie required, 500-character questions) picks passages without a model (abstract, named figures and citing paragraphs, BM25 top 8), asks the `GROQ_MODELS` chain for JSON, keeps only verbatim quotes and sent cites, and replaces unsupported answers. Suggested answers cached in `paper_answers`; `ASK_DAILY_CAP` per reader (`ask_usage`); counts-only `ask_log` kept 30 days; no `interactions` writes | `app/ask_svc.py`, `partials/paper/ask_panel.html`, `scripts/eval_ask.py` |
| Feed card v2 | Behind `FEED_CARD_V2` (off, preview, on); the feed only, every other surface keeps `paper_card.html`. At most eight things to click: menu (Why this paper, 3D, arXiv, copy ID), title, Code chip, first figure, In plain terms / Abstract tabs, Save, Pass. Title link and Save/Pass values identical to today's card (§3.11). `card_data.enrich()` adds figure, code link and a cached plain summary from local tables only, after exposures are logged, and never changes order, ids or instrumentation; a summary shows only when cached for the abstract on the card | `partials/paper_card_v2.html`, `app/card_data.py`, `app/routers/recommendations.py` |
| Wide feed | Behind `FEED_WIDE` (off, preview, on). From 900 px a rail (issue line, interest filter with counts, Edit interests, Recent papers, which text cards open on); from 1180 px two columns. The feed stays one list in served order: a grid of 4 px rows, each card spanning rows to its height (ResizeObserver), `grid-auto-flow: row`, never dense. The filter hides cards client-side and never refetches. Browser tests: `tests/test_e2e_feed_v2.py` | `app/templates/index.html`, `app/static/app.js`, `app/static/styles.css` |
| Layout logging | `interactions.layout` is `list` or `grid2` (anything else stored as NULL) on saves, passes and clicks from cards (`lay=` on title links); replicated, with the remote column added by `REMOTE_COLUMNS` in `ensure_remote_schema`. Feed-control events (`filter_on`, `filter_off`, `cardtext`) go to `ui_events`, which is local and not replicated | `app/db.py`, `app/turso_sync.py`, `app/routers/paper.py` |
| Figure warm-up | Behind `FIGURE_PREWARM` (off, on). Reads papers and prepares their first figure before anyone opens them, from two sources: papers people were shown (every feed page and the top 5 search results call `figure_prewarm.enqueue()`; queue of 300, oldest dropped) and, every 6 hours, papers Turso dates in the last 2 days (batches of 300). Every request takes the background lane, one per 15 s at most; 800 papers a day; stops while the breaker is open; one runner per container (file lock). Queue, reads today and the cap in `/healthz/sources` | `app/figure_prewarm.py` |
| Search thumbnails | Behind `SEARCH_FIGURES` (off, preview, on). A small figure (7.5rem; 4.75rem on a phone) beside a search result whose paper has one saved, via `card_data.enrich()` (local reads only). The classic card's markup is unchanged without a figure. The thumbnail repeats the title's §3.11 link with `#sec-fig`, out of the tab order. Browser tests: `tests/test_e2e_search.py` | `partials/paper_card.html`, `app/routers/search.py` |
| Feature switches | Redesign features read `off`, `shadow`, `preview` or `on` from Space variables; `preview` shows only to a browser that visited `?preview=<PREVIEW_TOKEN>` | `app/features.py` |
| Corpus freshness | 1,853,777 papers through 2026-09-29 (2026-10-01). New papers go to `arxiv_recent` + Turso via `scripts/ingest_arxiv.py` (`--source oai`, arXiv's OAI-PMH feed, for the daily job; `--source search` for backfills), and the OAI path writes each paper's licence to Turso `paper_licences`. The sidecar catches up at the next publish + deploy. A daily job exists (`scripts/daily_refresh.py`, `.github/workflows/daily-refresh.yml`) but is off until `DAILY_REFRESH=on` | PHASE7 §8 |
| Citations | Semantic Scholar counts for the whole corpus, refreshed 2026-10-01 (99.47% found); the daily job refreshes the last three months | `scripts/refresh_citations.py` |

## Constants worth protecting

- EWMA alpha: long 0.03, short 0.40, negative 0.15.
- Ward: normalized embeddings, Euclidean distance, adaptive clusters, maximum 7.
- Recommendation fusion: quota, never cross-interest RRF; per-cluster MMR lambda 0.6.
- Search fusion: RRF k=60; cross-encoder depth 10 by default.
- Feed: 10 core cards, up to 2 exploration cards; ranked pool 60.
- Refresh suppression: 7 days for personalized retrieval. Cold start remembers the
  available local impression history and recycles only when fresh supply is thin.
- Policy version: `v11_rescaled_pubdate` (from the 2026-10-16 follow-up deploy;
  `v10_fresh_discovery` before); search `search_v2_independent_lexical`.

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
