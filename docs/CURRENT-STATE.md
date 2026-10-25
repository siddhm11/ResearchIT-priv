# Current technical contract

Updated: 2026-10-25. The Hugging Face Space runs `ff196f8` (deployed 2026-10-24/25 IST): the redesign
stack through `feat/keyword-relax`, trending v2 (`a572e3d`, PR #57) and the sidecar `0d1449d` (PR #58).
GitHub `main` holds rollout days 1–6 plus the citation history (#53) and the Hugging Face list history
(#56); the rest of the stack merges on 25–28 Oct, then #57 and #58. The 2026-09-28 live checks are in
DISCOVERY-PLAN.md (deployment record).
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
| Search | Original + optional rewritten query (questions of any length reach the rewriter since the 2026-10-23 fix; with `SEARCH_REWRITE_KEYWORDS` on, a rewritten question whose own words match no paper in full drops its keyword list from RRF); dense shard fanout and independent FTS5; Zilliz without FTS; RRF; trending step-up (below); optional mxbai-rerank-xsmall cross-encoder (top 10; branch, not deployed); title/citation boost | `app/hybrid_search_svc.py`, `app/config.py` |
| LLM features | Query rewrite, search overview and paper explanations call Groq through an ordered failover chain, `GROQ_MODELS` (default `qwen/qwen3.8-27b`, `openai/gpt-oss-20b`, `openai/gpt-oss-120b`). Retired models are benched for 1 h and rate-limited ones for their Retry-After. Truncated output is trimmed or discarded, and every failure degrades to no LLM output. `/healthz/deep` `services.llm` reports availability, and the keepalive fails once the primary is gone. Llama 3.3 70B, the previous single model, was shut down by Groq on 2026-08-16, and the three features were off until this change (deployed 2026-10-04) | `app/groq_svc.py`, `app/config.py`, `scripts/eval_groq_models.py` |
| AI usage | Every model call is one `llm_calls` row (replicated), for all five features (`search_rewrite`, `search_overview`, `explain`, `briefing`, `ask`): tokens in and out from Groq's usage, the answering model and its place in `GROQ_MODELS` (0 = primary; higher = a fallback answered), whether the reply hit the token cap, and the requests left today and tokens left this minute from Groq's headers. `ai_daily` (replicated) keeps per-day counts of cache hits and misses, outcomes (declined, busy, unanswerable...), points and quotes dropped by the checks, and reader reports. `/healthz/ai` shows today per feature and model, the live windows, the briefing budget and 14 days of history; counts only. `/admin/ai` is the same data as a dashboard for the owner (preview token only, 404 otherwise): today's totals, what stands out, model meters, a per-feature table and 14 days of tokens by feature, with a table view. Groq's limits per model (30 requests and 8,000 tokens a minute, 1,000 requests and 200,000 tokens a day) are `GROQ_REQUESTS_PER_DAY`, `GROQ_TOKENS_PER_DAY`, `GROQ_TOKENS_PER_MINUTE`; `LLM_DAILY_CALLS` covers only the briefing and Ask | `app/ai_usage.py`, `app/groq_svc.py`, `app/llm.py`, `app/routers/admin.py` |
| Reader and search analytics | `search_log` (replicated): one row per search with the query (collapsed, at most 200 characters), its `query_id` (which result clicks and saves carry), result count, retrieval mode, the AI rewrite, time, and an owner flag; no reader is stored. `search_overviews` (replicated) keeps the AI overview per query and top five papers and is the overview cache (7 days). `reader_days` (replicated): one row per reader cookie per UTC day counting pages, searches, feed pages, papers opened, saves, passes and questions, owner = 1 for a browser holding the preview token, counted from every request by `analytics.observe()` in the HTTP middleware. Query text, rewrites, overviews and search subjects in `llm_calls` are cleared after 90 days locally and in Turso. Dashboards `/admin/readers`, `/admin/searches`, `/admin/ai/history` (owner only, own use left out unless `?me=1`) | `app/analytics.py`, `app/routers/admin.py` |
| Missing encodings | Available FTS query forms still run; actual contributing arms determine `retrieval_mode` | `app/hybrid_search_svc.py` |
| Search total miss | arXiv keyword fallback; no invented hits for unfamiliar concepts | `app/routers/search.py` |
| Search results cache | In memory only (`app/search_cache.py`): a query's ranked results are reused for `SEARCH_CACHE_TTL_S` (600 s; 120 s when a retrieval arm was missing), keyed by the query with case and spacing ignored, at most 256 queries. Going back to a search, the paper page's back link, a reload or another reader's identical search costs no encode, Groq rewrite or retrieval. A reader coming back to the same results keeps their query_id and is not logged or counted again; that link is memory-only. `/healthz/runtime` counts `search_cache_hit` / `search_cache_miss`. Responses carry `Vary: HX-Request` and htmx fragments `Cache-Control: no-store`, so Back never shows a bare results fragment | `app/search_cache.py`, `app/routers/search.py`, `app/main.py` |
| Detailed interests | Saved papers -> long/short/negative EWMA. 2–4 saves: one interest per save; 5+: Ward medoids, K ≤ min(7, ceil(N/2)), finest singleton-free cut (2026-09-30) | `app/recommend/profiles.py`, `clustering.py` |
| Multi-interest balance | Importance-weighted quota, within-cluster MMR, quota-bound served order | `fusion.py`, `app/routers/recommendations.py` |
| Scoring | Personalized heuristic default; optional citation-trained LightGBM | `reranker.py`, `config.py` |
| Cold start | Balanced per ticked interest; citations per month over 24 months, every third slot from the corpus's last 3 months (needs refreshed citations to rank those); epsilon-greedy fresh block and deterministic recycled block | `app/discovery_svc.py`, `local_meta.py`, recommendation router |
| Exploration | 2 per page; from ticked interests no medoid covers, else the far half of own candidates | recommendation router |
| Refresh | Personalized candidate retrieval excludes last seven days of served papers; one exhaustion retry allows labelled repeats | recommendation router, `app/db.py` |
| Paging | Cached ordered pool per query ID, bound to user; per-user lock includes serving/impression write in one process | recommendation router |
| Preferences | Editable `/interests`; zero to eight known categories; no deletion of existing saves | `app/routers/onboarding.py` |
| First seeds | Suggestions from selected categories; explicit search/save; count advances on successful response | onboarding router and seed templates |
| Described interest | Onboarding step 1 takes a sentence (≤300 chars): searched without the LLM rewrite and not stored; areas its top papers sit in (primary category 1, cross-list ½) are picked for the reader to check, and step 2 opens on its papers. Every area card on `/onboarding` and `/interests` has a drawing (`partials/area_art.html`, written by `scripts/build_area_art.py`) | `app/routers/onboarding.py`, `discovery_svc.areas_for_papers` |
| Reading history | Latest 50 distinct opens or served papers; metadata failure retains a navigable ID | `app/routers/saved.py`, `app/db.py` |
| Feedback semantics | Latest save/unsave/dismissal sets explicit state; click/view never overrides it; unsave survives cache reload | `get_current_feedback`, `get_save_history`, `user_state.py` |
| Library | All current saves; the 20-example retrieval deque no longer limits the library view. With `TRENDING_PAGE` visible, Library has its own tabs (Saved · Collections · Recent) and Trending takes Collections' place in the top and bottom bars; with it off the bars keep Collections (doc 06, 2026-10-25) | saved router, `partials/library_tabs.html`, `base.html` |
| Attribution | Ranked visits are clicks with origin fields; direct visits are views without fabricated policy/propensity. Card links carry their surface (`sf=search`, `recommendation`, `saved`) since the card-link fix (branch `fix/click-surface-baseline`); click rows before its deploy say `recommendation` whatever page they came from. Paper-page timings and explanation cache hits/misses appear in `/healthz/runtime`; `scripts/baseline_metrics.py` turns an interactions export into weekly reading metrics | `app/routers/paper.py`, `partials/paper_card.html`, `scripts/baseline_metrics.py` |
| Persistence | Local SQLite; periodic Turso replication of four core user tables plus the redesign's paper caches and logs (`paper_explanations`, `paper_abstracts`, `paper_sources`, `paper_briefings`, `paper_answers`, `llm_calls`, `content_reports`, `ai_daily`, `search_log`, `search_overviews`, `reader_days`), which use an AUTOINCREMENT `id` watermark and restore in pages of 2,000. Not replicated, so reset on every deploy: feed impressions and exposures, collection follows, UI events, cluster snapshots, Ask's log and daily counter, paper full text and paragraphs (fetched again when needed), caches and thumbnails | `app/turso_sync.py:TABLES` |
| Full abstracts | Behind `FULL_ABSTRACTS` (off by default). A paper whose stored abstract is cut at 500 characters is queued on first open; the arXiv API is called with up to 20 ids at least 3.1 s apart, honouring Retry-After, up to `FULL_ABSTRACTS_DAILY_CAP` ids a day. Saved abstracts replace stumps on the paper page and in `/explain`; queue state is in `/healthz/runtime` | `app/abstract_svc.py`, `app/routers/paper.py` |
| Paper sources | Behind `PAPER_SOURCES` (off, shadow, on). Opening a paper schedules one background fetch of `arxiv.org/html/<id>`; the page is parsed in a thread (standard-library parser) into licence, version, sections, paragraphs and figures. `paper_sources` (replicated, at most 24 figures) and local `paper_fulltext`/`paper_paragraphs` (pruned to `FULLTEXT_MAX_PAPERS`). A deploy or prune drops the local text while the row stays; `has_text()` tells, and `ensure(need_text=True)` (opening a paper, the briefing, Ask) fetches it again. Pacer: 1 request per 2 s, bursts of 3, for fetches a reader triggers; background work (the warm-up) also waits 15 s between its own requests, arXiv's robots.txt `Crawl-delay`. 3 soft blocks in 10 min open a 15-min breaker. Figures show for every licence unless `FIGURE_LICENCES` lists codes. `/healthz/sources` reports coverage | `app/paper_source_svc.py`, `app/arxiv_html.py`, `app/licences.py` |
| Figure thumbnails | Behind `FIGURES`. `/fig/<id>/<n>-<hash>.webp` serves a saved picture (PNG, JPEG, GIF) as a WebP at most 800 px wide, made with Pillow in a thread; JPEGs decode at reduced scale, so print-resolution exports work. `/fig/<id>/<n>-<hash>.svg` serves a vector figure, an SVG file or a drawing inline in the arXiv page (one page fetch per paper), rebuilt from drawing elements only by `app/svg_clean.py` and served with `Content-Security-Policy: default-src 'none'; …; sandbox`. Up to 24 figures per paper. Everything is fetched once from arXiv through the paper-source pacer and cached in `FIG_CACHE_DIR` (wiped on restart, swept to `FIG_CACHE_MAX_MB`). Rate limit 600 a minute per IP. A figure that can't be made is hidden by `app.js`. Every figure shows a credit and licence line | `app/routers/figures.py`, `app/thumbnails.py`, `app/svg_clean.py` |
| Paper page v2 | Behind `PAPER_PAGE_V2` (off, preview, on); `paper.html` serves when off. Outline and links, reading column (summary box with In plain terms / Abstract tabs, key figure, all figures, nearby papers), Ask slot. Reads only saved data; the plain summary (`/explain?v=2`), Code chip (`/api/papers/<id>/code`, Hugging Face, cached weekly) and briefing load after paint. KaTeX 0.18.11 vendored under `/static/vendor/`. Static URLs carry `?v=<content hash>`. The back link returns to the page the reader came from (same-origin Referer: a search keeps its query; feed, saved, recently opened, a collection, the previous paper), else to the click's surface; a shared link gets none. Browser tests: `tests/test_e2e_paper.py` and `tests/test_e2e_search.py` on the world from `scripts/seed_e2e.py` | `app/templates/paper_v2.html`, `app/static/paper.js`, `app/routers/paper.py` |
| Briefing | Behind `BRIEFING` (off, shadow, preview, on). `/api/papers/<id>/briefing` returns three cited points, steps and a use line written by the `GROQ_MODELS` chain from up to 16,000 characters of the paper's passages, picked by place in the paper, not section title (the abstract, the introduction and conclusion, the opening of every section and subsection, captions, then further paragraphs; never related work or front matter) and sent in page order; prompt b2 declines only when the passages don't say what the paper does; points citing an unsent passage or holding a number not in it are dropped. Cached in `paper_briefings` per paper version and prompt; a good briefing is never replaced by a worse one, and rows stored under the older key (which also held parser and model) are still found. When this container lacks a saved paper's text (it is local only, so a deploy drops it), the page is fetched again before writing; if that fails nothing is stored. One call at a time, 8 s apart; readers wait at most 12 s. Model calls are logged to `llm_calls` with daily caps. Report a problem goes to `content_reports` | `app/briefing_svc.py`, `app/llm.py`, `scripts/eval_briefings.py` |
| Ask this paper | Behind `ASK` (off, preview, on). `POST /api/papers/<id>/ask` (cookie required, 500-character questions) picks passages without a model (abstract, named figures and citing paragraphs, BM25 top 8), asks the `GROQ_MODELS` chain for JSON, keeps only verbatim quotes and sent cites, and replaces unsupported answers. Suggested answers cached in `paper_answers`, except one written from the abstract alone while the paper's text was missing in this container (the text is fetched again first); `ASK_DAILY_CAP` per reader (`ask_usage`); counts-only `ask_log` kept 30 days; no `interactions` writes | `app/ask_svc.py`, `partials/paper/ask_panel.html`, `scripts/eval_ask.py` |
| Feed card v2 | Behind `FEED_CARD_V2` (off, preview, on); the feed only, every other surface keeps `paper_card.html`. At most eight things to click: menu (Why this paper, 3D, arXiv, copy ID), title, Code chip, first figure, In plain terms / Abstract tabs, Save, Pass. Title link and Save/Pass values identical to today's card (§3.11). `card_data.enrich()` adds figure, code link and a cached plain summary from local tables only, after exposures are logged, and never changes order, ids or instrumentation; a summary shows only when cached for the abstract on the card | `partials/paper_card_v2.html`, `app/card_data.py`, `app/routers/recommendations.py` |
| Wide feed | Behind `FEED_WIDE` (off, preview, on). From 900 px a rail (issue line, interest filter with counts, Edit interests, Recent papers, which text cards open on); from 1180 px two columns. The feed stays one list in served order: a grid of 4 px rows, each card spanning rows to its height (ResizeObserver), `grid-auto-flow: row`, never dense. The filter hides cards client-side and never refetches. Browser tests: `tests/test_e2e_feed_v2.py` | `app/templates/index.html`, `app/static/app.js`, `app/static/styles.css` |
| Layout logging | `interactions.layout` is `list` or `grid2` (anything else stored as NULL) on saves, passes and clicks from cards (`lay=` on title links); replicated, with the remote column added by `REMOTE_COLUMNS` in `ensure_remote_schema`. Feed-control events (`filter_on`, `filter_off`, `cardtext`) go to `ui_events`, which is local and not replicated | `app/db.py`, `app/turso_sync.py`, `app/routers/paper.py` |
| Figure warm-up | Behind `FIGURE_PREWARM` (off, on). Reads papers and prepares their first figure before anyone opens them, from two sources: papers people were shown (every feed page and the top 5 search results call `figure_prewarm.enqueue()`; queue of 300, oldest dropped) and, every 6 hours, papers Turso dates in the last 2 days (batches of 300). Every request takes the background lane, one per 15 s at most; 800 papers a day; stops while the breaker is open; one runner per container (file lock). Queue, reads today and the cap in `/healthz/sources` | `app/figure_prewarm.py` |
| Search thumbnails | Behind `SEARCH_FIGURES` (off, preview, on). A small figure (7.5rem; 4.75rem on a phone) beside a search result whose paper has one saved, via `card_data.enrich()` (local reads only). The classic card's markup is unchanged without a figure. The thumbnail repeats the title's §3.11 link with `#sec-fig`, out of the tab order. Browser tests: `tests/test_e2e_search.py` | `partials/paper_card.html`, `app/routers/search.py` |
| Feature switches | Redesign features read `off`, `shadow`, `preview` or `on` from Space variables; `preview` shows only to a browser that visited `?preview=<PREVIEW_TOKEN>` | `app/features.py` |
| Corpus freshness | Turso `papers` holds 1,972,571 papers and `arxiv_recent` 375,474 points (2026-10-25), and the sidecar baked into the image (`0d1449d`) holds the same 1,972,571, newest 2026-10-23. Measured against arXiv's full metadata listing on 2026-10-24, every paper in our 43 categories from June 2025 on is present except those after the listing's 17 Oct cut, which the daily job covers. 91,876 of them were added that day: the 2025-26 backfill had asked arXiv for each category in one oldest-first query and lost everything past its 10,000-result cap. They were encoded on Kaggle (no credentials there; rows whose `transformers` 5.x tokens differ from 4.44.2 re-encoded with the ingest's own encoder) and written with the ingest's `Upserter`. Papers before June 2025 come from the original snapshot, which is complete for computer science and machine learning but holds 63% of our 43 categories overall (700,939 physics and maths papers missing; Fix Log item 15). New papers arrive daily through `scripts/daily_refresh.py` and `.github/workflows/daily-refresh.yml` (`DAILY_REFRESH=on`), which runs `scripts/ingest_arxiv.py --source oai` and writes `arxiv_recent`, Turso and `paper_licences`. Vector search, feeds and paper pages see new papers at once; keyword search and the starter feed only after the next sidecar publish + deploy | PHASE7 §8, `scripts/ingest_backends.py` |
| Citations | Semantic Scholar counts for the whole corpus, refreshed 2026-10-01 (99.47% found), and for every paper since June 2025 on 2026-10-25 (373,718 of 375,473 found, 94,839 changed, in Turso and the sidecar). The daily job refreshes the last three months and, since #53, keeps each change in Turso `citation_history` (arxiv_id, checked_on, before and after), one all-or-nothing Hrana batch per 500 papers. Semantic Scholar throttled all callers on 2026-10-24 (HTTP 429 with or without the key) and the citation step failed both runs that day | `scripts/refresh_citations.py`, `scripts/daily_refresh.py` |
| Hugging Face list history | Since #56 the daily job's first stage records the last 14 Daily Papers lists, every page of them, into Turso `hf_daily_papers` (one row per paper: list date, title, Hugging Face keywords and summary, repo, project page) and `hf_upvote_history` (per paper per day: upvotes, comments, stars, place on its day). It never fails the run. 26 weeks were loaded on 2026-10-24 (5,203 papers, 130 lists) | `scripts/daily_refresh.py` `stage_hf_daily` |
| Trending shelf | Behind `TRENDING_SHELF` (on). Up to 6 Daily Papers of the last 7 days after the feed's fourth card (above the first card while the reader has fewer than 5 saves, titled for their areas), on every tier, chosen by closeness to the reader's interests or, failing that, by being listed in an area they picked; places go round the interests and areas, papers shown before come after new ones, and a reader nothing fits gets the week's most upvoted under "Trending this week", so the shelf shows at weekends and without saves (doc 06, 2026-10-23 and 2026-10-25). /trending adds a row per chosen area: its count of the week's papers and its 3 most upvoted. Each card says why in counts from the lists ("Most upvoted of 92 papers on 30 Sep · top 1% of this week's 404"), shows Hugging Face's one-line summary labelled as theirs, and, with `TRENDING_PAGE`, the rising topic it belongs to; comment counts are never shown | `app/trending_shelf.py`, `app/templates/partials/trending_card.html` |
| Rising topics and /trending | Behind `TRENDING_PAGE` (on since 2026-10-24). Once a day the stored 26 weeks of lists are clustered like a reader's saves (Ward on vectors from Qdrant, K = 24, Hungarian-kept ids); a topic shows only when its weekly-share slope beats 2,000 week shuffles (Benjamini–Hochberg q < 0.05) and under a phrase 30% of its papers carry. On 2026-10-24: 5,176 papers, Skills, Robots and World Models rising, none cooling, about 30 s a day. `/trending` lists the week's papers in the reader's interests, the topics and the week's top 9; failures keep the last topics and say so. `/healthz/runtime` reports the job The tab says which lists it shows ("Lists of 19–23 Oct · next list Mon 26 Oct" at weekends), leads with chips for rising and cooling topics and the reader's areas, gives shares in plain numbers too ("1 in 37 → 1 in 18"), and lists the named topics that fail the trend test as "Steady" with their share now. Every named topic has a page, /trending/topics/<id>: its 26-week line and label, its papers on this week's lists (papers listed since the daily grouping go to their nearest topic), its 10 most upvoted, recent arXiv papers closest to its middle paper from the recent shard (not on the lists; cached 6 hours), and its nearest topics. Timed in /healthz/runtime as trending_page, trending_topic and trending_shelf | `app/trending_topics.py`, `app/routers/trending.py` |
| Search trending step-up | Behind `SEARCH_TRENDING` (on since 2026-10-24). A top-quarter Daily Paper of the week that retrieval found in its first 50 may take one of 2 extra cross-encoder places; at ≥ half the best score it gains up to 0.25 × the best, full for 3 days and gone at 7, at most 2 per search, and the result is tagged "Trending on Hugging Face". A replay of 99 searches moved no result and tagged 4 that already ranked; counter `search_trend_on` | `app/hybrid_search_svc.py` |

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
- Follows and exposure logs are not replicated, so every deploy resets them.
  Discovery history can reset on ephemeral storage replacement. Open history is in the replicated interactions
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
