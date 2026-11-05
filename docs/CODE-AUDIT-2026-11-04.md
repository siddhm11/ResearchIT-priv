# Code audit — 2026-11-04

Read of `main` at `bd649d8`. No code was changed. Visual version with charts and
screenshots: https://claude.ai/artifact/3QbwytayBzFMrzEiNStdB1 (private to the owner).

**Method.** Full read of the recommendation, search, data, sync, startup, rate-limit and
switch code; ruff + vulture; small reproduction scripts against the real functions and a
temporary SQLite DB; the real app run locally on `scripts/seed_e2e.py` and driven with
Chromium at 1280 px and 390 px; the project's own recorded evaluations (doc 06,
`reports/`). The live Space, huggingface.co and arxiv.org were blocked by the session's
network policy, so no new live numbers were measured.

**Test suite:** 1,651 passed, 1 failed (F13, network), 11 skipped (`-m "not live and not browser"`).

## Ratings (weighted rubric; see the artifact's "How each score was computed")

| Area | Score | Confidence | Main reasons |
|---|---|---|---|
| Search | 6.3 | Medium | MRR@10 0.563 (847 queries, 0.618 with full abstracts); precise on technical queries; F15 cliff at fused rank 11; F16 boost scale; 10 results, no paging |
| Recommendations | 5.4 | Medium | Strong design; F27 interest detection, F2, F4, F6 reproduced in the pipeline; 68.4% top-10 (simulated), AUC 0.703 (real votes); no online metrics |
| Query rewriter | 5.3 | Medium-low | 45/54 queries to the LLM; 6/26 rewrites over the 8-word cap; never measured on vs off |
| Ask this paper | 7.3 | Medium | 20/20 off-topic refused, 0.72 s median; ~4/19 "limitations" answers about prior work; no human ratings |
| Plain terms / briefings | 5.7 | Medium | Good prompts; 48/50 briefings ok; fed 500-char stumps; feed shows cached only; no ratings |
| UI/UX | 7.0 | Medium-high | Consistent, no overflow; F3 broken error state; F14 copy issues |
| Code health | 5.6 | High | 1,651 tests + property tests pass; 580-line functions, dead code, 23+ switches, 3 LLM layers |
| Production readiness | 4.1 | High | F1 35% cross-reader leak under concurrency, F8, F9, F10, F11, F23 reproduced |
| Measurement & logging | 3.6 | High | Feed exposures wiped each deploy; no feed report; no scheduled quality check |

## Test results (layered)

- Property tests: 3,000 random fusion/MMR cases + 400 clustering cases; all invariants held.
- Pipeline simulation (real `_build_feed`/`_build_page` on a synthetic 4,080-paper world, 3 fake shards):
  F2 page 5/3/2 → 0/6/4 after 3 cs.LG passes; F4 the single short-term paper sits at position 31,
  0 of 750 cards; F5 fix moves ~2% (downgraded to Low); F7 not reproduced in 40 visits (Low);
  F19 one-save path halves paper age (4.3 → 2.1 years) and adds 2 exploration cards.
- **F27 (new, High):** interest detection. Reader 25/4/3/3/2 → 2 interests; 30/3/3/3/3 → 2; single
  interest of 30 → 7. A silhouette-chosen cut on the same Ward tree found the true K in 11/11 shapes
  (20 worlds each). Pipeline: a 14/5/4/3/2/2 reader got robotics, biology, astro 0 of 1,400 cards.
  Validate on real saved vectors before changing (`app/recommend/clustering.py`, needs owner approval).
- F1 end-to-end: with `SEARCH_FIGURES=off`, 52/150 concurrent other readers saw a paper as Saved that
  only another reader saved; 0/150 with the switch on (enrich copies), shared cache still corrupted.
- F8: a raising `stop()` aborts shutdown before the Turso final flush. F9: 2,000 cookieless visits →
  2,000 in-memory users. F10: 360/360 searches allowed with a spoofed X-Forwarded-For (224/240
  blocked without). F11: a same-second profile write is never pushed. F15: exact-title paper at
  fused rank ≤10 → #1, ≥11 → absent. F16: CE p 0.50 + title/citations (1.41) beats CE p 0.95 (0.97).
- P1: 18 SQLite connections per first feed page (38 ms total locally). P3: 10 snapshot rows for
  2 distinct cluster sets over 5 refreshes. Fuzz: 35 hostile requests, no 5xx, no reflection.
  Local load: 60 concurrent searches p50 383 ms / p95 792 ms; 60 cold feeds p50 960 ms / p95 1.5 s.

## Findings

| ID | Sev | Finding | Where | Change | Verify |
|---|---|---|---|---|---|
| F1 | High | Search/collections mutate dicts owned by the shared metadata cache; other readers see `saved`, `query_id`, `position`, `trending`, `named_paper` (related-papers cards) | `app/turso_svc.py:104-132`, `app/routers/search.py:106-120`, `app/routers/collections.py:49-51`, read in `app/routers/paper.py:107` | Return copies from `fetch_metadata_batch`; copy in routes | Test: mutate result, refetch, unchanged. Repro printed `{'saved': True, 'query_id': 'reader-A-qid'}` |
| F2 | High | 3 Passes in one primary arXiv code in 14 days hard-filter the whole code from Tier 1, ignoring saves | `app/db.py:850-878`, `app/routers/recommendations.py:1112-1124` | **Decision needed.** Proposed: passes ≥ 3 and pass share ≥ 0.8 in that code, never for a medoid's code, soft penalty | Repro: 20 cs.LG saves + 3 passes → `{'cs.LG'}` |
| F3 | High | Feed-unavailable fragment returned with 503; htmx doesn't swap it → skeleton forever + generic toast | `recommendations.py:289-296`, `app/static/app.js:280` | Return 200 (+ `X-Feed-State` header) or allow 503 swap for `#rec-section` | Run with `QDRANT_URL=` and 2 saves |
| F4 | Med | Raw cluster importances mixed with the short-term supplement's share → supplement gets 2/60 slots instead of ~10 | `recommendations.py:1171-1178`, `app/recommend/fusion.py:212-220` | Normalise importances to (1 − share) first | Repro: `[39, 19, 2]` vs `[33, 17, 10]` |
| F5 | Low | Within each interest, relevance = similarity to the global long-term profile; minority interests filled with papers leaning to the dominant one | `app/recommend/reranker.py` `heuristic_score` | **Experiment:** score on cosine to own medoid (feature 0) or blend; measure with multi-interest personas first | Simulation: bridge papers took 10/10 of B's top 10 |
| F6 | Med | Unmatched new cluster takes smallest free id = the rejected old cluster's id | `app/recommend/clustering.py` `stabilize_cluster_ids` | Start new ids above max old id | Repro: vision got RL's id 1 (cos −0.03) |
| F7 | Low | Seen filtering after a fixed `limit×2` fetch; small interests run dry silently | `app/qdrant_svc.py:608-630` | Server-side `must_not` MatchAny on `arxiv_id`, or `limit + |seen|`; count empty clusters | Unit test with saturated seen set |
| F8 | Med | Unguarded sequential `stop()` calls can skip the final Turso flush | `app/main.py:104-125` | Flush first; guard each stop | Test with a raising stop |
| F9 | Med | Unbounded `user_state._cache`; every cookieless request adds an entry | `app/user_state.py:72-90` | LRU cap; no state for anonymous GETs | Load test with no cookies |
| F10 | Med | Rate limit keys on client-controlled left-most X-Forwarded-For; write routes unlimited and accept any `paper_id` | `app/rate_limit.py` `client_key`; `app/routers/events.py` | Use proxy-appended entry; validate arXiv ids; limit writes | `/healthz/client` from two networks |
| F11 | Low | Second-resolution text watermarks can skip rows; removed clusters never deleted in Turso | `app/turso_sync.py:565-587` | Integer change sequence or `>=` + pk dedupe; one cluster row per user | Sync test with same-second writes |
| F12 | Low | Masthead "N saved" and `user_total_saves` cap at 20; issue date in UTC | `app/main.py` home; `recommendations.py` | Count from save history; local date | — |
| F13 | Low | `tests/test_http_client.py::test_a_live_request_works_after_the_loop_is_replaced` hits the network unmarked | tests | `@pytest.mark.live` | Offline suite green |
| F14 | Low | Trending failure copy refers to "papers above" with none shown; "1.8M" hard-coded ×3; "Pipeline timings" shown to all; 18 px checkboxes; 404 has no h1 | `trending.html:86`, `search.html:25,41,59`, `search_results.html:63` | Copy fixes; read count from stats; preview-only debug | Screenshots |
| F15 | Med | Title/citation boosts and the cross-encoder only see RRF top 10; landmarks at rank 11+ can't be rescued | `app/hybrid_search_svc.py:355-376` | Pre-score fused top-50 (RRF + title + citations) to pick CE candidates | `scripts/embedding_eval/search/search_eval.py` |
| F16 | Med | After the CE, title overlap (+0.8–1.0) outweighs the CE probability (0–1) | `hybrid_search_svc.py:679-688` | Re-tune boost scale after CE (0.2–0.3×) | MRR / nDCG on 847 queries |
| F17 | High | ~78% of abstracts cut at 500 chars; caps reranker (0.563 vs 0.618) and plain terms | sidecar, `app/abstract_svc.py` | Bulk backfill from the local Kaggle CSV, rebuild sidecar | Truncated share ≈ 0; re-measure MRR |
| F18 | Med | Search returns exactly 10, no paging/filters/sort | `config.ARXIV_MAX_RESULTS` | Page from the cached fused list; year/area filters | — |
| F19 | Med | 1-save feed is raw Recommend API (no rerank/MMR/recency) | `recommendations.py:534-552` | **Decision:** `MIN_SAVES_FOR_INTERESTS = 1` | Persona eval |
| F20 | Med | Ask: ~4/19 "limitations" answers about prior work; 2/80 leak `(S5.SS1.p3)` ids; ratings.csv empty | `app/ask_svc.py`, `reports/ask/2026-10-19` | Prompt rule; strip ids; rate stored outputs | Re-run `scripts/eval_ask.py` |
| F21 | Med | Feed cards show plain terms only when already cached | `app/card_data.py` | Nightly pre-generation for new + trending papers after F17 | Coverage % on feed cards |
| F22 | Med | Rewriter: 45/54 queries to LLM incl. titles; 6/26 rewrites over the 8-word cap; never measured on/off | `app/groq_svc.py` `_looks_academic`, `rewrite` | Measure on/off; gate on conversational cues; enforce 8 words; half-weight rewrite lists | 847 + conversational set |
| F23 | High | `feed_exposures` / `feed_impressions` are local only: every deploy erases the feed metrics denominator | `app/turso_sync.py` TABLES | Replicate exposures (id watermark) or a replicated daily roll-up | Exposures present after a redeploy |
| F24 | Med | No feed dashboard, no scheduled search-quality check | `app/analytics.py`, workflows | `/admin/feed`; nightly Action running fixed queries via `/api/space/search` (hit@10, MRR, latency) | — |
| F25 | Low | Only 12-hourly keepalive; print logs vanish with the container | keepalive.yml, `app/errors.py` | 5-min uptime monitor; per-module error counters; daily digest in Turso | — |
| F27 | High | Interest detection: heavy readers' side interests merged (2 found of 4–5); single-topic readers split up to 7 | `app/recommend/clustering.py` `_adaptive_threshold` | **Decision:** silhouette-chosen cut, validated on real vectors first | Clustering sweep + pipeline (see Test results) |
| F26 | Low | Zilliz kept alive for a fallback production never uses | `app/zilliz_svc.py` | Drop service, ping and `pymilvus` (decision) | — |

## Speed

- **P1** One aiosqlite connection (new thread + open) per DB helper (~70 call sites, ~20 on a first feed page) → reuse connections.
- **P2** Batch medoid searches with `query_batch_points` (K×3 calls → 3).
- **P3** Stop per-feed writes: cluster snapshots (dedupe by `paper_ids_hash`), unchanged clusters, ~120 `paper_metadata` rows ×2 (only for the suppression JOIN).
- **P4** Per-shard deadline (~2.5 s); client timeout is 30 s and fan-out waits for the slowest shard.
- **P5** Move crawling, figure prewarm, topic clustering and HN aggregation out of the web process.
- **P6** Lazy-load LightGBM; compute only the heuristic's 6 features; drop the per-request print.
- **P7** Experiment: ONNX int8 BGE-M3 query encoder (accept only if cosine to fp32 ≥ 0.99). Also test copying weights out of mmap (RAM is mostly idle) instead of keep-warm page touching.

Recorded production timings: search warm p50 ~2.8 s (1.9–4.6 s on 28 Sep), cold after idle 10.6 s;
first feed 7.3 s then ~0.9 s; Ask median 0.72 s / p95 1.41 s. Local server code (no cloud): paper
page 15 ms, keyword search 7–14 ms, feed fragment 30 ms.

## Remove / simplify

- **X1** `qdrant_svc.multi_interest_search` + `_run_prefetch_rrf` (forbidden RRF-for-recs, no caller).
- **X2** Point-id path: `lookup_qdrant_ids`, `_scroll_by_arxiv_ids`, `_get_vectors_by_ids`, `paper_qdrant_map` + `db.get_qdrant_id/save_qdrant_id/get_qdrant_ids_batch`.
- **X3** HF discovery shadow pipeline (`app/discovery_*`, `scripts/hf_discovery.py`, its doc/tests) — decision.
- **X4** `inject_exploration`, `reranker._cosine_sim`, `_parse_authors`, `qdrant_svc._merge_scored`, `db.forget_oldest_impressions`, `db.explanation_stats`, `hybrid_search_svc._rrf_fuse` (move to scripts), `llm.BudgetExceeded`, `collections_svc.all_anchor_ids`.
- **X5** Config: `METADATA_CACHE_TTL_DAYS`, `SEARCH_SEMANTIC_WEIGHT`, `SEARCH_RECENCY_WEIGHT`, `space_app_enabled`; use or delete `COOKIE_MAX_AGE` (`365*24*3600` repeated in 10+ routes).
- **X6** 12 unused imports, 3 unused `loop =`, inline re-imports (ruff F401/F841).
- **X7** `pytest`, `pytest-asyncio` out of `requirements.txt`.
- **X8** uint8 quantise/dequantise path, if `/healthz/deep` confirms all shards float16.
- **X9** MiniLM rollback model in the image; LightGBM serving path — decision.
- **S1** Retire switches that have been `on` for 2+ weeks with their old paths (needs production values).
- **S2** One LLM client instead of `groq_svc` + `llm` + `llm_router`.
- **S3** Split `_multi_interest_recommend` (~580 lines, one try) into stages.
- **S4** One settings module (98 env reads in 14 files) + `/healthz/config`.
- **S5** Move history narratives out of comments (6,150 prose lines vs 13,700 code lines).
- **S6** Fix doc drift: README tier table, CLAUDE.md notebooks list and "no tests/fixtures", recommendations.py header.
- **S7** Redundant interleave before `enforce_quota_on_ranking`; unused `compute_clusters` n<5 branch.

## Decisions needed from the owner

1. F2 suppression rule. 2. F5 experiment go-ahead. 3. F19 one-save path. 4. X3 / X9 / F26 deletions.
5. Production switch values for S1. 6. Network access to the Space, huggingface.co and arxiv.org for a live pass.

## Suggested order

Days 1–3: F1, F3, F11, F4, F6, F8, F9, F10, F12, F13, F23. Week 1: baselines (847 queries, rewrite on/off,
personas, rate 130 AI outputs), `/admin/feed`, nightly search check, uptime monitor. Week 2: F27 validation, F17, F15,
F16, F2, F19. Week 3: P1–P4. Week 4: X and S items. Then F18, F21, F19, P5.

## Strategic recommendations (2026-11-05)

Diagnosis: the project has outbuilt its evidence. 44% of app code is the core loop (search 8%,
feed 19%, data 18%); the rest is paper-page AI and figures (19%), ops/analytics (14%),
trending/HN (9%), LLM plumbing (9%) and side projects (6%), with no measure of whether readers
return. In order of importance:

1. **One weekly number, and a freeze on side features.** North star: weekly savers. Guardrails:
   7-day return of new readers, saves per 100 feed cards, search success %, p95 time to first card.
2. **Durable, portable libraries.** SQLite on HF persistent storage (20 GB, listed at $5/month;
   https://huggingface.co/docs/hub/spaces-storage) with Turso as a nightly off-site copy; a private
   "feed key" link that restores the library on any device and serves an RSS/Atom feed; later an
   optional weekly email digest.
3. **Repair the multi-interest core and lock it with contract tests** (F1, F2, F4, F6, F27; the
   audit's simulation becomes CI tests of the promises).
4. **Search basics.** arXiv IDs, arxiv.org links and author names return nothing from the keyword
   index today (it indexes title + abstract only; tested). Add ID/URL routing, an authors column,
   paging with year/area filters and newest-first, cross-encoder auditions (F15), full abstracts (F17).
5. **Lead with the idea; a newcomer mode.** Nightly plain-language line per new paper in the daily
   Action. Onboarding choice: "Learn the field" (recency 0.15, measured 73.5% top-10 hit) vs
   "Keep up" (0.45, 68.4%, today's behaviour).
6. **Spend AI quota per paper, not per query.** 200,000 tokens/model/day; search ≈ 1,850 tokens,
   briefing ≈ 3,700 → ~300 uncached searches or ~160 briefings a day across three models. Overview on
   demand; rewriter on/off test then gate or retire; quota to per-paper lines and opened briefings.
7. **Narrow scope to AI/ML/CS** (complete coverage; partial elsewhere, labelled) and **shrink moving
   parts**: drop Zilliz, shadow pipeline, MiniLM/LightGBM; background jobs to the daily Action;
   benchmark an int8 HNSW of the static 1.6M papers in idle RAM (~2 GB) with Qdrant kept for
   recent papers only (decide on recall@60 and 2-vCPU latency).
8. **Agent process.** CLAUDE.md to ~150 lines of invariants; switch registry with CI-enforced
   expiry; invariant tests (no mutation of cached objects, metric tables durable, contract tests);
   harness numbers on every ranking PR.
9. **Discoverable paper pages** (after F9): sitemap, robots.txt, canonical URL without tracking
   params, key figure as og:image.

Stop: new surfaces before the weekly number; unproven per-query LLM spend; dead fallbacks;
history in comments. Six-week order: (1) disk + F1/F3/F8/F9/F10 + metric page + freeze;
(2) search basics; (3) multi-interest repairs + contract tests; (4) feed key/RSS, two modes,
nightly lines; (5) discoverability + 30–50 real readers; (6) review numbers, decide scope and index.
