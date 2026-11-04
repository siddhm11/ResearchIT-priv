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

## Ratings (judgement, with evidence in the artifact)

| Area | Score | Main reasons |
|---|---|---|
| Search | 6.5 | MRR@10 0.563 on 847 known-item queries (0.618 with full abstracts); precise on technical queries, weak on landmark papers; 10 results, no paging |
| Recommendations | 5.5 | Strong design; F2, F4, F5, F6, F7; 68.4% top-10 (simulated), AUC 0.703 (real votes); no online metrics |
| Query rewriter | 5.5 | 45/54 queries go to the LLM; 6/26 stored rewrites exceed the 8-word cap; never measured on vs off |
| Ask this paper | 7 | 20/20 off-topic refused, 0.72 s median; some "limitations" answers about prior work; no human ratings |
| Plain terms / briefings | 7 | Good prompts and checks; mostly fed 500-char stumps; feed cards only show cached summaries |
| UI/UX | 7.5 | Consistent and accessible basics; feed error state broken (F3) |
| Code health | 5.5 | 580-line functions, ~70 one-shot DB connections, 23+ switches, 3 LLM layers, dead code |
| Production readiness | 5 | One process on 2 vCPU, /tmp DB with 60 s backup, spoofable rate limit, cross-reader cache leak |
| Measurement & logging | 4 | Feed exposures wiped each deploy; no feed dashboard; no scheduled quality check |

## Findings

| ID | Sev | Finding | Where | Change | Verify |
|---|---|---|---|---|---|
| F1 | High | Search/collections mutate dicts owned by the shared metadata cache; other readers see `saved`, `query_id`, `position`, `trending`, `named_paper` (related-papers cards) | `app/turso_svc.py:104-132`, `app/routers/search.py:106-120`, `app/routers/collections.py:49-51`, read in `app/routers/paper.py:107` | Return copies from `fetch_metadata_batch`; copy in routes | Test: mutate result, refetch, unchanged. Repro printed `{'saved': True, 'query_id': 'reader-A-qid'}` |
| F2 | High | 3 Passes in one primary arXiv code in 14 days hard-filter the whole code from Tier 1, ignoring saves | `app/db.py:850-878`, `app/routers/recommendations.py:1112-1124` | **Decision needed.** Proposed: passes ≥ 3 and pass share ≥ 0.8 in that code, never for a medoid's code, soft penalty | Repro: 20 cs.LG saves + 3 passes → `{'cs.LG'}` |
| F3 | High | Feed-unavailable fragment returned with 503; htmx doesn't swap it → skeleton forever + generic toast | `recommendations.py:289-296`, `app/static/app.js:280` | Return 200 (+ `X-Feed-State` header) or allow 503 swap for `#rec-section` | Run with `QDRANT_URL=` and 2 saves |
| F4 | Med | Raw cluster importances mixed with the short-term supplement's share → supplement gets 2/60 slots instead of ~10 | `recommendations.py:1171-1178`, `app/recommend/fusion.py:212-220` | Normalise importances to (1 − share) first | Repro: `[39, 19, 2]` vs `[33, 17, 10]` |
| F5 | Med | Within each interest, relevance = similarity to the global long-term profile; minority interests filled with papers leaning to the dominant one | `app/recommend/reranker.py` `heuristic_score` | **Experiment:** score on cosine to own medoid (feature 0) or blend; measure with multi-interest personas first | Simulation: bridge papers took 10/10 of B's top 10 |
| F6 | Med | Unmatched new cluster takes smallest free id = the rejected old cluster's id | `app/recommend/clustering.py` `stabilize_cluster_ids` | Start new ids above max old id | Repro: vision got RL's id 1 (cos −0.03) |
| F7 | Med | Seen filtering after a fixed `limit×2` fetch; small interests run dry silently | `app/qdrant_svc.py:608-630` | Server-side `must_not` MatchAny on `arxiv_id`, or `limit + |seen|`; count empty clusters | Unit test with saturated seen set |
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

Days 1–3: F1, F3, F4, F6, F8, F9, F10, F12, F13, F23. Week 1: baselines (847 queries, rewrite on/off,
personas, rate 130 AI outputs), `/admin/feed`, nightly search check, uptime monitor. Week 2: F17, F15,
F16, F2, F5, F7. Week 3: P1–P4. Week 4: X and S items. Then F18, F21, F19, P5.
