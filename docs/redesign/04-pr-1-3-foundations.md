# PR 1–3: Foundations

Three small pull requests make the numbers honest and the data fresh before any screen changes. None of them changes what readers see, except PR 3, which shows full abstracts.

## PR 1: fix click logging, time the paper page, measure the baseline

PR 1 makes the "before" numbers trustworthy. Today every click is stored as a feed click (T1), and the paper page has no timings, so nothing later could be compared honestly.

**Build steps**

1. In `app/templates/partials/paper_card.html` line 97, change `_source` to `source`. `source` is set at the top of the card with a default, and `search_results.html` and `rec_page.html` both set it before the include.
2. In `app/routers/paper.py`, time `paper_page()` in three parts (metadata, related papers, total) and call `runtime_stats.record("paper_page", {"meta_ms": …, "related_ms": …, "total_ms": …})`. In `explain()`, call `runtime_stats.count("explain_cache_hit")` or `count("explain_generated")`. Both then appear in `/healthz/runtime`.
3. Add `scripts/baseline_metrics.py`, read-only, with two modes:
    1. `--export out.sqlite` copies `user_interactions` from Turso into a local file, using `TURSO_URL` and `TURSO_DB_TOKEN`. Only the owner runs this mode, because it reads real reader data.
    2. `--db out.sqlite --since YYYY-MM-DD` computes the metrics below per ISO week, in UTC, and writes `reports/baseline/<date>.md` and `.json`. Output holds counts and rates only, never a user id.
4. Metric definitions, written into the report so nobody has to guess later:
    1. **Readers:** distinct `user_id` with any event that week.
    2. **Session:** a reader's events with no gap over 30 minutes.
    3. **Saves per session** and **Pass share**: `not_interested` ÷ (`save` + `not_interested`).
    4. **Paper opens:** `click` plus `view` events. **Open-to-save:** sessions where paper X is opened and then saved in the same session, ÷ sessions with an open.
    5. **Seven-day return:** readers with a session on day D who have another session on days D+1 to D+7.
    6. **Feed clicks by surface:** only from the merge date of this PR on. Earlier `click` rows all say `recommendation` (T1), and the report says so in one line.
    7. **Excluded:** user ids with more than 500 events in a day, reported as a count.
5. Add `--runtime-url` to save one snapshot of the live `/healthz/runtime` (a read-only GET) into the report, so feed `build_ms` p50 and p95 are recorded.
6. Update `docs/CURRENT-STATE.md`, Attribution row: clicks before the merge date carry `source=recommendation` regardless of surface.

**Tests**

- `tests/test_paper_page.py::test_card_link_names_its_surface`: render `paper_card.html` with `source="search"` and with `"recommendation"`, and assert `sf=search` and `sf=recommendation` in the title link. This test fails before the fix.
- `tests/test_paper_page.py::test_paper_page_records_timings`: one request through `TestClient`, then `runtime_stats.timings()["paper_page"]` has `total_ms`.
- `tests/test_baseline_metrics.py`, on a synthetic database built in `tmp_path`: a 29-minute gap stays one session and a 31-minute gap splits it; return on D+7 counts and D+8 does not; open-then-save in one session counts and across sessions does not; unsave is not a pass; a 600-event bot is excluded; an empty database gives zeros, not errors; timestamps without a zone are read as UTC.

**Done when**

- The click fix and its test are merged and deployed, so clicks from search are stored as `search`.
- The owner has run the export and pasted the report's figures into the Before column of How we'll know it works (Implementation plan tab).
- `/healthz/runtime` on the live Space shows `paper_page` timings after a few visits.

**Watch out for:** don't backfill or rewrite old interaction rows; the report explains the gap instead. Don't call the live feed to measure latency: it writes impressions for whoever calls it.

## PR 2: daily harvest on, licence kept

PR 2 brings new papers in every day and records each one's licence. The live feed's newest paper is from 2026-09-29, and the daily job has never been switched on.

**Starting point.** Branch `feat/oai-daily-harvest` has 3 commits on top of `7aaf9ea`, plus uncommitted OAI-PMH code in `scripts/ingest_arxiv.py` (`parse_oai_record`, `fetch_oai_page`, `harvest`, `--source oai`) and an untracked `tests/test_oai_harvest.py` with 8 tests. `parse_oai_record` reads the full abstract but drops `<license>`. The untracked `reports/embedding_eval/` and `scripts/embedding_eval/` folders belong to other work that is already on `main`; leave them out of this branch.

**Build steps**

1. Ask the owner to confirm the uncommitted ingest work can be committed as is. Commit only `scripts/ingest_arxiv.py` and `tests/test_oai_harvest.py`, then run that test file.
2. Rebase onto `origin/main`, resolve conflicts, and run `make check`.
3. Add `app/licences.py`, standard library only, so scripts can import it too. `code_from_url(url)` maps licence URLs to short codes: `CC-BY-4.0`, `CC-BY-SA-4.0`, `CC-BY-NC-SA-4.0`, `CC-BY-NC-ND-4.0`, `CC0-1.0`, `ARXIV-NONEXCLUSIVE-1.0`, older `3.0` versions by the same pattern, and `""` for anything unknown. `code_from_text("CC BY 4.0")` does the same for the label on arXiv HTML pages, which PR 4 needs. `allows_figures(code)` returns true only for the approved list (Decision on the Implementation plan tab; proposed: CC BY, CC BY-SA, CC0).
4. In `parse_oai_record`, read `a:license` and add `licence_url` and `licence` to the dict.
5. In `scripts/ingest_backends.py`, create a Turso table once per run and fill it in the same batch as the `papers` rows: `paper_licences(arxiv_id TEXT PRIMARY KEY, licence TEXT NOT NULL, licence_url TEXT, source TEXT, updated_at TEXT)`, written with `INSERT … ON CONFLICT(arxiv_id) DO UPDATE SET …`. Don't add a column to `papers` (T5). The search-API path has no licence, so it writes nothing there.
6. In `scripts/daily_refresh.py`, make `stage_ingest` pass `--source oai`, and add licence coverage to the summary: the share of new papers with a licence, and the share that allow figures.
7. Owner steps, which an agent can't do: set the GitHub secrets the workflow lists, set the repository variable `DAILY_REFRESH=on`, then catch up from 2026-09-29 in runs of 5 days (`lookback_days` input). One 19-day run would encode about 20,000 papers on a CPU runner, which risks the 120-minute job limit. Each run is safe to repeat.

**Tests**

- In `tests/test_oai_harvest.py`: the sample page yields `CC-BY-4.0` for a CC BY record and `ARXIV-NONEXCLUSIVE-1.0` for a default one; a record with no `<license>` yields `""`.
- New `tests/test_licences.py`: a table of URLs (http and https, with and without a trailing slash, 3.0 and 4.0, public-domain zero) and labels (`CC BY 4.0`, `CC BY-NC-SA 4.0`, `arXiv.org perpetual non-exclusive license`), each with its expected code; `allows_figures` is false for NC, ND, the arXiv licence and unknown.
- In `tests/test_daily_refresh.py`: `stage_ingest` builds a command containing `--source oai`, and the summary reports licence coverage.
- A test with a fake `_t` that the licence statements ride in the same batch as the `papers` rows.

**Done when**

- The workflow has run green on 7 consecutive days.
- Turso `papers` holds papers dated yesterday each morning, and one of them opens at `/p/<id>` on the live site with its full abstract.
- At least 95% of each day's new papers have a licence row.

**Watch out for:** the sidecar does not change; new papers appear through Turso and the recent shard (PHASE7 §8). The preflight's `--max-recent-points` check exists because the recent shard sits on a free Qdrant cluster with a hard size limit, so read that check before raising any limit.

## PR 3: full abstract on first open

PR 3 replaces cut-off abstracts with full ones the first time anyone opens an older paper, and keeps explanations across deploys. About 90% of pre-2025 abstracts in the sidecar stop at 500 characters, so "In plain terms" is mostly written from stumps today.

**Build steps**

1. Add `app/features.py` as described under Branches and switches, with `PREVIEW_TOKEN` read from the environment. All later PRs use it.
2. Add the table `paper_abstracts(id INTEGER PRIMARY KEY AUTOINCREMENT, arxiv_id TEXT NOT NULL UNIQUE, abstract TEXT NOT NULL, source TEXT, fetched_at TEXT NOT NULL DEFAULT (datetime('now')))` to `_SCHEMA`, and register it in `turso_sync.TABLES` and `REMOTE_DDL` with an int watermark on `id` (T3).
3. Fix T7: in `init_db()`, rebuild `paper_explanations` with `id INTEGER PRIMARY KEY AUTOINCREMENT` and `cache_key TEXT NOT NULL UNIQUE` when the `id` column is missing, keeping every row. Then register it in `TABLES` the same way.
4. Add `app/abstract_svc.py`:
    1. `overlay(papers)` returns copies of the paper dicts with the stored full abstract swapped in when the current one is cut (`templates_env._is_truncated`) and the stored one is longer. One `IN (…)` query per call. It never edits the dicts it is given (T6).
    2. `request(arxiv_id)` queues a fetch when the abstract is cut and no row exists. It ignores ids that aren't in our metadata (T11).
    3. One worker task, strongly referenced and started on first use, drains the queue in batches of up to 20 ids, one arXiv API call at least 3.1 s after the previous one. It reuses `arxiv_svc._parse_entry`, but not `fetch_metadata_batch`, whose 0.35 s spacing is ten times faster than arXiv's published 1 request every 3 seconds. It honours `Retry-After` on 429 and 503, and stops for the day at `FULL_ABSTRACTS_DAILY_CAP` (default 3,000 ids).
    4. `ensure(arxiv_id, timeout=4.0)` queues and waits up to the timeout; a timeout returns `None` and leaves the fetch running. Concurrent calls for one id share one fetch.
    5. Ids that failed are skipped for 24 hours.
5. In `app/routers/paper.py`: `_fetch_one` returns `overlay([paper])[0]`. `paper_page()` calls `request()` when the switch is on and the abstract is cut. `explain()` first awaits `ensure()` for a cut abstract, so the explanation is written from the full text. The existing cache key already includes the abstract text, so a full abstract gets a fresh explanation automatically.
6. Add counters `abstracts_fetched`, `abstracts_failed` and the queue length to `/healthz/runtime`.
7. Update CURRENT-STATE: the Persistence row (six replicated tables), the LLM features row, and a new Full abstracts row.

**Tests**

- `tests/test_features.py`: each mode; an unknown value is `off` with a warning; the right token sets the preview cookie and a wrong one doesn't; `?preview=off` clears it; `preview` is visible only with the cookie.
- `tests/test_abstract_svc.py`: overlay swaps only cut abstracts for longer stored ones and leaves its input untouched; 45 queued ids become 3 requests at least 3.1 s apart (patch the clock and `asyncio.sleep`); a 429 with `Retry-After: 30` pauses 30 s; the daily cap stops fetching; an unknown id is never fetched; two concurrent `ensure()` calls make one request; a timed-out `ensure()` still lets the row land.
- In `tests/test_paper_page.py`: with the switch on, `/explain` waits for the full abstract and stores its explanation under the new key; with the switch off, the page and `/explain` behave exactly as today.
- In `tests/test_turso_sync_restore.py`: both tables are registered with an int watermark on `id`; a restore and push round trip works with the fake `_execute`; the explanation rebuild keeps 3 seeded rows and gives each an id.

**Done when**

- Locally with the switch on, an older paper shows its full abstract on the second view, and its "In plain terms" changes to match.
- Explanations and full abstracts survive a restart in a restore test.
- After deploy, the counters move and arXiv returns no 429s in a day of use.
