# Start here: agent hand-off

This tab is the contract for any agent building the redesign: what to read, the rules, the traps already found in the code, and the routine for each pull request. The PR tabs after it hold the step-by-step build.

## Hand-off prompt

Start each pull request in a fresh agent session with this message. Fill in the three angle-bracket fields from the Branches and switches table at the end of this tab. One session builds one pull request.

```text
You are continuing the ResearchIT redesign in the repo at ~/Projects/researchit.

Read these first, in this order, and follow them over your defaults:
1. CLAUDE.md in the repo root.
2. The Claude Doc https://claude.ai/code/artifact/bdb619a6-faef-4e91-b36f-06e848c10871
   Read it with the Claude Docs tools, never by fetching the URL. Read the tab
   "Implementation plan", then its sub-tabs "Start here: agent hand-off",
   the tab for your pull request, "Testing playbook" and "Release and rollback".

Your task is PR <N> (<name>) and nothing else. Do not start the next one.
- Run git fetch, then branch from origin/main as <branch>. The local main is stale.
  PR 2 is the exception: it continues the existing branch feat/oai-daily-harvest.
- Write the tests listed for the PR first and watch them fail.
- Keep new behaviour behind its switch, off by default.
- Fix every trap on the Start here tab that your PR touches, with a test.
- Stop and ask me before anything on the "Stop and ask" list.
- Never write to production Turso, never point browser tests at the live Space,
  never push to the hf remote.

When you finish: run the full non-live test suite and the PR's browser checks,
open a pull request with .github/pull_request_template.md, paste the test output
and the required screenshots into it, set the PR's Status in the Build order table
to "In progress" if it isn't already, and list anything you could not verify.
```

The project owner marks a pull request Done after merging it; agents never do. If a session runs out of context, start a new one with the same message plus a line saying what is already committed.

## What the finished product looks like

These are the target screens, taken from the approved v3 preview on 2026-10-18 with real September 2026 papers. Match their layout and behaviour; exact spacing can follow `app/static/styles.css` tokens. The [clickable preview](https://claude.ai/artifact/9KyGoDxgMyD1PBnmkFVM1X) shows every state, with its answers and data hard-coded.

**Feed on a laptop (PR 9–10).** Interest rail on the left, two columns of cards in served order, a figure only when one is saved and licensed, and a per-card Plain terms / Abstract switch.

![Proposed feed at 1440 px: interest rail and two columns of cards](https://raw.githubusercontent.com/siddhm11/ResearchIT-priv/redesign-images/docs/redesign/img/01-feed-laptop.jpg)

**Paper page on a laptop (PR 6).** Outline and actions on the left, the reading column in the middle, Ask this paper fixed on the right.

![Proposed paper page at 1440 px with outline, summary box and Ask panel](https://raw.githubusercontent.com/siddhm11/ResearchIT-priv/redesign-images/docs/redesign/img/02-paper-laptop.jpg)

**An answer in Ask this paper (PR 8).** A plain answer, one verbatim quote, and section chips that open the exact passage on arXiv.

![Ask panel answering What's new here, with a quote and section links](https://raw.githubusercontent.com/siddhm11/ResearchIT-priv/redesign-images/docs/redesign/img/03-paper-ask-answer.jpg)

**Figure lightbox (PR 5–6).** Caption, credit and licence, plus a button that asks about the figure.

![Figure 1 of SatNav enlarged in a dialog with caption, credit and Ask button](https://raw.githubusercontent.com/siddhm11/ResearchIT-priv/redesign-images/docs/redesign/img/04-figure-lightbox.jpg)

**Phone (PR 6 and 8).** One column with actions under the title; Ask opens as a bottom sheet from a floating button.

![Paper page at 390 px with the summary switch and Ask button](https://raw.githubusercontent.com/siddhm11/ResearchIT-priv/redesign-images/docs/redesign/img/05-paper-phone.jpg)

![Ask bottom sheet at 390 px answering What are the limitations](https://raw.githubusercontent.com/siddhm11/ResearchIT-priv/redesign-images/docs/redesign/img/06-phone-ask-sheet.jpg)

**Today's paper page, for contrast.** One centred column, the raw abstract, no figures and no way to ask.

![Current paper page at 1440 px: a single column of abstract text](https://raw.githubusercontent.com/siddhm11/ResearchIT-priv/redesign-images/docs/redesign/img/07-paper-today-laptop.jpg)

## Read before coding

Read these on `origin/main` (`git show origin/main:<path>` works without switching branches). Line numbers are from commit `9888b1e`, 2026-10-18.

| File | Why it matters | Needed for |
| --- | --- | --- |
| `CLAUDE.md` | The rulebook. §3.1 (quota binds served order), §3.6b (signal integrity, no blocking calls), §3.11 (instrumentation), §5 (workflow) | All |
| `docs/CURRENT-STATE.md` | What is implemented and deployed; you update its table in every PR | All |
| `docs/research/06-Deep-Research-Verdict.md`, end section | Where decisions are logged (§6.2 of CLAUDE.md gives the format) | All |
| `app/routers/paper.py` | The paper page (`/p/{id}`), `_fetch_one` (metadata with arXiv fallback) and `/api/papers/{id}/explain` | 1, 3, 4, 6, 7, 8 |
| `app/groq_svc.py` | The model chain: `_complete()` with failover and benching, `explain_paper()`, `explain_cache_key()` | 3, 7, 8 |
| `app/db.py` | SQLite schema in `_SCHEMA`, column migrations in `init_db()`, `paper_explanations` | 1, 3, 4, 7, 8, 10 |
| `app/turso_sync.py` | How local tables are backed up: `TABLES`, `REMOTE_DDL`, `restore()`, `sync_once()` | 3, 4, 7, 8, 10 |
| `app/turso_svc.py` | `fetch_metadata_batch()`: in-process cache, then sidecar, then Turso | 3, 9 |
| `app/rate_limit.py`, `app/main.py` | Per-IP limits by path prefix; middleware and static file headers | 6, 7, 8 |
| `app/http_client.py`, `app/compute_pool.py`, `app/errors.py` | Shared HTTP pool, the thread pools, and how swallowed errors are logged | 3, 4, 5, 7, 8 |
| `app/discovery_runtime.py` | The pattern for a background worker owned by the app lifespan | 10 |
| `app/templates/paper.html`, `partials/paper_card.html`, `partials/action_buttons.html`, `partials/rec_page.html` | Today's markup, and the hx-vals that carry §3.11 fields | 1, 6, 9, 10 |
| `app/static/app.js`, `app/static/styles.css` | Deferred dismiss with Undo, the infinite-scroll observer, design tokens | 6, 9, 10 |
| `scripts/ingest_arxiv.py`, `scripts/ingest_backends.py`, `scripts/daily_refresh.py`, `.github/workflows/daily-refresh.yml` | How new papers reach Turso and Qdrant | 2, 10 |
| `tests/conftest.py`, `tests/test_paper_page.py`, `tests/test_event_loop_blocking.py`, `tests/test_e2e_browser.py` | Test isolation, the fake-metadata pattern, the AST guard, the browser suite | All |
| `.github/workflows/tests.yml`, `Makefile`, `constraints.txt` | CI installs its own short dependency list; `make lock` pins versions | Any new dependency |

The main tab of this doc explains why the redesign exists. You don't need it to build.

## Traps found in the code

These are the mistakes a capable agent would make on a first pass. Each was found by reading `origin/main` on 2026-10-18; T1 was confirmed by rendering the template. Every trap your PR touches gets fixed with a regression test, and the PR description names its T-numbers.

### Data and storage

- **T1. Every click is logged as a feed click** (PR 1). `paper_card.html` line 97 builds `sf=` from `_source`, which is only set inside `action_buttons.html`, and a `set` in an included template never flows back. So every card link ends `sf=`, and `paper.py` line 158 stores `source="recommendation"`, search clicks included. **Do:** use `source`, and add a test that renders the card for search and for the feed.
- **T2. Turso never sends rows back after boot** (PR 4, 10). `turso_sync.restore()` fills only tables that are empty at boot; after that, `sync_once()` only pushes. A row that a GitHub Action writes to Turso never reaches a running Space. **Do:** write paper rows only inside the Space, on first open or from a worker the app lifespan owns. The Action only harvests papers.
- **T3. Timestamp watermarks drop rows** (PR 3, 4, 7, 8). `sync_once()` pushes `WHERE wm > ? ORDER BY wm LIMIT 500`. With a 1-second `created_at` watermark, a batch that ends partway through rows sharing one timestamp skips the rest for good, and a warm-up writes hundreds of rows a minute. **Do:** give every new replicated table `id INTEGER PRIMARY KEY AUTOINCREMENT` plus a `UNIQUE` natural key, and register it with `"watermark": "id", "wm_kind": "int"`; `_reserve_id_space()` hard-codes the column name `id`. Upsert with `INSERT OR REPLACE`, which gives an updated row a new id so it is pushed again.
- **T4. Restore is one unpaged query** (PR 4). `restore()` reads a whole remote table in one request with a 60 s timeout. Once `paper_sources` holds tens of thousands of rows, restore fails and every deploy starts with no figures. **Do:** keep replicated rows slim (at most 8 figures, captions cut to 300 characters, no paragraph text) and page restore by `id` in chunks of 2,000, with a test.
- **T5. Replace wipes the columns you don't list** (PR 2). `INSERT OR REPLACE` deletes the old row; `ingest_backends.py` already resets `citation_count` to 0 that way. A licence column on Turso `papers` would vanish on any re-ingest that omits it. **Do:** keep licences in their own table, or use `INSERT … ON CONFLICT(key) DO UPDATE SET` naming each column.
- **T6. The metadata cache keeps the 500-character abstract** (PR 3, 9). `turso_svc._METADATA_CACHE` holds up to 50,000 paper dicts with the sidecar's cut abstract. **Do:** overlay the full abstract on a copy after `fetch_metadata_batch()` returns. Never edit the cached dict, which every request shares.
- **T7. Explanations vanish at every deploy** (PR 3). `paper_explanations` isn't in `turso_sync.TABLES`, and its primary key is `cache_key`, so it can't take an id watermark as it is. **Do:** in `init_db()`, rebuild the table with `id` plus `UNIQUE(cache_key)` when the `id` column is missing (rename, create, copy, drop, in one transaction), then replicate it.
- **T8. A new column on interactions must reach Turso too** (PR 10). `TABLES["interactions"]["cols"]` and `REMOTE_DDL` list columns by hand, and `CREATE TABLE IF NOT EXISTS` never adds a column to the existing remote table. Miss a step and every push fails, so nothing is backed up. **Do:** add the local migration, run `ALTER TABLE user_interactions ADD COLUMN …` in `ensure_remote_schema()` ignoring only the duplicate-column error, then extend `cols`; test it with the fake `_execute` from `test_turso_sync_restore.py`.

### Requests, limits and cost

- **T9. The rate limiter can't express per-paper routes** (PR 7, 8). `rate_limit._rule_for()` matches path prefixes, so `/api/papers/{id}/ask` can't be listed, and `/api/papers` would throttle Save and Pass. `/explain` has no limit today, so a crawler can spend the Groq quota through it. **Do:** add suffix rules for `/explain`, `/briefing`, `/ask` and `/code`, with tests.
- **T10. One model worker serves all of search** (PR 8). On the 2-vCPU Space, `compute_pool` runs a single inference worker. Embedding 30–60 paragraphs with BGE-M3 when someone asks a question would queue every search behind it. **Do:** rank the paper's own paragraphs with BM25 in plain Python, which needs no model and takes about a millisecond for 150 paragraphs.
- **T11. Any id can trigger an arXiv fetch** (PR 3, 4). `_fetch_one()` falls back to the arXiv API for unknown ids, so `/p/<anything>` costs an outside call. **Do:** fetch only for ids already in our metadata that match `^\d{4}\.\d{4,5}$`, keep one fetch in flight per paper, and cap total concurrency.
- **T12. Changing a Space variable restarts the Space** (all). A restart wipes `/tmp`: feed impressions, exposures, follows, thumbnails and cached full text reset (CURRENT-STATE, Operational boundaries). **Do:** batch switch changes, flip them at a quiet hour, and note it in the release log. Confirm the restart on the first flip.

### arXiv HTML and figures

- **T13. Image paths are relative, and some are empty** (PR 4, 5). A source looks like `2609.31507v1/overview-final.png` and resolves against `https://arxiv.org/html/`, not the page URL; after a redirect to a trailing-slash URL it 404s. Missing images are `src=""` with class `ltx_missing_image` (6 of them in 2609.20888), and alt text is always "Refer to caption". **Do:** store the relative path, validate it with a strict pattern, build `https://arxiv.org/html/` + path, skip empty ones, and use the caption as alt text.
- **T14. Each formula appears three times** (PR 4). A `<math>` element holds `alttext` (TeX), MathML token text and a TeX `<annotation>`. Naive text extraction repeats every formula and breaks quote matching. **Do:** replace each `<math>` with `$` + `alttext` + `$` and skip its children.

### Templates and front end

- **T15. Static files have no version in the URL** (PR 6). `base.html` loads `/static/styles.css` with `max-age=3600`, so after a deploy a reader can get new HTML with old CSS and JS for an hour. **Do:** add a `static_version` template global (a short hash of the file contents at startup) and append `?v=` to every static URL, before any UI change.
- **T16. Dots in arXiv ids break CSS selectors** (PR 6, 8, 9). `#actions-1706.03762` matches nothing; it already broke Save once (see `action_buttons.html`). **Do:** use `[id='…']` in every new `hx-target` and `querySelector`.
- **T17. CSS columns break the served order** (PR 10). `columns: 2` fills the left column top to bottom, so card 2 lands under card 1 instead of beside it, undoing the quota interleave (§3.1). Two separate column containers break infinite scroll, which swaps the loader in place. **Do:** keep one `.feed` list in served order and lay it out with CSS grid plus a small row-span script.
- **T18. Model output and arXiv text are untrusted** (PR 6, 7, 8). Captions carry HTML, and paper text can carry instructions aimed at a model. **Do:** extract text only from captions, mark passages as data in prompts, accept only JSON in a fixed schema, build every link from validated anchors, and never use `|safe` in a new template.
- **T19. A replicated row doesn't mean the local data is here** (found 2026-10-21). `paper_sources` comes back from Turso after a deploy, but the paper's `paper_fulltext` and `paper_paragraphs` don't, and `prune()` drops them too. `ensure()` saw a current row and never fetched again, so briefings were stored as "too few passages" for good. **Do:** check `paper_source_svc.has_text()`, call `ensure(need_text=True)` before reading the text, and never store a failure that only means this container lacked something.
- **T20. One URL, two bodies** (found 2026-10-21). `/search?q=` answers htmx with a fragment and a browser with a page; without `Vary: HX-Request`, Back showed the cached fragment as the page. **Do:** keep the middleware's `Vary` and `no-store` headers, and give `HX-History-Restore-Request` the whole page in any route that pushes its URL.
- **T21. Cache keys must hold only what changes the output** (found 2026-10-21). The briefing key held the parser version and model, so a figure-only parser bump threw away every briefing and one came back worse. **Do:** key on the paper version and prompt, and never let a worse result replace a good one. Ask's key still holds both.
- **T22. Test harness blind spots** (found 2026-10-21). Playwright's Chromium disables the back-forward cache, so Back always reloads in browser tests; and `TestClient` keeps the cookie the server sets, so `cookies.set()` alone doesn't switch readers. **Do:** test Back through server-side behaviour, and clear cookies before switching readers.
- **T23. The daily job's operating limits** (found 2026-10-20). A runner encodes about 2,400–2,900 papers an hour; a newer dispatch replaces a pending one in the `daily-refresh` group; GitHub fires the schedule hours late. **Do:** catch up one run at a time with `today`, `lookback_days` and `ingest_timeout_min` (PHASE7 §8.1).

## Ground rules

These come from CLAUDE.md and from how the code already works. Copy the existing pattern before inventing a new one.

- **Ranking is off limits.** No PR changes `app/recommend/`, the scoring or order in `app/routers/recommendations.py`, or `app/hybrid_search_svc.py`. Figures, briefings and Ask are display-only. PR 10 changes layout and logging, never order.
- **Every feature has a switch, off by default.** Read it in `app/config.py` with `os.getenv`, like `RATE_LIMIT_ENABLED`. Values are `off`, `shadow` (does the work, shows nothing), `preview` (shows only to the owner) and `on`. An unknown value means `off` and prints a warning.
- **Nothing blocks the event loop** (§3.6b). Use `asyncio.to_thread` for `sqlite3`, `local_meta`, HTML parsing and Pillow; `compute_pool.run_inference` for model calls; `http_client.get_client()` for HTTP, with a per-request `timeout=` and never `async with`.
- **Background work is strongly referenced.** Spawn tasks through a module-level set, as `events._spawn` does. Keep one fetch in flight per paper with a `dict[str, asyncio.Future]`.
- **Storage follows `app/db.py`.** Tables go in `_SCHEMA` as `CREATE TABLE IF NOT EXISTS`; new columns go in a migration list in `init_db()`. Use `aiosqlite` for the app database. Replicated tables follow T3.
- **Fail quietly, log loudly.** A new section renders nothing when its data is missing, and the page never returns 500 because arXiv, a model or Turso failed. Swallowed errors go through `errors.report(module, message, exc)`; other logs are `print("[module] …")`.
- **arXiv ids are strings.** Validate new-style ids with `^\d{4}\.\d{4,5}$` and keep the version (`v1`) in its own field.
- **The page loads nothing from other origins.** `base.html` promises this. Vendor KaTeX under `/static/vendor/`, and serve figures from our own `/fig/` route rather than linking arxiv.org images.
- **Identify ourselves to arXiv and Hugging Face.** Move the ingest's `USER_AGENT` into `app/config.py` as `OUTBOUND_USER_AGENT` and send it on every outside request.
- **Docs ship with the code.** Each PR updates its row in `docs/CURRENT-STATE.md`, adds a doc 06 changelog entry for any decision (licence policy, model features, layout and logging), and updates this doc's Build order status.

## Routine for every pull request

1. `git fetch origin`, then `git switch -c <branch> origin/main`. Run `make test` once to see the suite green before you change anything.
2. Read your PR's section in full, including its Done when and its tests. If something in it contradicts the code you find, stop and ask; the code wins until the owner decides.
3. Write the listed tests first and run them to see them fail for the right reason.
4. Build behind the switch. Commit in small steps with conventional messages (`feat(paper): …`, `fix(card): …`, `test(…): …`).
5. Run `make check` (compile everything, then every non-live test). It must pass with no new warnings.
6. For UI work, start a scratch server and run the browser tests against it (Testing playbook, Commands). Take screenshots at 1440, 1180, 900 and 390 px wide, light and dark, and compare them with the target screens above.
7. Run the PR's live checks only if its section lists them, and only read-only.
8. Update CURRENT-STATE, the doc 06 changelog where needed, and the Build order status.
9. Push the branch and open a pull request with `.github/pull_request_template.md`. Include the test output, the screenshots, the T-numbers fixed, and what you could not verify.
10. Stop. The owner reviews, merges without squashing, and decides when to deploy.

## Stop and ask

Ask the owner, and wait for an answer, before any of these. Say what you want to do, why, and what happens if the answer is no.

- Adding a dependency, including Pillow, KaTeX and any model SDK (CLAUDE.md §5.2).
- Changing ranking, quota, MMR, clustering, search fusion, `policy_id` or the ranker version string.
- Changing the schema of a replicated table that already exists (`interactions`, `user_*`) or of Turso `papers`.
- Reading production user data or writing to production Turso, Qdrant or the Space. PR 1's export is run by the owner.
- Anything that costs money: a paid model, a higher budget cap, a paid storage bucket.
- Showing figures under a licence that is not on the approved list.
- Pushing to the `hf` remote, or changing Space variables, secrets or GitHub repository settings.
- Changing an existing test's expectations, or deleting anything outside your own branch's work.
- Finding that this plan and the code disagree.

## Branches and switches

Switches are Space variables read in `app/config.py`. `app/features.py`, built in PR 3, turns them into one call: `features.mode("ASK")` returns the mode, and `features.visible("ASK", request)` is true for `on`, or for `preview` when the request carries the owner's preview cookie. The cookie is set by visiting any page with `?preview=<PREVIEW_TOKEN>` (a Space secret) and cleared with `?preview=off`.

| PR | Branch | Switch | Values, default first |
| --- | --- | --- | --- |
| 1 | `fix/click-surface-baseline` | None: a bug fix and a script | — |
| 2 | `feat/oai-daily-harvest` (exists) | GitHub variable `DAILY_REFRESH` | `off`, `on` |
| 3 | `feat/full-abstract-on-open` | `FULL_ABSTRACTS` | `off`, `on` |
| 4 | `feat/paper-source-svc` | `PAPER_SOURCES` | `off`, `shadow`, `on` |
| 5 | `feat/figure-thumbnails` | `FIGURES` | `off`, `on` |
| 6 | `feat/paper-page-v2` | `PAPER_PAGE_V2` | `off`, `preview`, `on` |
| 7 | `feat/paper-briefing` | `BRIEFING`, plus `BRIEFING_PROVIDER`, `BRIEFING_MODEL`, `LLM_DAILY_USD_CAP` | `off`, `shadow`, `preview`, `on` |
| 8 | `feat/ask-this-paper` | `ASK`, plus `ASK_DAILY_CAP` | `off`, `preview`, `on` |
| 9 | `feat/feed-card-v2` | `FEED_CARD_V2` | `off`, `preview`, `on` |
| 10 | `feat/wide-feed` | `FEED_WIDE`, `FIGURE_PREWARM` | `off`, `preview`, `on`; `off`, `on` |

Order matters: 4 needs 3's `features.py`, 5 and 6 need 4, 7 and 8 need 4 and 6, 9 needs 4 and 5, and 10 needs 9. PR 1 and PR 2 can run in parallel with everything.
