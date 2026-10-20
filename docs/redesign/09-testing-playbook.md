# Testing playbook

Every pull request ships with tests at several layers, and every trap on the Start here tab gets a test that would have caught it. Nothing in the default test run touches the network or production data.

## Test layers

Each layer catches a different kind of mistake. CI runs the first six on every push; the rest run locally or by hand at the points listed.

| Layer | Catches | How | When |
| --- | --- | --- | --- |
| Unit | Wrong logic in pure functions: parser, licence codes, number and quote checks, BM25, sessions | Plain pytest, no app, no network | Every push |
| Integration | Wiring: routes, templates, SQLite, switches, limits, replication specs | `TestClient(app)` on the temporary database from `tests/conftest.py`, with fakes for HTTP and the model | Every push |
| Golden | The parser drifting on real pages | Trimmed CC BY arXiv pages with expected JSON beside them | Every push |
| Static guards | Rule breaks that look harmless in review | AST and text checks like `test_event_loop_blocking.py` (list below) | Every push |
| Resilience | Failures that should stay quiet | Fakes returning 429, 503, timeouts, malformed HTML, Turso down, a full disk | Every push |
| Security | XSS, SSRF, path traversal, image bombs, prompt injection, abuse | Shared payload lists in `tests/security_payloads.py` | Every push |
| Browser | What only a real browser shows: swaps, focus, layout, overflow, console errors, layout shift | Playwright, marker `browser`, against a scratch server | Each UI PR, locally. CI doesn't install Playwright |
| Visual | Drift from the target screens | Screenshots at 1440, 1180, 900 and 390 px, light and dark, attached to the PR | Each UI PR |
| Performance | Slower pages | `scripts/perf_paper_page.py` (new) and the feed timings | Each PR that touches `/p/` or the feed |
| Model evaluation | Bad briefings and answers | `eval_briefings.py`, `eval_ask.py`, owner ratings | Before PR 7 or 8 turns on |
| Live smoke | The deployed Space working | `pytest -m live`, read-only | After each deploy, by hand |

**Static guards to add:**

- Calling `arxiv_html.parse` or `Image.open` inside an `async def` fails (extend the existing AST test).
- `async with http_client.get_client()` anywhere fails, because it would close the shared pool.
- A bare `asyncio.create_task(` in a new module fails unless its result is stored.
- `|safe` in any template added after this plan fails.
- An `http://` or `https://` URL in `app/static/` or a `<script src>` or `<link href>` to another origin fails. Plain `<a href>` links to arXiv and GitHub are fine.

**What "deep" means here:**

- Every T-number gets a test that fails on today's code or on the naive fix.
- Failure paths get as many tests as the happy path.
- New tests take the `no_network` fixture (below), so an accidental outside call fails loudly instead of quietly hitting arXiv.

## Fixtures and fakes

Nothing in the default run touches the network or production. These are the stand-ins, each built once and shared.

- **arXiv pages.** Create `tests/fixtures/arxiv_html/`, the repo's first fixtures folder (CLAUDE.md §5.5 notes none exists; say so in PR 4's description). It holds six CC BY 4.0 pages, all checked on 2026-10-17:
    - 2609.31507 (SatNav): panels, appendices, 142 paragraphs
    - 2609.20888: six missing images
    - 2609.27176: tables and a JPEG
    - 2609.29861: a short page
    - 2609.30356: 16 PNG figures
    - 2609.30670: a `.jpeg` image

    Trim each to 60 KB or less with a new `scripts/make_arxiv_fixture.py <id> --keep S1,S3.SS4,…`. A trimmed page keeps the infobox, abstract, two or three sections, two figures, one table and a stub bibliography. Add a `README.md` crediting each paper's authors, title and licence. Never commit a page under another licence.
- **Golden files.** `<id>.expected.json` beside each page. Add a `--update-golden` option in `tests/conftest.py` (`pytest_addoption`) that rewrites them, and review every diff by hand.
- **Fake arXiv and Hugging Face.** A fixture `fake_http(routes)` swaps `http_client._client` for an `httpx.AsyncClient` on an `httpx.MockTransport` that maps URLs to fixture files or status codes, and restores the real client afterwards.
- **No network.** A fixture `no_network` installs a transport that fails the test on any request it wasn't given. Existing tests that rely on real Turso metadata reads keep working, because the fixture is opt-in.
- **Fake model.** `LLM_PROVIDER=fake` returns canned JSON per test, or echoes passages back for injection tests. It refuses to load unless `RESEARCHIT_ENV=test`.
- **Fake Turso.** Reuse the `_execute` fake from `tests/test_turso_sync_restore.py`.
- **Fake clock.** Patch `time.monotonic` and `asyncio.sleep` for the pacer, queue and breaker tests, so they run instantly.
- **Images.** Generated with Pillow inside the test, never committed.
- **Browser test world.** A scratch server with its own database, replication off, the fake model, and `ARXIV_HTML_BASE` pointing at a tiny local static server. The seed script builds that site in `/tmp/e2e_arxiv` from the fixtures (pages at `html/<id>`, images generated), so nothing generated is committed. `ARXIV_HTML_BASE` is ignored unless `RESEARCHIT_ENV=test`, so production can never be pointed elsewhere. `scripts/seed_e2e.py` (new) fills the scratch database with saved sources, explanations and briefings for the fixture papers.

## Required tests per pull request

A tick means the PR is not done without tests at that layer. The PR tabs list the specific tests.

| PR | Unit | Integration | Golden | Guards | Browser | Performance | Security | Model eval | Live smoke |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 Click fix, baseline | ✓ | ✓ | | | | | | | ✓ |
| 2 Harvest, licence | ✓ | ✓ | | | | | | | Workflow runs |
| 3 Full abstracts | ✓ | ✓ | | ✓ | | | ✓ | | ✓ |
| 4 Paper sources | ✓ | ✓ | ✓ | ✓ | | ✓ | ✓ | | ✓ |
| 5 Thumbnails | ✓ | ✓ | | ✓ | | ✓ | ✓ | | ✓ |
| 6 Paper page | | ✓ | | ✓ | ✓ | ✓ | ✓ | | ✓ |
| 7 Briefing | ✓ | ✓ | | | ✓ | | ✓ | ✓ | ✓ |
| 8 Ask | ✓ | ✓ | ✓ | | ✓ | ✓ | ✓ | ✓ | ✓ |
| 9 Feed card | | ✓ | | ✓ | ✓ | ✓ | ✓ | | ✓ |
| 10 Wide feed | ✓ | ✓ | | | ✓ | ✓ | | | ✓ |

Every PR also runs the whole existing suite. A PR that changes an existing test's expectations needs the owner's OK first.

## Commands

Run from the repo root with the project's `.venv` (Python 3.12).

```bash
# Everything CI runs: compile all files, then every non-live test
make check

# One file, verbose
.venv/bin/python -m pytest tests/test_arxiv_html.py -v

# Rewrite golden files after a deliberate parser change, then review the diff
.venv/bin/python -m pytest tests/test_arxiv_html.py --update-golden
git diff tests/fixtures/

# Browser tests. Seed first: builds the scratch database and a fake arXiv site in /tmp
.venv/bin/python scripts/seed_e2e.py --db /tmp/e2e.db --site /tmp/e2e_arxiv
.venv/bin/python -m http.server 8799 --directory /tmp/e2e_arxiv &
DB_PATH=/tmp/e2e.db TURSO_SYNC_DISABLED=1 RESEARCHIT_ENV=test LLM_PROVIDER=fake \
  ARXIV_HTML_BASE=http://127.0.0.1:8799 PORT=7861 .venv/bin/python run.py &
E2E_BASE_URL=http://127.0.0.1:7861 .venv/bin/python -m pytest -m browser -v

# Page timings against the scratch server
.venv/bin/python scripts/perf_paper_page.py --base http://127.0.0.1:7861 --n 50

# Live smoke after a deploy (read-only checks only)
.venv/bin/python -m pytest -m live tests/test_live_smoke.py -v
```

The browser server must not use port 7860, which the dev server uses. Stop both background processes when you're done.

## As built (2026-10-19)

The browser world is `scripts/seed_e2e.py`. It also writes a tiny metadata sidecar, so the server needs no Turso:

```bash
.venv/bin/python scripts/seed_e2e.py --db /tmp/e2e.db --site /tmp/e2e_arxiv --sidecar /tmp/e2e_sidecar.sqlite
.venv/bin/python -m http.server 8799 --bind 127.0.0.1 --directory /tmp/e2e_arxiv &
DB_PATH=/tmp/e2e.db METADATA_SIDECAR_PATH=/tmp/e2e_sidecar.sqlite RESEARCHIT_ENV=test \
  ARXIV_HTML_BASE=http://127.0.0.1:8799 TURSO_URL= TURSO_DB_TOKEN= QDRANT_URL= QDRANT_B_URL= \
  QDRANT_RECENT_URL= ZILLIZ_URI= GROQ_API_KEY= TURSO_SYNC_DISABLED=1 SIDECAR_WARMUP=0 \
  PAPER_PAGE_V2=on FIGURES=on PAPER_SOURCES=on PORT=7861 .venv/bin/python run.py &
E2E_BASE_URL=http://127.0.0.1:7861 .venv/bin/python -m pytest tests/test_e2e_paper.py -v
```

Blanking the service variables matters: `load_dotenv` never overrides a name that is already set, so the empty values keep `.env.local` out.
