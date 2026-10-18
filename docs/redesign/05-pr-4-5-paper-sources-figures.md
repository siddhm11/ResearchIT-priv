# PR 4–5: Paper sources and figures

PR 4 reads each paper's arXiv HTML page once and saves its licence, sections, paragraphs and figures. PR 5 turns saved figures into small cached images. Readers see nothing new until PR 6.

## How arXiv HTML is laid out

arXiv builds its HTML pages with LaTeXML, and the markup is regular enough to parse with Python's standard `html.parser`, so no new dependency is needed. The rules below were checked on six pages fetched on 2026-10-17 (2609.20888, 2609.27176, 2609.29861, 2609.30356, 2609.30670, 2609.31507), which range from 0.26 to 1.4 MB.

| Part | Markup | What to keep |
| --- | --- | --- |
| Licence | `<a id="license-tr">License: CC BY 4.0</a>` inside `div#infobox` | The label, turned into a code by `licences.code_from_text` |
| Version | `div#watermark-tr`: `arXiv:2609.31507v1 [cs.CV] 25 Sep 2026` | `v1` |
| Title | `h1.ltx_title_document` | Text |
| Abstract | `div.ltx_abstract`, then `p.ltx_p` | Text, with maths as TeX |
| Sections | `section` with `id` and class `ltx_section`, `ltx_subsection`, `ltx_paragraph` or `ltx_appendix`, e.g. `S3`, `S3.SS4`, `S3.SS4.SSS0.Px1`, `A1` | Anchor, number from `span.ltx_tag`, title from the first `h2`–`h6.ltx_title` inside |
| Paragraphs | `div.ltx_para` with an `id` such as `S3.SS4.SSS0.Px1.p1`; SatNav has 142 | Anchor, the enclosing section's anchor, text, and the figure and table anchors it links to (`a.ltx_ref` with `href="#S…"`) |
| Maths | `<math alttext="m\in\mathcal{M}">` with MathML and a TeX annotation inside | `alttext` only, wrapped in `$…$` (T14) |
| Figures | `figure.ltx_figure` with an id like `S1.F1`; sub-panels are nested `figure.ltx_figure_panel` (`S3.F3.sf1`) | Anchor, label from `span.ltx_tag_figure` ("Figure 1"), caption text without the label, image path, width and height |
| Images | `img.ltx_graphics` with `src="2609.31507v1/overview-final.png" width="476" height="235"`; a missing one has `src=""` and class `ltx_missing_image` | The relative path (T13). PNG and JPEG only in this sample |
| Tables | `figure.ltx_table` with an id like `S5.T1` | Anchor, label and caption, as citation targets. Not figures |
| Skip | `nav.ltx_TOC`, `section.ltx_bibliography` (`#bib`), arXiv's header and footer, any `img` under `/static/` | — |

One in ten new papers has no HTML page (54 of 60 in the September sample had one). Older papers mostly do: 28 of 30 from 2019–2024 and 22 of 24 from 2007–2010. Some older pages have fewer images than figures, so an older paper may show no figure even with a page.

Use a stack-based `HTMLParser` subclass: push on start tags, pop on end tags, and route text to whatever the top of the stack is collecting. Treat `<math>` as a leaf that emits its `alttext` and ignores everything inside it until `</math>`.

## PR 4: paper_source_svc

PR 4 fetches a paper's arXiv HTML page once, parses it, and saves the result so that no reader-facing request ever waits on arXiv. It ships in `shadow` mode first: it fetches when papers are opened and shows nothing.

**Build steps**

1. Add `app/arxiv_html.py`, a pure parser with no I/O: `parse(html: str, arxiv_id: str) -> ParsedPaper`, built from dataclasses (`Section`, `Paragraph`, `Figure`, `TableRef`), following the table above. Set `PARSER_VERSION = 1`. Refuse pages over 8 MB, and stop collecting after 600 paragraphs and 80 figures. Malformed input returns a result marked `error`; it never raises.
2. Add three local tables to `_SCHEMA`:
    1. `paper_sources`, replicated and kept slim (T4): `id INTEGER PRIMARY KEY AUTOINCREMENT`, `arxiv_id TEXT NOT NULL UNIQUE`, `status` (`ok`, `no_html`, `error`, `skipped`), `version`, `licence`, `figures_json` (at most 8 figures: anchor, label, caption cut to 300 characters, path, width, height), `n_figures`, `n_sections`, `n_paragraphs`, `code_repo`, `code_stars`, `code_checked_at`, `parser_version`, `fetched_at`, `error`.
    2. `paper_fulltext`, local only: `arxiv_id TEXT PRIMARY KEY`, `version`, `sections_json`, `figures_json` (all figures), `tables_json`, `fetched_at`, `last_used_at`.
    3. `paper_paragraphs`, local only: `(arxiv_id, anchor)` as the primary key, `section_anchor`, `ord`, `text`, `refs_json`. PR 8 ranks these per question. A paper has about 150 paragraphs, so no search index is needed.
3. Add `app/paper_source_svc.py`:
    1. `get(arxiv_id)` and `get_many(ids)` read local rows only. The feed and paper page use these, so they never wait on arXiv.
    2. `ensure(arxiv_id)` fetches and saves, one fetch in flight per paper. It refuses ids that fail the pattern or aren't in our metadata (T11). It requests `https://arxiv.org/html/{id}` through `http_client.get_client()` with `timeout=15`, redirects followed, `OUTBOUND_USER_AGENT`, and stops reading at 8 MB. It parses with asyncio.to_thread and writes all tables in one transaction. A refetch rewrites the paper_sources row whole and carries over the code columns that PR 6 fills.
    3. A shared pacer allows 1 arXiv request every 2 s with bursts of 3. Three soft blocks (403, 406, 429, 503) within 10 minutes open a breaker for 15 minutes, or for `Retry-After` if longer. While it is open, `ensure()` returns what is saved and fetches nothing. PR 5's image requests use the same pacer.
    4. Refetch rules: a lower `parser_version` refetches on next use, `no_html` is retried after 7 days (arXiv converts some papers later), and `error` after 1 day.
    5. `schedule(arxiv_id)` starts `ensure()` in the background through a strongly referenced set.
    6. A daily prune keeps full text for the `FULLTEXT_MAX_PAPERS` most recently used papers (default 5,000; measure the database size before raising it).
4. In `paper_page()`, call `schedule()` when `PAPER_SOURCES` is `shadow` or `on`, after the response is built. Render nothing new.
5. Register `paper_sources` in `turso_sync.TABLES` with an int watermark on `id`, and add `"paged": True`, which makes `restore()` read it in pages of 2,000 by `id` (T4). Only tables marked `paged` use the new path, so the restore of `interactions` doesn't change.
6. Add `GET /healthz/sources`: counts by status, the licence mix, fetches and soft blocks in the last hour, breaker state, queue length, median fetch and parse times, and full-text rows and file size.
7. Add a doc 06 changelog entry: "Paper source layer: arXiv HTML fetched once per paper; licence read from the page; display-only."

**Tests**

- `tests/test_arxiv_html.py`, on the trimmed fixtures (Testing playbook): licence and version for each fixture; SatNav's section anchors, titles and numbers; each formula appears once (T14); `src=""` gives a missing figure (T13); panels nest under their figure; tables are not figures; nothing from the bibliography appears; captions lose their "Figure 1:" prefix; paragraph refs list `S1.F1`; caps are enforced; cutting a fixture at 50 random points never raises; bytes that aren't UTF-8 still parse.
- `tests/test_paper_source_svc.py`, with an `httpx.MockTransport` client swapped into `http_client._client`: 200 saves every table; 404 saves `no_html`; three 429s open the breaker and the next call makes no request; a redirect to a versioned URL is followed; a 9 MB body is cut at 8 MB; 10 concurrent `ensure()` calls make 1 request; an unknown id makes none; parsing the 1.4 MB fixture keeps event-loop lag under 50 ms, measured by a ticker task; replicated rows hold at most 8 figures with captions of 300 characters or fewer.
- In `tests/test_turso_sync_restore.py`: a `paged` table with 4,500 fake remote rows restores in 3 requests, and `interactions` still restores in one.
- Extend `tests/test_event_loop_blocking.py` so a call to `arxiv_html.parse` inside an `async def` fails the build.

**Done when**

- A local run over the 60-paper September sample matches the probe within 5 points: about 90% with HTML and 42% with a figure-allowing licence and figures.
- After 3 days in `shadow` on the live Space, `/p/` has returned no 500s, soft blocks are under 1% of fetches, and `/healthz/runtime` shows no rise in feed or paper-page p95.

## PR 5: figure thumbnails

PR 5 serves each allowed figure as a small WebP from our own server, fetched from arXiv once and cached on disk. Original images run to 2.1 MB (median 182 KB, 90th percentile 919 KB, September sample), too heavy for a feed.

**Needs first:** the owner's approval for Pillow (CLAUDE.md §5.2), and the figure licence decision. Add Pillow to `requirements.txt`, run `make lock`, and add it to the install list in `.github/workflows/tests.yml` (T8).

**Build steps**

1. Add `app/routers/figures.py` with `GET /fig/{arxiv_id}/{n}-{h8}.webp`. `n` is the figure's position in the saved list (0 to 7) and `h8` is the first 8 hex characters of the SHA-1 of its saved path, so a refetched figure gets a new URL. The URL never contains a source address.
2. Check, in this order, and answer 404 at the first failure: `FIGURES` is `on`; the id matches the pattern; a `paper_sources` row exists with status `ok`; `licences.allows_figures(row.licence)`; figure `n` exists; `h8` matches. The licence check lives here and in the template helper, never only in a template.
3. Serve a cached file from `FIG_CACHE_DIR/{arxiv_id}/{n}-{h8}.webp` (default `/tmp/figcache`) when present, with `Cache-Control: public, max-age=31536000, immutable`.
4. Otherwise generate it, one generation in flight per figure:
    1. Validate the saved path against `^<id>v\d+/[A-Za-z0-9._/-]+\.(png|jpe?g|gif)$`, rejecting any `..`. Fetch `https://arxiv.org/html/` + path through PR 4's pacer. Require an `image/*` content type and at most 10 MB.
    2. In `asyncio.to_thread`, behind a semaphore of 2 (the Space has 2 vCPUs): set `Image.MAX_IMAGE_PIXELS = 50_000_000`, open, apply `ImageOps.exif_transpose`, take the first frame, flatten transparency onto white, convert CMYK and palette images to RGB, shrink to at most 800 px wide (never enlarge), and save WebP at quality 80. If the file is over 150 KB, retry at 70, then 60.
    3. Write to a temporary file and `os.replace` it into place, so a half-written file is never served.
    4. On failure, write a marker that suppresses retries for 24 hours, and answer 404. The page then shows no figure.
5. An hourly sweep deletes the least recently used files once the cache passes `FIG_CACHE_MAX_MB` (default 1,024).
6. Add a rate-limit rule for `/fig/` (240 a minute per IP) to stop scraping, and a template helper `figure_url(source, n)` that returns `None` whenever the checks above would fail.
7. Add a credit helper that builds "First author et al. Year, CC BY 4.0" from metadata and the licence code. Every displayed figure carries it.

**Tests**

- `tests/test_thumbnails.py` builds images in the test with Pillow: an RGBA PNG comes out on white; a CMYK JPEG converts; an animated GIF gives its first frame; an EXIF-rotated photo comes out upright; a 12,000 × 12,000 image is refused; a 300 px image is not enlarged; a noisy 4,000 px image still lands under 150 KB.
- `tests/test_figures_route.py`, with fake `paper_sources` rows and an `httpx.MockTransport`: an allowed licence returns 200 `image/webp`; NC, ND, the arXiv licence and unknown return 404; a bad `n` or a wrong `h8` returns 404; paths like `2609.31507v1/../../x.png` or `https://evil.example/a.png` are refused before any request; an HTML response returns 404; the second request makes no network call; 5 concurrent requests make 1 fetch; `FIGURES=off` returns 404; the sweep removes the oldest files first.
- An AST guard: no `Image.open` inside an `async def`.

**Done when**

- On the 25 figure-allowed papers in the September sample, at least 90% of first figures produce a thumbnail under 150 KB, in under 1.5 s cold and under 20 ms from cache.
- A live spot check of 10 papers under other licences returns 404 for every figure.
