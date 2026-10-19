# PR 6–7: Paper page and briefing

PR 6 rebuilds the paper page from what PR 3–5 saved. PR 7 adds the AI briefing to it. Both ship behind switches, and today's page comes back when they are off.

## Target layout

The page has three zones on a laptop, two on a small laptop or tablet, and one on a phone. Build against these screenshots; the Start here tab has the Ask answer and lightbox states too.

![Paper page at 1440 px: outline and actions, reading column, Ask panel](https://raw.githubusercontent.com/siddhm11/ResearchIT-priv/redesign-images/docs/redesign/img/02-paper-laptop.jpg)

| Window width | Columns | Ask this paper |
| --- | --- | --- |
| 1180 px and wider | Outline and actions (12rem), reading column (up to 44rem), Ask panel (23rem) | Panel fixed beside the text, full window height |
| 900–1179 px | Reading column and Ask panel; the outline hides and the actions sit under the title | Panel beside the text |
| Under 900 px | One column, actions in a row under the title | Floating button opens a bottom sheet |

With Ask switched off, the reading column takes the panel's space. Sections run in this order, and each one renders only when it has content: header, summary box (Plain terms / Abstract), key figure, three things to know, how it works, where you'd use this, all figures, what sits near this.

![Paper page at 390 px: actions under the title and the summary switch](https://raw.githubusercontent.com/siddhm11/ResearchIT-priv/redesign-images/docs/redesign/img/05-paper-phone.jpg)

## PR 6: paper page v2

PR 6 builds the new page as a separate template, so switching it off brings back today's page, changed only by the versioned static URLs from step 1. It adds no outside call to `GET /p/{id}`: everything slow loads after the page paints.

**Needs first:** the owner's approval to vendor KaTeX (static files, not a pip package).

**Build steps**

1. Fix T15 in its own first commit: a `static_version` global in `app/templates_env.py`, and `?v={{ static_version }}` on every `/static/` URL in `base.html` and the new templates.
2. Add `app/templates/paper_v2.html` and partials in `app/templates/partials/paper/`: `header.html`, `summary_box.html`, `key_figure.html`, `figures.html`, `outline.html`, `actions.html`, plus empty slots for the briefing (PR 7) and the Ask panel (PR 8). `paper_page()` renders `paper_v2.html` when `features.visible("PAPER_PAGE_V2", request)`, otherwise `paper.html`, which stays untouched.
3. The v2 context adds only local reads: the overlaid paper (PR 3), `paper_source_svc.get(id)` (PR 4), the allowed figures with their `figure_url()` and credit (PR 5), and what exists today (related papers, reading density).
4. **Summary box.** Two tabs, "In plain terms" and "Abstract", built as an ARIA tab list (arrow keys, Home and End, roving `tabindex`). The plain panel loads `/api/papers/{id}/explain` with `hx-trigger="load"`. If that returns nothing, script selects Abstract and hides the plain tab. The last choice is remembered in `localStorage` under `researchit-sumtab`, wrapped in try/catch. The abstract panel shows the full abstract, and the "stored truncated" note only while it is still cut.
5. **Figures.** Each figure is a `<figure>` with `<img src width height loading="lazy" decoding="async" alt="<caption>">` inside a box styled with `aspect-ratio`, so nothing jumps when images load. A caption and credit line sit underneath. A button opens a `<dialog>` with `showModal()`: the image, caption, credit, Close, and "Ask about this figure", which stays hidden unless Ask is visible. Esc closes it and focus returns to the button.
6. **Code chip.** Add `fetch_paper_code(arxiv_id)` to `app/hf_papers_svc.py`: `GET https://huggingface.co/api/papers/{id}`, 4 s timeout, reading `githubRepo` and `githubStars`. The chip loads through a new `GET /api/papers/{id}/code` after paint. Results, including "no repo", are saved in the `paper_sources` code columns and rechecked after 7 days. Accept only https://github.com/<owner>/<repo> URLs. Save them by rewriting the whole paper_sources row with INSERT OR REPLACE and every column: a bare UPDATE keeps the row's id, so Turso would never receive the change (T3), and a replace that leaves columns out wipes them (T5). Add a `/code` suffix rule to the rate limiter.
7. **Maths.** Vendor a pinned KaTeX release (one at least two weeks old) into `app/static/vendor/katex/` with its fonts, and load it with `defer` only from `paper_v2.html`. Run auto-render on elements with class `tex`, with `$…$` and `$$…$$` delimiters and `throwOnError: false`, `trust: false`, `maxExpand: 1000`, on load and after each htmx swap into such an element. Jinja has already escaped the text, and KaTeX reads text content, so markup in a formula can't run. Page titles and link previews keep the raw text.
8. **Outline.** Built from the sections actually rendered; an `IntersectionObserver` sets `aria-current` on the section in view. Hidden below 1180 px.
9. **Back link.** "← Your feed" appears only when the URL carries `qid`, meaning the reader came from the feed.
10. **Styles.** Add a `/* paper page v2 */` block to `app/static/styles.css` using existing tokens and fonts. The grid is `12rem minmax(0, 1fr) 23rem` from 1180 px, `minmax(0, 1fr) 23rem` from 900 px, and one column below. The Ask panel is `position: sticky` under the top bar.
11. Update CURRENT-STATE with a Paper page row.

**Tests**

- Route tests: `off` renders today's page unchanged apart from the ?v= on static URLs (compare against a snapshot taken before this PR, with that query stripped); `preview` without the cookie renders today's page and with it renders v2; `on` renders v2.
- With an outside-call trap (a `MockTransport` that fails the test on any request), `GET /p/{id}` with v2 on makes no outbound call.
- A paper with no saved source shows only the header, summary box and nearby papers, and its outline lists only those.
- A paper under the arXiv licence renders no `/fig/` image. An allowed one renders `width`, `height` and the caption as `alt`.
- Escaping: a title, author, abstract, caption and explanation each set to `<script>alert(1)</script>` or `"><img src=x onerror=alert(1)>` come out escaped.
- Code chip: a non-GitHub URL is ignored; a 404 is saved and not refetched for 7 days; a timeout shows nothing.
- Browser tests (Playwright, scratch server, fixture data): the tabs work by mouse and keyboard; an empty explanation selects Abstract; the lightbox opens, closes on Esc and returns focus; a formula renders and the payload `$<img src=x onerror=alert(1)>$` opens no dialog; layout shift stays under 0.1; no horizontal scroll at 390 px; no console errors in light or dark.
- A timing script compares `GET /p/{id}` p95 over 50 local requests for v1 and v2: v2 is at most 30 ms slower.

**Done when**

- The owner has used it in `preview` on the live Space for 3 days across 20 papers: figure-allowed with figures, arXiv licence, no HTML page, one from 2008, and maths-heavy ones.
- Paper-page p95 in `/healthz/runtime` is unchanged, and the owner switches it `on`.

## PR 7: briefing

PR 7 adds "Three things to know", "How it works" and "Where you'd use this" to the paper page, written by a model from the paper's own paragraphs and checked by code before anyone sees them. A briefing that fails the checks is not shown.

**Needs first:** the owner's model choice (Groq or Claude; run the trial below), a monthly budget, and approval for a model SDK if Claude is chosen.

**Build steps**

1. Add `app/llm.py` with one entry point, `complete_json(feature, system, user, *, max_tokens, timeout) -> (dict | None, usage)`. Providers: `groq`, which reuses `groq_svc._complete()` with an optional JSON response format added as a keyword argument, leaving every existing caller unchanged; `anthropic`, only if approved; and `fake`, for tests, which refuses to load unless `RESEARCHIT_ENV=test`. Parse the first JSON object in the reply; anything else counts as a failure.
2. Log every call to a replicated `llm_calls` table (`id` watermark; feature, provider, model, input and output tokens, estimated USD, latency, status, time). A budget guard sums today's estimated spend and refuses new calls past `LLM_DAILY_USD_CAP`. Because the log is replicated, a restart doesn't reset the day's spend.
3. Add `app/briefing_svc.py`. Passage selection: the abstract; the first 4 paragraphs of the introduction; the first 3 paragraphs of sections whose titles mention method, approach, experiment, result, evaluation, limitation, discussion or conclusion; and every figure and table caption. Tag each passage with its anchor and stop at about 24,000 characters (about 6,000 tokens).
4. The prompt asks for this JSON and nothing else: `{"insufficient": false, "things": [{"lead", "text", "anchor"} × 3], "how": [{"name", "text"} × 3–5], "use": "…"}`. The rules: use only the passages, cite one passage anchor per thing, plain text with no LaTeX or markdown, and `insufficient: true` when the passages can't support it.
5. Verify in code (`briefing_svc.verify`), then store only what passes:
    1. Every anchor must be one of the passages sent.
    2. Every number in a thing (`118,494`, `67.3%`, `24K`, `0.5`) must appear in its cited passage after removing commas and spaces. `24K` also matches a number from 24,000 to 24,999. Numbers in "how" and "use" must appear somewhere in the passages.
    3. A thing that fails is dropped. Fewer than 2 surviving things means the whole briefing is stored as `insufficient` and not shown.
    4. Text is cut to 400 characters per item; a longer item counts as a schema failure.
6. Cache in a replicated `paper_briefings` table: `id` watermark, `cache_key TEXT UNIQUE` (a hash of id, paper version, parser version, prompt version, provider and model), `arxiv_id`, `status` (`ok`, `insufficient`, `failed`), `briefing_json`, the model details and usage. Failed rows are retried after a day.
7. Add `GET /api/papers/{id}/briefing`, loaded after paint. It returns the cached briefing if there is one. Otherwise it waits up to 6 s for `paper_source_svc.ensure()`, then generates within a 12 s budget, one generation in flight per paper. Past the budget it returns nothing and lets the generation finish in the background, so the next visit has it. In `shadow` it generates and stores, but returns nothing. Add a `/briefing` suffix rule (20 a minute per IP).
8. Render in `partials/paper/briefing.html`: each thing with an anchor chip linking to `https://arxiv.org/html/{id}{version}#{anchor}`, the steps, and the use line. Label it "Generated from the paper's text. Check the paper before citing it." Add a "Report a problem" link that posts to `/api/papers/{id}/report` and stores the report in a replicated `content_reports` table.
9. Add `scripts/eval_briefings.py`. It runs a committed list of 50 papers (25 recent figure-allowed ones from the September sample, 15 from 2019–2024, 10 maths-heavy) through one or both providers. It writes `reports/briefings/<date>/index.html` showing each briefing, every dropped item and why, tokens, cost and time, plus a `ratings.csv` for the owner to fill in.
10. Add a doc 06 changelog entry for the briefing (display-only, verified, cached per paper), and a CURRENT-STATE row.

**Tests**

- `tests/test_briefing_verify.py`: a number missing from its passage drops that thing; `24K` matches `24,310`; a thousands comma doesn't matter; an anchor that wasn't sent drops the thing; one surviving thing gives `insufficient`; missing keys, wrong types and over-long text give `failed`.
- `tests/test_briefing_svc.py`, with the fake provider: the prompt holds only selected passages, each tagged; `shadow` stores but returns nothing; the budget guard stops calls at the cap and the counter survives a restore; 5 concurrent requests make 1 call; a new prompt version gives a new cache key; model output containing `<script>` renders escaped.
- `tests/test_groq_model_chain.py` still passes unchanged.
- Rate-limit tests for `/briefing` and `/explain`.

**Done when**

- The owner rates at least 45 of the 50 trial briefings good for the chosen provider.
- No displayed briefing contains a number the verifier can't find.
- Median generation time is under 8 s, and cost per briefing is recorded, with a monthly projection under the cap.
