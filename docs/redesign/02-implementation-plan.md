# ResearchIT redesign: implementation plan

2026-10-17 · siddh

Build the redesign as ten pull requests in five milestones. Each one ships behind an on/off switch and is checked against a measured baseline.

The backbone is one new service that reads a paper's arXiv HTML page once and saves what it finds. Figures, the briefing and Ask this paper all build on that saved copy, and the feed never waits on arXiv or a language model.

This file is the summary. The step-by-step build for an agent lives in the files after it: start with [Start here: agent hand-off](03-start-here-agent-handoff.md), then the file for each pull request, the [Testing playbook](09-testing-playbook.md) and [Release and rollback](10-release-and-rollback.md).

## Architecture

One new service, `paper_source_svc`, does the slow work once per paper. Everything readers see reads what it saved.

![Architecture · 4 lanes, 3 new services](https://raw.githubusercontent.com/siddhm11/ResearchIT-priv/redesign-images/docs/redesign/img/diagram-architecture.png)

The main tab's figure-flow diagram shows when the fetch happens: in the daily harvest for new papers, and on first open for older ones.

## Build order

Five milestones, in this order. Each pull request is small enough to review in one sitting and can be switched off without a redeploy. Sizes are my estimates (S under a day, M a few days, L about a week); targets marked *proposed* are mine to agree with you.

| # | Pull request | Main files | Done when | Size | Status |
| --- | --- | --- | --- | --- | --- |
| **M0** | **Foundations** |  |  |  |  |
| 1 | Fix click logging, baseline metrics | `scripts/baseline_metrics.py` (new, read-only), `paper_card.html` | One dated table of opens and saves per visit, Pass rate, paper-page-to-save rate and 7-day returns exists before any UI change ships | S | Not started |
| 2 | Daily harvest on, licence kept | `scripts/ingest_arxiv.py` (this branch), `scripts/daily_refresh.py`, the daily workflow | The live newest-paper date is yesterday, every day for a week | S | In progress |
| 3 | Full abstract on first open | `app/routers/paper.py`, `app/db.py`, `app/arxiv_svc.py` | An older paper shows its full abstract on the next view, and "In plain terms" is regenerated from it | S | Not started |
| **M1** | **Paper source layer** |  |  |  |  |
| 4 | Paper source service | `app/paper_source_svc.py` (new), `app/db.py`, `app/turso_sync.py` | It reads arXiv HTML once per paper and saves the licence, sections with anchors, and figures with captions. On 50 papers its coverage matches the probe, and nothing blocks the event loop | M | Not started |
| 5 | Figure thumbnails | `app/routers/figures.py` (new), Pillow (new dependency) | Images are built only from saved arXiv paths and shown only for CC papers. Each is resized once to 800 px, under 150 KB (*proposed*), and arXiv sees one request per figure, ever | M | Not started |
| **M2** | **Paper page** |  |  |  |  |
| 6 | Paper page v2 | `paper.html` plus new partials, `styles.css`, `app.js`, a `PAPER_PAGE_V2` switch | Three zones, the Plain terms / Abstract switch, key figure and figure grid, Code chip and rendered maths. The page paints before any figure or model call returns, works at 390 px and 1440 px, and today's page comes back when the switch is off | L | Not started |
| 7 | Briefing | `app/briefing_svc.py` (new), the model client | Three things (each with its anchor), how it works and where you'd use it, cached per paper. You rate 45 of 50 papers good (*proposed*), no unsupported number is shown, and the cost per paper is recorded | M | Not started |
| **M3** | **Ask this paper** |  |  |  |  |
| 8 | Ask this paper | `app/ask_svc.py` (new), `/api/papers/{id}/ask`, `app/rate_limit.py`, chat partials | Every quote shown matches the paper word for word. The median answer takes under 5 s (*proposed*), suggested answers are cached, and a daily cap per reader holds | L | Not started |
| **M4** | **Feed** |  |  |  |  |
| 9 | Feed card v2 | `partials/paper_card.html`, `action_buttons.html`, `styles.css` | A card shows 8 items or fewer and a figure only if one is saved. Feed latency is unchanged, and save and Pass rates hold against the baseline | M | Not started |
| 10 | Wide feed, interest filter, warm-up | `index.html`, the recommendations router, `app.js`, `scripts/daily_refresh.py`, interaction logging | Laptops show two columns and infinite scroll still appends correctly. Filtered views are logged as their own surface, and new CC papers have figures before anyone opens them | L | Not started |

**M5, measure and tune:** compare against the baseline, watch five readers use it, and fix what they trip on before adding anything else.

PR 2 shows as in progress because the OAI harvest on this branch already parses full abstracts. It still drops the licence field.

## Data model

Small, valuable rows go to Turso so they survive deploys. Bulky text that is cheap to re-fetch stays local. Images live on disk now and move to a Hugging Face bucket later.

| Store | Holds | Lives in | Why there |
| --- | --- | --- | --- |
| `paper_sources` | Licence, figure list (label, caption, image path, anchor), section list (title, anchor), fetch status and time | Local SQLite, replicated to Turso | The feed reads it without calling arXiv, and it survives deploys |
| `paper_fulltext` | Section text | Local SQLite only | Large, and cheap to rebuild: one fetch, median 0.57 s |
| `paper_abstracts` | Full abstracts for older papers | Local SQLite, replicated to Turso | The 2.7 GB sidecar can't change without a rebuild and redeploy |
| `paper_briefings` | Briefing JSON, model, prompt version, keyed on content like today's explanations | Local SQLite, replicated to Turso | Model output costs money, so it shouldn't be regenerated after every deploy |
| `paper_answers` | Cached answers to suggested questions | Local SQLite, replicated to Turso | Same reason |
| `paper_paragraphs` | Paragraph text with anchors, for Ask and the briefing | Local SQLite | Cheap to rebuild. Ask ranks a paper's paragraphs in plain Python, with no model |
| Thumbnails | One WebP per figure | Space disk cache now, a Hugging Face bucket later | Images don't belong in SQL. The disk is wiped on deploy, so they regenerate lazily |

One cheap fix rides along: `paper_explanations` isn't replicated today, so every deploy throws away all cached "In plain terms" text. Add it to `turso_sync.TABLES`. Turso writes cost about 65 ms a row, which is fine at about 1,000 new rows a day. The table needs an id column first (T7 in the Start here file).

## Decisions you need to make

Seven calls are yours. CLAUDE.md asks for your OK before adding dependencies and before product decisions like the licence policy. Each one blocks a specific pull request.

| Decision | Options | My recommendation | Blocks |
| --- | --- | --- | --- |
| New dependencies | Pillow (thumbnails), KaTeX (maths, bundled into `/static`), maybe the Anthropic SDK | Yes to Pillow and KaTeX. Decide the SDK after the PR 7 trial | PR 5, PR 6 |
| Figure licence policy | Creative Commons only, or every paper with a credit line | CC only for now: 42% of new papers, no legal question | PR 5 |
| Model for the briefing and Ask | The Groq chain already running (free, but full papers may exceed its limits), or Claude Haiku with its Citations feature (doc 07's pick, about 2–3 cents per briefing) | Try both on the same 50 papers in PR 7 and choose on quality and cost | PR 7, PR 8 |
| Monthly budget | Model calls plus any storage | Set a cap before PR 7. Usage is cached per paper, so cost tracks new papers rather than readers | PR 7 |
| Rollout | Feature switches as Space variables, like `GROQ_MODELS` | One switch per feature, so a bad change is undone without a rebuild | All |
| Briefing and Ask for papers under arXiv's default licence | Allow them (short quotes with links; full text kept only on our server), or offer them only for Creative Commons papers | Allow: quotes stay under 200 characters and every answer links to arXiv | PR 7, PR 8 |
| Keep the text of Ask questions | Counts only, or question text for 30 days to improve answers | Counts only at launch; revisit after the evaluation | PR 8 |

## Guardrails

None of this changes how papers are ranked. Figures, briefings and Ask are display-only. These are the existing rules the work has to respect:

- **Quota sets the served order (§3.1).** The two-column layout places cards left, right, left, right in served order, so the interleaving of interests survives. Never group a column by interest.
- **Logging stays honest (§3.11).** In a grid, position affects clicks differently than in a list. Log the layout (list or grid) with every interaction, and log interest-filtered views as their own surface, so later evaluation never mixes them.
- **Nothing blocks the event loop (§3.6b).** Fetching, HTML parsing, Pillow and SQLite all run in `asyncio.to_thread`. Background work goes through `events._spawn`. Extend `tests/test_event_loop_blocking.py` to cover the new paths.
- **Escape everything.** Model output and arXiv captions are untrusted text. The AI overview XSS fixed on 2026-10-04 came from rendering model output unescaped.
- **No open image proxy.** The thumbnail route builds arXiv URLs only from saved paths, never from a URL in the request, and fetches only from arxiv.org.
- **Rate-limit the model routes.** `app/rate_limit.py` matches path prefixes only, so add suffix rules for `/ask`, `/briefing`, `/explain` and `/code` (T9), plus a daily cap per reader.
- **No model in the ranking path (§3.4).** The feed's scoring is untouched. A cross-encoder or language model in recommendations stays out of scope.
- **Tests and docs.** Unit tests for the parser (fixtures cut from CC papers) and an integration test per endpoint (§5.4). Add doc 06 changelog entries for the licence policy, the briefing and Ask, and the layout change, and update `CURRENT-STATE.md` per feature.
- **Deploys stay deliberate.** Merge without squash, and deploy with `git push hf main` only after a green PR.

## Risks

The biggest risk is arXiv throttling us; the rest are handled by caching and checks.

| Risk | What it would look like | How it's handled |
| --- | --- | --- |
| arXiv throttles our requests | Missing figures, slow first opens. The ingest already hit 429s on 2026-10-01 | Fetch once per paper and cache. The daily job goes at a polite pace and backs off on `Retry-After`. A circuit breaker hides figures rather than slowing pages |
| A deploy wipes the Space disk | Thumbnails and full text vanish | Figure lists and model output are replicated to Turso. Thumbnails and text rebuild lazily on next view |
| Full papers exceed the model's limits | Briefings or answers fail on long papers | Send only the relevant paragraphs, picked without a model so search never waits (T10). Measure in PR 7, with the model fallback chain behind it |
| Wrong or invented numbers | A briefing or answer claims what the paper doesn't | Numbers are checked, quotes must match exactly, generated text is labelled, and every piece has a Report link |
| Older papers look worse | No figures (before 2011 none of our sample is CC) and cut-off abstracts | Cards are designed text-first. Full abstracts load on first open, and figures are a bonus |
| Gaps in the two-column feed | Figure cards are taller than text cards | Place each new card in the shorter column after each page loads, with a fixed figure aspect ratio |
| Slower pages on a 2-vCPU Space | Feed or paper page feels heavier | Nothing new sits on the critical path. Figures, briefing and chat load after the page paints, and thumbnails are cached |
| Licence misuse | A non-CC figure appears | The CC check lives in code, not in templates. Every figure carries a credit line, and the policy goes in doc 06 |

## How we'll know it works

The redesign works if readers save more and come back more, without the feed getting slower. PR 1 fills the "Before" column; targets are proposals to agree on.

| Number | Before | Target (*proposed*) | Where it comes from |
| --- | --- | --- | --- |
| Paper-page visits that end in a save | From PR 1 | Up | Interactions table |
| Feed save rate and Pass rate | From PR 1 | Hold or rise | Interactions table |
| Readers who return within 7 days | From PR 1 | Up | Interactions table |
| Feed response time, median and slowest 5% | Measure in PR 1 | No worse | Server timings |
| New papers whose card shows a figure | 0% | About 40% | `paper_sources` |
| Briefings you rate good, out of 50 | None yet | 45 | PR 7 review |
| Quotes shown that fail the exact-match check | Not built yet | 0 | Ask logs |
| Model spend per day | About zero (Groq free tier) | Under the cap you set | Provider dashboard |
| arXiv fetch failures per day | Not built yet | Under 1%, with no visible slowdown | Service logs |

With few readers, these numbers move slowly. The five-reader sessions in M5 will show problems sooner than the charts will.

## Evidence

Everything above rests on checks run on 2026-10-17. Papers were fetched from arXiv with a pause between requests.

| Sample | Has arXiv HTML | Has figures | CC licence and figures |
| --- | --- | --- | --- |
| 60 random new papers (September 2026) | 54 (90%) | 50 (83%) | 25 (42%) |
| 30 random corpus papers from 2019–2024 | 28 (93%) | 23 (77%) | 11 (37%) |
| 24 corpus papers from 2007–2010 | 22 (92%) | 17 (71%) | 0 |

- **Speed and size (new papers):** HTML fetch median 0.57 s, slowest 3.5 s. First-figure images have a median of 182 KB, a 90th percentile of 919 KB and a maximum of 2.1 MB, with a median of 5 figures per paper.
- **Every arXiv HTML page states its licence** in the page itself, so one fetch answers the licence question even for papers older than the OAI harvest.
- **Stable anchors exist** for sections, tables and figures. I checked `#S3.SS4`, `#S5.T1` and `#S4.F4` on SatNav, which is what makes Ask's citations clickable.
- **Some older HTML pages have fewer images than figures** (for example, 7 figures and 1 image), so figure coverage per paper is lower for older work.
- **The live site** has 1,853,777 papers with the newest from 2026-09-29. About 90% of pre-2025 abstracts are cut at 500 characters.
- **The codebase:** Turso replicates 4 user tables, and `paper_explanations` isn't one of them. Pillow, BeautifulSoup, KaTeX and the Anthropic SDK aren't in `constraints.txt`. `app/rate_limit.py` matches path prefixes only, so per-paper routes like Ask need a suffix rule (T9).
