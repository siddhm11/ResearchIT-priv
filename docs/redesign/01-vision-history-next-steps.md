# ResearchIT — Vision, History & Next Steps

2026-10-17 · siddh

ResearchIT already finds good papers across several interests at once. Its next job is helping people **understand and use** a paper without leaving: the plain summary first, then the key figure and three key points, then a path to code.

The one next step: fix the foundations over the next two weeks (a measured baseline, the daily harvest, full abstracts, calmer cards). Then prototype the paper briefing on 50 papers before building it for everyone.

Short answers to your questions:

- **"In plain terms"** is written by an AI model from the abstract alone, and the page labels it that way.
- **PaperCode** is worth borrowing from, not copying. Take its idea of breaking a paper into buildable pieces.
- **Figures** can be pulled from arXiv's HTML version of most new papers, depending on each paper's licence.
- **Beyond papers:** yes. Start with a "builder radar" of papers that come with code.

**See it:** the [ResearchIT Redesign Preview](https://claude.ai/artifact/9KyGoDxgMyD1PBnmkFVM1X) puts today's feed and paper page next to the proposed versions. It uses the app's real stylesheet, real September 2026 papers with their Creative Commons figures, and the live site's own summaries.

## The use case

ResearchIT is a personal feed of research papers, picked for the several things you work on. It swaps ten minutes of reels for ten minutes of ideas you can build on.

The promise in one line: **scroll for ten minutes, leave with one idea you can use.**

Who it serves:

- **Builders.** Engineers and founders who want an idea while it is still a paper, before it becomes a product.
- **Researchers and students trying to keep up.** Doc 01 cites that only about 15% of researchers feel they keep up with their field.
- **Curious people crossing fields.** They need a paper translated into their terms, not dumbed down.

Where people go today, and the gap ResearchIT fills:

| Where people go | What it does well | What's missing |
| --- | --- | --- |
| Reels and Shorts | Fast and effortless | Someone else's take, mostly fluff, rarely the source |
| arXiv listings, Google Scholar | Complete and current | You must already know what to look for; no sense of your taste |
| Single-paper tools (alphaXiv, Semantic Scholar) | Deep help on one paper: chat, summaries, citations | Not a feed built around several interests at once |
| **ResearchIT today** | A feed that keeps each of your interests alive; 1.85M papers; a 3D map | Understanding one paper still means leaving for arXiv |

The last row is the point of this doc. Discovery is strong. The next step is helping people **understand and use** a paper without leaving: figures, key points, explanations and, later, code.

What no one else has, and what to protect:

- **A multi-interest feed.** It finds up to seven separate interests in your saves and guarantees each a share of the feed, so one interest can't swallow the rest.
- **Honest signals.** "Why this?", "Something different" and the reading-density label say what the system knows and nothing more.
- **A map of the field.** The 3D map shows where a paper sits among its neighbours.

## History

ResearchIT went from notebooks in February 2025 to a deployed multi-interest feed over 1.85M papers in October 2026. The current repo has 252 commits since April 2026.

One idea changed along the way. Onboarding began with fixed subject vectors, swung to purely behavioural signals, and settled on a hybrid: picked categories guide the start, then saves take over (docs 05 and 06).

| Date | Milestone | What changed |
| --- | --- | --- |
| 2026-10-16 | Search quality pass (PRs #32–#35) | New search reranker (mxbai-rerank-xsmall), faster word matching, feed policy v11 |
| 2026-10-09 | Personal scoring retuned | Profile similarities rescaled, recency weight 0.45, paper age counted from first publication |
| 2026-10-04 | AI features restored | Fallback chain of Groq models plus a health alert, after a model retirement silenced them for seven weeks |
| 2026-10-01 | Corpus caught up | 1,853,777 papers through 2026-09-29; citations refreshed for 99.47% of them; daily job built but off |
| 2026-09-30 | Multi-interest from two saves | Each early save counts as its own interest; starter feed balanced per interest, with a recent-papers lane |
| 2026-09-28 | Fresh discovery deployed (PR #9) | No repeats for 7 days, editable interests, reading history, starter papers |
| 2026-08-21 | Paper page and "In plain terms" | Each paper gets its own page with an AI explanation, reading-density label and nearby papers |
| 2026-08-16 | Interests get names | Clusters are labelled, so each card shows which interest it serves |
| 2026-08-13 | Collections | Curated reading tracks that also seed the recommender |
| 2026-08-09 | Feed redesign | Infinite feed, a hand-written "editorial but alive" design, undo on Pass |
| 2026-08-02 | Recent-papers shard and 3D map | A shard for papers from 2025-06 on; the 3D map project (researchit-space) starts |
| 2026-07-29 | arXiv ingestion | A pipeline for new papers and the Phase 7 freshness plan |
| 2026-05-02 | Learned reranker and instrumentation | A 37-feature LightGBM model, plus the logging later evaluation needs |
| 2026-04-26 | Onboarding and quota fusion | Category picks, seed papers and a popularity fallback; each interest guaranteed a share of the feed |
| 2026-04-19 | Hybrid semantic search; current repo starts | BGE-M3 dense and sparse retrieval fused with RRF; on Hugging Face Spaces a day later |
| Before 2026-04 | Phases 1–2 | A zero-ML recommender, then the multi-interest engine: taste profiles, Ward clustering, diversity re-ranking |
| 2025-06 to 2025-11 | Research_IT repo | A second iteration of the project |
| 2025-03 | React front end (researchitreact) | Later dropped; React is now ruled out in favour of HTMX |
| 2025-02 | First ResearchIT repo | Mostly Jupyter notebooks; the BGE-M3 encoding of about 1.6M papers came from Kaggle notebooks |

## Your question: where "In plain terms" comes from

Yes, it is AI-generated. A language model writes it live from the paper's title and abstract, and nothing is pulled from another site.

How it gets onto the page:

1. The paper page loads first, with the abstract. Then the page quietly asks the server for an explanation in the background.
2. The server checks a shared cache. The key is the paper ID plus the exact abstract text, the prompt version and the model name. If anyone has opened this paper before, the saved text comes back instantly.
3. On a miss, the title and abstract go to Groq. The model chain is Qwen3.8-27B, then gpt-oss-20b, then gpt-oss-120b, at temperature 0.2, up to 220 tokens, with an 8-second limit.
4. The prompt asks for exactly three sentences: the problem, what the authors did, and what they found and why it matters. It must use only what the title and abstract say, and explain any jargon in passing.
5. If the abstract is too thin, the model answers INSUFFICIENT and the box doesn't appear. It also stays hidden after a timeout or if the API key is missing.

The label "Generated from the abstract" is honest. Four caveats sit behind it:

- **Most papers are summarised from a cut-off abstract.** 90% of stored abstracts stop at 500 characters, about 40% of a typical 1,244-character abstract. Measured in the sidecar today: papers from 2024 average 494 characters, and papers from 2026 (the new ingest) average 1,268.
- **Nothing checks it.** The rules live only in the prompt. Nothing verifies that each number or claim actually appears in the abstract. Doc 07 recommended a checking step, and it hasn't been built.
- **It was silent for seven weeks.** Groq retired Llama 3.3 on 2026-08-16. Every explanation failed quietly until the fallback chain of models shipped on 2026-10-04.
- **It only knows the abstract.** It can't mention a figure, a table or a result the abstract leaves out. That is why the figures idea below needs the full paper.

Two cheap fixes come before any new feature:

- The first time someone opens an older paper, fetch its full abstract from arXiv once and store it. The cache key already includes the abstract text, so a better summary replaces the old one automatically.
- Add a number check: if the summary contains a number that isn't in the source text, regenerate it or hide it. Add a small "Report" link so readers can flag a bad summary.

## Paper to code: what PaperCode is, and what to borrow

PaperCode is LeetCode for ML papers. Borrow its idea of breaking a paper into buildable pieces, not its hand-made problem sets.

("C neural network" in your voice note is **cneuralnets**, PaperCode's co-creator. I opened the site and a task page to check how it works.)

What it is:

- [PaperCode](https://papercode.vercel.app/) is built by [cneuralnets](https://cneuralnets.xyz/) (a CS student and NLP researcher who interned at AI4Bharat and Cisco) and Supritam. Its tagline: "Don't just read the paper. Compile it."
- The free list has 18 classic ML papers, from support vector machines (1995) to Tiny Recursive Reasoning (2025).
- Each paper is a track of small tasks plus one task that assembles them. *Attention Is All You Need* has six: positional encoding, scaled dot-product attention, one attention head, multi-head attention, layer norm, then a tiny Transformer forward pass.
- Each task is a coding problem: the paper's maths, a function signature, pure NumPy, an example and a 2-second limit. You press Run or Submit, and unit tests check the result.
- Free covers the core. [Plus at $5/month](https://papercode.vercel.app/pricing) adds a job feed and a cold-email generator. Pro at $15/month adds a professor email database and specialist roadmaps.
- The home page claims 20+ paid users and 200+ submissions a day. Both numbers are self-reported.

There's real demand for "make the paper runnable". KAIST's [Paper2Code](https://arxiv.org/abs/2504.17192) (ICLR 2026) uses several LLM agents to write a whole repo from one paper. The [Sutskever 30 implementations](https://github.com/pageman/sutskever-30-implementations) repo (5.2k stars) names PaperCode as its inspiration.

| Idea from PaperCode | Take it? | Why |
| --- | --- | --- |
| Break a paper into its 3–6 building blocks | **Yes, as reading rather than exercises** | An LLM can write it for any paper from the full text. It bridges "I read it" and "I can build it" |
| A "Code" link to runnable code | **Yes, now** | The Hugging Face papers API returns a paper's GitHub repo, stars and linked models in one call (checked on Paper2Code: 4,988 stars) |
| Clean maths rendering | **Yes** | Abstracts show raw LaTeX, such as "3$\times$". In a sample of 20,000 recent papers, 39% contain it (18% in AI/ML). Render it properly |
| "Practise this on PaperCode" for the 18 papers it covers | **Yes, after asking them** | Exposure for both sides, with no infrastructure |
| Hand-written problem sets with unit tests | No | Doesn't scale beyond dozens of papers, and we have 1.85M |
| Running people's code in a sandbox | Not now | Security and compute cost on a free 2-vCPU Space |
| Job feed, cold email, professor database | No | Off-mission. Doc 01 notes researchers punish aggressive monetisation (Academia.edu) |
| Generating a whole repo, Paper2Code-style | Later, on demand | Takes minutes and many LLM calls per paper. A labelled "sketch the core idea" button for ML papers is the affordable version |

PaperCode sells one feeling: *I made the paper work.* ResearchIT's version should be: *I understood this paper in two minutes, and I know whether to build on it.* Then hand off to code, whether that's the official repo, PaperCode or a sketch.

The two products fit together: ResearchIT is where you find papers, PaperCode is where you practise them. A two-way link is the natural first step, and it's worth a message to cneuralnets.

## A richer paper page

Make the paper page a two-minute briefing: the plain summary first, then the key figure, three key points and how the method works. It all comes from the full paper, which arXiv already publishes as HTML.

Today the page shows a 250-word abstract, then "In plain terms" under it, then Save. I checked a live page: the easy part sits below the hard part.

Proposed order, top to bottom:

1. **Header.** Title, authors and year, plus a **Code ★ 4,988** chip when a repo exists, PDF and Save.
2. **In plain terms.** Moved above the abstract.
3. **Key figure.** The figure that shows the main result, with its caption.
4. **Three things to know.** Key findings with their numbers, each linked to the section it came from.
5. **How it works.** The 3–6 building blocks of the method, each with one line and its core equation (the idea borrowed from PaperCode).
6. **Where you'd use this.** One concrete scenario, tied to the reader's own interest when it matches ("Connects to your *LLM efficiency* interest"). Labelled as generated.
7. **All figures.** A strip of thumbnails that enlarge on tap.
8. **Abstract.** Collapsed to its first sentence, expandable.
9. **Ask this paper.** Optional and collapsed (see below).
10. **What sits near this.** Unchanged.

![Proposed paper page · mock-up on a real paper](https://raw.githubusercontent.com/siddhm11/ResearchIT-priv/redesign-images/docs/redesign/img/diagram-3.png)

Every key point above comes from this paper's real abstract, and the caption is the paper's own Figure 2. The plain summary would be written from the full text rather than a 500-character stump.

A clickable version, built from a different real paper (SatNav, CC BY 4.0), is in the [ResearchIT Redesign Preview](https://claude.ai/artifact/9KyGoDxgMyD1PBnmkFVM1X).

Why it's feasible:

- **Figures are easy to extract.** I fetched three arXiv HTML pages (2609.01587, 2609.00605 and 1706.03762). Each returned 9–15 figures and tables, every one with a caption, and the images are plain PNG files.
- **Code links are one call.** The Hugging Face papers API returns the repo, stars, and linked models and datasets. The repo already has a Hugging Face papers adapter (`app/hf_papers_svc.py`).
- **Cost scales with papers, not views.** Generate once on first open and cache for everyone, like "In plain terms" today. At doc 07's Haiku prices, a 20,000-token paper costs about 2–3 cents. Full text may exceed Groq's free-tier limits, so measure that first.

Two things to settle first:

- **Figure licences.** arXiv says reuse [depends on each paper's licence](https://info.arxiv.org/help/license/reuse.html). Most papers use arXiv's default licence, which grants no reuse rights. The safe start: show figures inline for Creative Commons papers, and link to them for the rest. The OAI harvest on this branch could store each paper's licence.
- **Older papers.** arXiv's HTML versions are generated from LaTeX source, and coverage of older papers needs measuring. Fall back to today's page when there's no HTML.

On the question-answer bot: skip the general chatbot sidebar. It's costly, generic, and every competitor has one. An **Ask this paper** box is worth it later: collapsed by default, grounded in the full text, and quoting the passage each answer comes from. Starter questions do most of the work: "What's new here?", "Explain figure 2", "What would I need to reproduce this?". Build it after the briefing, because every question costs money, while a briefing is paid for once per paper.

## How figures and "Ask this paper" work

Figures can work for every paper without slowing the app. The server reads each paper's arXiv HTML page once, saves its figures and text, and every later reader gets the saved copy. The feed itself never calls arXiv.

To your question: yes, figures come from arXiv's HTML, but your browser doesn't fetch it when you click. For new papers, the daily harvest does it before anyone sees them. For an older paper, the server does it the first time anyone opens it. The page shows at once and the figures arrive about a second later. After that, the paper is instant for everyone.

![How figures and full text flow · measured on 60 papers](https://raw.githubusercontent.com/siddhm11/ResearchIT-priv/redesign-images/docs/redesign/img/diagram-5.png)

Measured on 60 random papers from September 2026, fetched from arXiv on 2026-10-17:

| Measure | Result |
| --- | --- |
| Has an arXiv HTML version | 54 of 60 (90%) |
| Has real figure images in it | 50 of 60 (83%) |
| Creative Commons and has figures | 25 of 60 (42%) |
| Time to fetch the HTML page | median 0.57 s, slowest 3.5 s |
| Size of the first figure image | median 182 KB, 90th percentile 919 KB, largest 2.1 MB |
| Figures per paper | median 5 |

What the numbers decide:

- **Cards need thumbnails.** Original images reach 2.1 MB, so a feed of eight would be far too heavy. Shrink the key figure once (about 800 px wide) and store it. The paper page loads full-size images only when someone enlarges one.
- **About 4 in 10 new papers get a figure** under the safe policy (Creative Commons only). The rest keep their captions and link to arXiv for the images. Showing figures from the other papers is a legal call for you, not a technical one.
- **Store thumbnails outside the Space.** The Space's disk is wiped on every deploy. A Hugging Face dataset, or another bucket, keeps them across deploys.
- **The daily job stays small.** About 1,000 new papers a day means about 1,000 page fetches, spread through the run.
- **No HTML (10%) means a text-only card.** Figures could later come from the paper's LaTeX source or PDF, but that is much harder.

**Ask this paper**, in plain words: a chat box that answers questions about one paper using only that paper's text. Every answer shows the passage it relied on, with a link to that exact spot on arXiv. arXiv's HTML has a stable anchor for every section, table and figure, and I checked them on SatNav.

How it works, in order:

1. Reuse the HTML fetch above: one request gives the figures and the full text.
2. Split the text into sections and keep each section's anchor (§3.4, Table 1, Figure 4).
3. For a question, send the relevant sections to the model, with instructions to answer only from them and to name the sections it used.
4. Before showing an answer, check that every quote appears word for word in the paper. Drop any that doesn't.
5. Cache the answers to suggested questions for everyone, as "In plain terms" is cached today. Cap free-typed questions per reader per day to control cost.

Open choice: the Groq models (fast and cheap), or Claude with its Citations feature, which returns checked citations itself (doc 07's recommendation). Decide during the 50-paper trial.

On a laptop the layout uses the full width (see the [Redesign Preview](https://claude.ai/artifact/9KyGoDxgMyD1PBnmkFVM1X)):

- **Feed:** a left rail for your interests (click one to filter), refresh and the card setting, with papers in two columns. A phone keeps one column.
- **Paper page:** contents and actions on the left, the reading column in the middle, and Ask this paper on the right. On a phone, Ask opens from a button.
- **Summary box:** switches between "In plain terms" and the full abstract, on every card and on the paper page.

## Less clutter

The design is good; the clutter is in how many small things each card carries. Give each card one job and move the rest one tap away.

A feed card can show up to 14 things today: a "previously discovered" note, the interest label, a category code, a match meter with a raw score, the title, authors, the abstract, "Read more", "Full abstract on arXiv", the ID/year/citations line, Save, Pass, "Why this?" and "See it in 3D". The feed page also stacks about seven rows of controls before the first paper. (Checked against the templates on main and a live paper page.)

| Element | Today | Change |
| --- | --- | --- |
| Match meter and score (e.g. 0.62) | A number readers can't interpret | Remove it from the card; show it inside "Why this?" |
| Category code (cs.LG) | A code, not a word | Drop it when an interest label exists; otherwise show a readable name such as "Machine learning" |
| "Read more" plus "Full abstract on arXiv" | Two stacked links | Lead with the one-line plain summary. One "More" opens the abstract, which ends with the arXiv link |
| "See it in 3D" and "Why this?" | Their own row on every card | Move both into a "⋯" menu, along with Share and the arXiv ID |
| Feed header | About seven rows before the first paper | One line: your interest chips plus a refresh button. Show the progress bar only until 10 saves |
| Visuals | None, every card is text | Add the paper's key figure as a thumbnail (next point) |

The biggest single change is a **figure on the card**. It's what makes a feed feel like Instagram rather than a list. Doc 01 cites that papers with graphical abstracts get about twice the page views. The paper-page work produces the key figure, so cards get it for free.

Keep what's working: the "editorial but alive" look, Save as the one green action, undo on Pass, and honest labels. Streaks and gamification were rejected on 2026-08-13 and stay out.

Three additions that make reading easier rather than busier:

- **A Skim / Detail switch.** Skim shows the title, the plain one-liner and a figure. Detail shows the abstract. The choice is remembered per reader.
- **Reading paths.** Collections become ordered and shareable, with "start here, then this" (doc 01's "paper playlists").
- **Link previews with the key figure.** Shared links have no preview image today (the `og:image` tag is still missing). A figure in every shared link is the cheapest growth loop.

## Beyond papers

Yes, it can grow beyond arXiv. The engine is general: it learns several interests from saves and explains long expert writing in plain words. Anything long-form, expert-written and published daily fits.

The direction I'd bet on is **"Instagram for expert knowledge", research first**. Grow next to papers before growing away from them, because the audience, the trust and the data are all here.

![Other uses · value × effort, my estimate](https://raw.githubusercontent.com/siddhm11/ResearchIT-priv/redesign-images/docs/redesign/img/diagram-1.png)

Start top left. **Builder radar** is closest to your goal of people building creative things: papers that already have code or a demo, ranked for each reader. The Hugging Face papers API supplies the code links. Big bets need accounts (teams, email) or more storage (new sources); Qdrant's free-tier capacity is already a measured limit (PHASE7).

## Paths and future scope

Understanding comes first, code second and growth third. Each phase opens only when the one before has proved itself with real readers.

![Roadmap · 4 phases, 3 gates](https://raw.githubusercontent.com/siddhm11/ResearchIT-priv/redesign-images/docs/redesign/img/diagram-2.png)

The gates matter more than the phases. The instrumentation that records opens, saves and passes already exists, so the "readers use it" gate can be measured rather than guessed. Accounts are the one big missing piece: identity is a cookie today, which blocks email digests, team feeds and syncing across devices.

## Next steps: the next two weeks

Each step repairs one of four foundations that every later feature stands on: fresh, complete papers; generated text you can trust; a calm page with room to grow; and proof that a change helped. Without these, every new feature is built on stale data, unchecked text or guesswork.

![How the next steps feed the app · 4 foundations](https://raw.githubusercontent.com/siddhm11/ResearchIT-priv/redesign-images/docs/redesign/img/diagram-4.png)

Three findings from checking the live system on 2026-10-17 sharpen the order:

- **The live feed is 18 days stale.** Production's newest paper is from 2026-09-29. At September's rate of about 1,030 papers a day, roughly 18,500 papers are missing. An "Instagram for research" with no new posts is the most urgent gap.
- **The harvest already fixes new papers.** arXiv's OAI records carry the full abstract and the licence. The parser on this branch keeps the abstract but drops the licence, a one-field change. Only the 1.4M older papers have cut-off abstracts.
- **Figures will work for about half the feed.** In one day of computer-science records (1,083 papers), 59% were Creative Commons and 41% used arXiv's default licence. Cards must look good with and without a figure.

What changed from the first list: I added a baseline and real readers, merged the two 50-paper prototypes, and moved two items that don't build foundations off the critical path. Each step below says how we'll know it worked.

- [ ] **Measure a baseline first.** Use a read-only script over production interactions: opens and saves per visit, Pass rate, the share of paper-page visits that end in a save, and readers who return within 7 days. *Done when* one dated table of these numbers exists before any UI change ships.
- [ ] **Turn the daily harvest on.** Merge the OAI harvest, keep the licence field, then switch the daily job on. *Done when* /healthz/deep shows yesterday as the newest date every day for a week.
- [ ] **Full abstracts on first open.** Fetch the full abstract when someone opens an older paper and store it in a small local table, not in the 2.7 GB sidecar, so the Docker image doesn't grow. *Done when* a truncated paper shows its full abstract on the next view, and "In plain terms" is regenerated from it.
- [ ] **Summary first on the paper page.** Move "In plain terms" above the abstract and render maths. KaTeX is a new dependency and needs your OK. *Done when* a week's paper-page save rate has been compared with the baseline.
- [ ] **Declutter the card.** Add the "⋯" menu, a single "More" and a one-line feed header. *Done when* a card shows 8 items or fewer by default, and save and Pass rates hold or improve against the baseline.
- [ ] **Add the Code chip.** Use the Hugging Face papers API, cached per paper. *Done when* the chip appears on papers that have a repo, and we know what share of feed papers have code. That number sizes the builder radar.
- [ ] **Prototype on 50 papers.** Build the figure store (one arXiv HTML fetch per paper, saving figures, captions, licence and a key-figure thumbnail), plus the briefing and number check, on 50 Creative Commons papers. *Done when* you've rated all 50 as good or bad, no generated number is missing from its source, cards show figures without any arXiv call at feed time, and cost and latency per paper are recorded.
- [ ] **Laptop layout.** Feed: a left rail and two columns. Paper page: contents, reading column and the Ask panel. Add the Plain terms / Abstract switch to every card. *Done when* 1280–1512 px screens use their width and the phone layout is unchanged.
- [ ] **Ask this paper on the same 50 papers.** Answer from the full text, check quotes and link sections, and cache the answers to suggested questions per paper. *Done when* every quote shown matches the paper word for word and the cost per question is known.
- [ ] **Recruit 5–10 real readers.** *Done when* they've used ResearchIT in two separate weeks, and you've watched two of them use it.

> **Status on 2026-10-21.** Built and live on the Space: the daily harvest with licences (catch-up
> done, 23,098 papers), full abstracts, the paper page with the summary first and maths, the
> decluttered card, the Code chip, the figure store and briefing, the laptop layouts and Ask.
> Each item's *Done when* is a measurement that has not been taken yet: the baseline export,
> a week of save rates, and the owner's ratings of 50 briefings and 80 Ask answers. Recruiting
> readers has not started; in October the only reader was the owner.

Off the critical path: messaging cneuralnets is distribution and can happen anytime. Choosing a second use case waits until the "readers use it" gate passes. Doc 06 changelog entries get written as each decision is made.

The step-by-step build plan for the redesign, with a tracker: the “$lid:f” tab

## Sources

Code and docs were read on GitHub main at commit 9888b1e. Sidecar figures come from the local metadata sidecar (1,853,777 papers), queried on 2026-10-17.

- Code: `app/groq_svc.py` (explain prompt and model chain), `app/routers/paper.py` (explain endpoint), `app/templates/partials/paper_card.html`, `app/templates/paper.html`, `app/config.py` (GROQ_MODELS)
- Docs: `docs/research/01` (vision), `06` (verdict and changelog), `07` (LLM summaries), `docs/phases/PHASE7` §3.2 (abstract truncation), `docs/CURRENT-STATE.md`, git history
- [ResearchIT live paper page](https://siddhm11-researchit.hf.space/p/2609.01587), checked 2026-10-17
- [PaperCode home](https://papercode.vercel.app/), [papers list](https://papercode.vercel.app/papers), [Attention track](https://papercode.vercel.app/papers/attention_is_all_you_need), [pricing](https://papercode.vercel.app/pricing), [about](https://papercode.vercel.app/about)
- [cneuralnets](https://cneuralnets.xyz/)
- [Paper2Code / PaperCoder (arXiv 2504.17192)](https://arxiv.org/abs/2504.17192)
- [Sutskever 30 implementations](https://github.com/pageman/sutskever-30-implementations)
- [arXiv reuse and licences](https://info.arxiv.org/help/license/reuse.html)
- arXiv HTML test pages: [2609.01587](https://arxiv.org/html/2609.01587), [2609.00605](https://arxiv.org/html/2609.00605), [1706.03762](https://arxiv.org/html/1706.03762)
- [Hugging Face papers API, example](https://huggingface.co/api/papers/2504.17192)
