# AEO plan — getting ResearchIT found, quoted and used by AI answer engines

> **Status: active.** Written 2026-11-24 from a live audit of the site and four
> research passes. Read this before touching anything listed under "Where it
> lives". Update the status column and the log at the bottom in the same
> commit as the work. The longer background doc is the AEO Playbook artifact
> (claude.ai/artifact/6kMkEwbWhBdALfynUns6Gx, 22 Nov); where they disagree,
> this file wins, because it has the 24 Nov evidence.

## 1. The goal, in one paragraph

When someone asks ChatGPT, Claude, Perplexity, Gemini or Copilot a question
that ResearchIT answers better than anyone — *which AI research topics are
rising, which arXiv papers Hacker News keeps returning to, what to read on
late-interaction retrieval, what is a good personalised arXiv feed* — the
answer cites a ResearchIT page or names ResearchIT, and the few people who
click become readers. Traffic from AI answers is small everywhere (0.2–1% of
visits in 2026 panels); being named is the prize, and the visitors who do
click are worth more than search visitors.

## 2. What the evidence says decides it (read before proposing tactics)

1. **Being retrieved is most of the game; page polish is a small part.** In
   Chen & Koudas (U Toronto, arXiv 2610.15033, 158k answers), ~32% of the
   variance in being cited was whether a page got retrieved at all, against
   2.8% for page features. For a brand the model does not know, a cited
   *third-party* page naming it raised the odds of being named **39.7× in
   ChatGPT, 17.7× in Gemini**. Ahrefs (75k brands): branded mentions correlate
   0.66 with AI visibility, backlinks 0.22.
   → Other sites talking about ResearchIT matters more than anything on our pages.
2. **What on-page work still helps:** a paragraph that directly answers the
   question people ask (≈+9.5%/SD in ChatGPT), a title matching the query
   (Gemini, Perplexity), and visible, true dates (older pages cited less;
   recency helps on time-sensitive queries). Primary data gets ~3.3× more
   citations **when framed as a ranking or comparison** (Search Engine Land,
   Jul 2026).
3. **What does not help:** schema.org / FAQ blocks showed no within-site
   citation effect (keep JSON-LD for Google rich results and Dataset Search,
   not as a citation lever); the GEO-paper "+40%" rewriting tricks do not
   survive a full retrieval pipeline; llms.txt gets no requests; Wikipedia /
   Wikidata entries need independent coverage first.
4. **Live search, not model memory, decides for a new product.** Google Search
   supplied 45% of brands in grounded Gemini answers; ChatGPT named <1 in 10
   brands before searching and then checks shortlisted tools' own sites.
   → The site must state its facts plainly (/about), and must be indexed.
5. **Where we can win** (competitor sweep, 24 Nov):
   - *Trends*: nobody publishes a recurring, dated, significance-tested report.
     "hugging face daily papers trends" has **no** analysis page at all. Top
     results are MIT Tech Review's undated editorial list and one blog post
     from March. **Our strongest opening.**
   - *HN record*: only a stale top-100 list (facts.dev) and one blog post
     (Dylan Castillo, Apr 2026). Nobody measures lasting attention.
   - *Reading lists*: no ColBERT / late-interaction reading list exists; the
     Stanford RAG list has no notes. Annotated lists win.
   - *Tool questions* ("Scholar Inbox alternative", "alphaXiv alternative"):
     results are spam directories or 2019–2023 posts. An honest dated
     comparison can become the cited page.
   - *Single papers*: arXiv and HF own them; we can be cited only for what
     they lack (summary, HN record, neighbours).

## 3. Rules — do not break these

- **No cloaking.** Crawlers and answer engines get exactly the HTML a person
  gets. `app/bots.py` only decides what we *record* and *spend*, never what we
  *show*. Never serve bots different text, Markdown copies, or "AI-only" pages.
- **Bots leave no trace in reader data** (CLAUDE.md §3.11): no cookie, no
  reader state, no click/view/exposure rows, no reader-day counts. They are
  counted only as `bot:<class>:<name>:<page>` in `ops_daily`.
- **Bots never cause a model call.** Cached text or nothing.
- **Every number on a public page is computed from data, never typed in**, and
  trend sentences are written by code from the numbers, never by an LLM.
- **One address per thing.** Papers live at `/p/{id}` (no version suffix);
  every alias 301s there. Canonicals, `og:url`, the sitemap and JSON-LD are all
  built from `config.PUBLIC_URL` and must agree.
- **Never claim more than the data shows.** "Rising" only when the permutation
  test passes; HN counts are "linked", never "recommended".
- **No fake independence.** No self-written "best tools" lists posing as third
  parties, no sock-puppet posts or reviews, no asking for upvotes. The
  comparison page is ours and says so, and says where others are better.
- **Do not sitemap all 2M papers**; only pages that carry something of ours.
- **robots.txt keeps one `*` group.** A per-bot group must repeat every Disallow.
- **Leave AI opt-out signals unset** (Bing/Google training and grounding
  controls); being in answers is the goal.

## 4. The plan

Legend: ✅ done · 🔧 code, Claude can do · 👤 owner only (accounts, money, posting)

### Phase 0 — Safe to be found (branch `feat/aeo`, 24 Nov) ✅

| Item | Where |
|---|---|
| ✅ `app/bots.py`: classes search / user_fetch / training / preview / other | `app/bots.py`, `tests/test_bots.py` |
| ✅ Bots: no cookie, state, clicks, views or exposures; no model calls | `routers/paper.py`, `main.py home()`, `routers/welcome.py`, `routers/collections.py` |
| ✅ Bot hits and AI referrals counted durably (`ops_daily`, replicated) | `analytics.observe` |
| ✅ People get a reader id on /trending, Lasting and topic pages (first click now matches an exposure) | `routers/trending.py _new_reader` |
| ✅ Cached plain-language summary in the HTML under the title; placeholder no longer page text | `paper_v2.html`, `paper.cached_explanation` |
| ✅ 301s: `/p/{id}vN`, `/p/abs/{id}`, `/abs/{id}`, `/paper/{id}`; `/pdf/{id}` → arXiv PDF | `paper.canonical_arxiv_id`, `seo.py` |
| ✅ HEAD answered like GET (was 405) | `main._head_as_get` |
| ✅ 404 pages: `noindex`, no canonical | `not_found.html`, `base.html` |
| ✅ Answer-first H1 / lead sentence / meta description on /trending, topic pages, Lasting, collections | `trending.py trend_sentence, trending_lead, _lasting_seo`; templates |
| ✅ JSON-LD: ScholarlyArticle (paper), Dataset (/trending, Lasting), ItemList (collections), WebApplication (/about) | `base.html` renders `jsonld` |
| ✅ Footer + front-page links to Trending, Lasting, Collections, About, Privacy | `base.html`, `welcome.html` |
| ✅ /about (facts an assistant checks), /privacy (required by both agent directories), /compare | `routers/pages.py` |
| ✅ Review fixes: "CUBOT"/HeadlessChrome are people; crawlers queue no arXiv fetches; redirects keep the query string; HEAD is a pure ASGI middleware (`main.HeadAsGet`) and never logs; a bot with a cookie writes no exposures; sitemap ids canonical | see log |
| ✅ Sitemap: topic pages, /about, /privacy, true `lastmod` only | `seo.entries` |
| ✅ Search Console / Bing verification tags and IndexNow key file, off until env set | `config.py`, `base.html`, `seo.indexnow_key` |

**Exit check** (run after deploy; all must hold):
```bash
H=https://<host>
curl -s -A GPTBot $H/p/2005.11401 | grep -c '>Writing a plain-language'      # 0
curl -s -A GPTBot -D - -o /dev/null $H/p/2005.11401 | grep -ci set-cookie     # 0
curl -s -o /dev/null -w "%{http_code}\n" $H/p/2005.11401v2                     # 301
curl -sI $H/ | head -1                                                          # 200
curl -s $H/about | grep -c 'free reading feed'                                  # >=1
curl -sI $H/ | grep -i '^link:'      # Hugging Face's canonical header — see Phase 1
```

### Phase 1 — Own the address (BLOCKER for indexing) 👤 + 🔧

**Why it blocks:** every response from `*.hf.space` carries an HTTP header
added by Hugging Face's proxy, `Link: <https://huggingface.co/spaces/siddhm11/ResearchIT>; rel="canonical"`
(verified 24 Nov; it is appended even when the app sends its own — HF forum
thread 177658). Our HTML says each page is its own canonical. Google treats
conflicting canonicals as a hint it may ignore, and the worst case folds every
page into the Space's iframe wrapper. Until this is gone, Search Console
submissions may index nothing.

1. 👤 Decide the name (Archivore recommended in the playbook) and register the
   domain (archivore.org ≈ $8 first year).
2. 👤 Hugging Face PRO ($9/mo) → Space settings → Custom domain → `www.<domain>`;
   DNS `www` CNAME → `hf.space`, DNS-only (not proxied).
3. 👤 **Verify:** `curl -sI https://www.<domain>/ | grep -i '^link:'`.
   - Header gone → continue.
   - Header still there → put Cloudflare in front (proxied) with a Response
     Header Transform Rule removing `Link`, or move to the Phase-7 VPS.
     Do not launch until this check passes.
4. 🔧 Set `PUBLIC_URL=https://www.<domain>`; add middleware 301 from the
   hf.space host to the same path on the new host, **except** `/healthz*`
   (keep-alive) and `/mcp` (agents; 308 later if ever). Rename brand strings
   (`APP_TITLE`, masthead, `og:site_name`) if renaming.
5. 🔧 Bare domain → 301 to `www`.

### Phase 2 — Get registered 👤 + 🔧

1. 👤 Google Search Console: verify (set `GOOGLE_SITE_VERIFICATION`), submit
   `/sitemap.xml`. Inspect `/trending/lasting`, `/trending`, one paper.
2. 👤 Bing Webmaster Tools: verify (`BING_SITE_VERIFICATION`), submit sitemap,
   turn on **AI Performance** (the only first-party report of AI citations:
   Copilot citations, grounding queries, cited URLs).
3. 👤 Generate an IndexNow key (`openssl rand -hex 16`), set `INDEXNOW_KEY`.
   ✅ `app/indexnow.py` then pings the sitemap URLs dated today, once a day,
   30 minutes after start (counts in `ops_daily` as `indexnow:<status>`).
4. 👤 Brave Search: submit the site (Claude's web search has used Brave).
5. 👤 Rewrite the HF Space card: `short_description` with the one-line pitch
   and 2.0M count, tags `arxiv`, `papers`, `recommender-system`, `research`,
   `semantic-search`, link to the live site.

### Phase 3 — Pages built to be quoted 🔧

Ordered by expected citations (section 2, point 5).

1. **Monthly trend reports at permanent addresses** — `/trending/2026-11`,
   frozen when the month closes, never recomputed. H1 with the month; first
   sentence a number ("X rose from 1 in 29 to 1 in 18 papers"); table of
   rising and cooling topics with shares, change and q-value; chart; "how
   measured" box; `dateModified`; CSV at the same address + `.csv`. Needs a
   `trend_snapshots` table (Turso) and **stable topic slugs** (persisted per
   topic; name ties broken by the topic's other phrases; `/trending/topics/{id}`
   301s to `/topics/{slug}`). This is the page no one else has.
2. **Evergreen question pages per topic** — the topic page already answers
   "is X growing?" in its first sentence (done); once slugs exist, title them
   `Is {topic} research growing?`.
3. **Lasting as a ranking + data** — a numbered top-50 table (paper, people,
   separate months, first/last link) on the unfiltered page; `/trending/lasting.csv`;
   an RSS feed of papers newly reaching the bar.
4. **Reading lists people search for** — ✅ *ColBERT & Late-Interaction
   Retrieval* (`data/collections/late-interaction.json`, 9 papers, ids checked
   against the corpus; ColBERT and ColBERTv2 moved there from RAG & Retrieval,
   since an anchor lives in one list). ✅ Collection titles and descriptions
   read "<Topic>: a reading list". 🔧 Still to do: an "updated" date.
5. ✅ **/compare** (`routers/pages.py COMPARE`, checked 24 Nov) — dated, honest table: Scholar Inbox, alphaXiv,
   arxiv-sanity-lite, Semantic Scholar feeds, HF Daily Papers, Paper Espresso,
   ResearchRabbit, zotero-arxiv-daily, ResearchIT; columns free?, sign-up?,
   coverage, how it learns, delivery, best at. Every cell sourced; say where
   each beats us. Re-check quarterly.
6. **Clean links** (cleanup, not a blocker now that bots are classified): move
   `qid/pos/src/prop/pol` out of hrefs into `data-` attributes sent by
   `sendBeacon`.

### Phase 4 — Get named where answers come from 👤 (Claude drafts)

The single biggest lever (section 2, point 1). One finding per channel.

1. Listings (≈2 h, free only): AlternativeTo (arxiv-sanity, Semantic Scholar
   pages), PR to `artnitolog/awesome-arxiv` (Notifications & Recommenders),
   `awesome-ai-research-tools`, AIChief submit form. Skip paid directories.
2. **Hacker News**: submit `/trending/lasting` as a regular story titled after
   the finding (not Show HN — lists are excluded there); first comment gives
   the rule and data. Before posting: prewarm Lasting summaries, cache
   cookie-less HTML for a few minutes, load-test.
3. r/MachineLearning `[P]` post on the trend method with the chart.
4. Hugging Face: dataset of weekly topic shares + Lasting (card links back);
   a community article on the method.
5. **A 2-minute YouTube demo** (YouTube is 16% of Perplexity citations, 9.5%
   of AI Overviews).
6. People who asked our question in public: email Dylan Castillo ("LLM research
   on Hacker News is drying up"); university library AI-tool guides (UCSB,
   CSUSB, Queen's, USD) by email to the librarian.
7. Wikidata item only after 1–2 independent write-ups exist.

### Phase 5 — Be called by agents 🔧 (after Phase 1, so the URL is final)

Read-only MCP server at `/mcp` (official `mcp` SDK, needs owner OK for the
dependency), no sign-in, ≤7 tools, each with a title and `readOnlyHint`:
`rising_topics`, `topic_trend`, `hn_lasting`, `paper_hn_record`,
`reading_list`, `similar_papers`, `search_papers` (rate-limited). Every result
carries the `/p/{id}` URL and title. Then: official MCP Registry
(`server.json`), Claude directory (Community listing; needs /privacy ✅), ChatGPT
apps (identity verification 👤). Stage 2 (OAuth, `my_feed`, `save_paper`) only
after there are readers to serve. Agent calls get their own `policy_id` and
never count as feed clicks.

### Phase 6 — Measure 🔧 + 👤

- `ops_daily` rows `bot:<class>:<name>:<page>`, `bot_status:<class>:<code>`,
  `ai_ref:<engine>:<page>` (live once Phase 0 deploys). 🔧 Add them to
  `/admin` as a weekly table.
- Bing AI Performance and Search Console (Generative AI report if the site
  qualifies).
- Prompt panel, monthly: the 10 prompts in the playbook, **each run 5–10 times
  per engine**, record the rate ResearchIT is cited or named (single runs are
  noise: ~65% of cited sources change day to day).

## 5. Targets for 31 Dec 2026 (and what would change the plan)

- Header check passes on the final domain; Search Console and Bing list
  `/trending` and `/trending/lasting` as indexed.
- ≥1 AI user-fetch (`ChatGPT-User`, `Claude-User`, `Perplexity-User`) a week
  in `ops_daily`.
- Listed on AlternativeTo and awesome-arxiv; ≥3 independent pages name ResearchIT.
- First monthly trend report live (`/trending/2026-12`, or 2026-11 backfilled).
- 100 outside readers in December, counted after the bot fix.

If six weeks after submission most paper URLs are "Crawled – currently not
indexed", stop adding paper URLs and put the effort into trend, Lasting and
collection pages. If the HN post sinks, lead with a different finding on a
different channel, then retry once.

## 6. Owner checklist (things only you can do)

- [ ] Name + domain + HF PRO custom domain; run the Link-header check
- [ ] Search Console + Bing (+ AI Performance) + IndexNow key + Brave
- [ ] Set `FIGURE_LICENCES=CC-BY-4.0,CC-BY-SA-4.0,CC0-1.0` before any launch post
- [ ] Set `CONTACT_URL` to an address you read (privacy requests go there)
- [ ] Space card rewrite
- [ ] Listings, HN post, Reddit, HF dataset/article, YouTube demo, emails
- [ ] OK the `mcp` dependency when Phase 5 starts

## 7. Log

- **2026-11-24** — Independent review of the Phase 0 diff found 7 defects and
  4 overstated privacy/about claims; all fixed with regression tests
  (`tests/test_bots.py`, 64 tests; full suite 1,873 passed). Added /compare,
  the late-interaction reading list and the IndexNow pinger.
- **2026-11-24** — Plan written. Phase 0 built on `feat/aeo` (not yet merged
  or deployed). Found the hf.space canonical header; promoted the custom
  domain from "nice to have" to the blocker.
