# PR 9–10: Feed

The feed changes last, because it is where a mistake costs the most: it is the surface whose order the quota rules protect and whose clicks every evaluation reads.

## PR 9: feed card v2

PR 9 swaps the feed card for a calmer one with at most 8 things to click, a figure when one is saved and licensed, and a per-card Plain terms / Abstract switch. Order, ids and logging fields stay exactly as they are.

![Proposed feed cards: interest label, title, authors and Code chip, figure, summary switch, Save and Pass](https://raw.githubusercontent.com/siddhm11/ResearchIT-priv/redesign-images/docs/redesign/img/01-feed-laptop.jpg)

**Build steps**

1. Add `partials/paper_card_v2.html`. `rec_page.html` uses it when `features.visible("FEED_CARD_V2", request)`. Search, Library, collections, history and related papers keep today's card until the owner has reviewed the feed.
2. **Card, top to bottom:**
    1. Interest label with its colour dot, or "Something different" for exploration picks.
    2. A "⋯" menu: Why this paper (today's reason text), See it in 3D (when the map is enabled), arXiv page, Copy arXiv id.
    3. Title link, with the same `href` and §3.11 query string as today, including the T1 fix.
    4. Authors (3, then "& N more") · year · Code chip, from saved columns only.
    5. Figure, only when `figure_url()` returns one, in a box sized by `aspect-ratio` and loaded lazily. Clicking opens the shared lightbox.
    6. Text tabs: "In plain terms" appears only when an explanation for the current abstract is already cached, otherwise just the abstract's lead.
    7. Save and Pass, through the unchanged `action_buttons.html` include.
3. **Removed from the card:** the match meter, the foot line (arXiv id and year move to the menu and authors line), and the separate 3D link. Citation counts stay on the paper page.
4. **Keep every hook app.js relies on:** `class="card"`, `id="paper-<id>"`, `data-arxiv-id`, `id="actions-<id>"`, and the hidden `.card-reason` with `id="why-<id>"`, which the menu's "Why this paper" now toggles. Deferred dismiss and top-up both depend on them.
5. **Data, one batch per page, local only.** Add `app/card_data.py` with `enrich(papers)`, called where the router renders a page, not where it caches the pool. It returns copies with the full abstract overlaid (PR 3), saved figure and code columns (PR 4), and the cached explanation looked up by `explain_cache_key(id, abstract)` for each paper, so an explanation written from a stump is never shown beside a full abstract. It never changes order, ids or instrumentation fields, and it never calls arXiv, Hugging Face or a model.
6. **Script, delegated like the existing dismiss handler:** card tabs, the "⋯" menu (one open at a time, Esc and outside clicks close it, focus returns to the button), and the default text preference from `localStorage` key `researchit-cardtext`. Add `[hidden] { display: none !important; }` to the stylesheet; the preview needed it because component styles overrode the attribute.
7. Update CURRENT-STATE with a Feed card row.

**Tests**

- Render today's card and v2 for the same paper: the Save and Pass `hx-vals` JSON and the title `href` are identical.
- A v2 card has at most 8 links and buttons.
- A figure appears only for a saved, allowed figure, with `aspect-ratio` set; an arXiv-licence paper has no `<img>`.
- The plain tab appears only with an explanation cached under the current abstract's key; rendering a feed page never hits `/explain` or the model.
- `enrich()` keeps order and ids for random inputs, makes no outbound call (outside-call trap), and adds under 20 ms for 12 papers against a local database of 10,000 rows.
- Browser: Save, Pass and Undo work on v2 cards; the menu works by keyboard; card tabs switch; no paper repeats across pages; no console errors; no horizontal scroll at 390 px.

**Done when**

- The owner has used it in `preview` for 3 days, then switches it `on`.
- Two weeks later, a rerun of PR 1's report shows save rate and Pass share within the baseline's normal week-to-week range, and feed `build_ms` p95 is unchanged.

## PR 10: wide feed, interest filter, figure warm-up

PR 10 uses laptop width without changing the order the quota rules produce, records which layout each click came from, and prepares figures for new papers before anyone opens them.

**Needs first:** the owner's approval for the new `interactions` column (Stop and ask), and a doc 06 changelog entry agreed with the owner, because the layout touches how the feed's order is shown.

**Build steps**

1. **Two columns from 1180 px, order intact (T17).** `.feed` stays one list in served order. At 1180 px and up it becomes a grid: `grid-template-columns: 1fr 1fr; grid-auto-rows: 4px; grid-auto-flow: row`, never `dense`, which would reorder cards. A `ResizeObserver` on every card sets `grid-row-end: span N` from its height, so the layout fixes itself when a figure loads, a tab switches or the window resizes. `.feed-more` and `.issue-end` span both columns, so infinite scroll keeps appending in place. Below 1180 px, nothing changes.
2. **Left rail from 1180 px,** sticky: the feed header (date, count, Refresh), "Your interests" with a count per interest, "Something different", "Edit interests", and "Cards open on: Plain terms / Abstract". Counts come from a new `data-interest` attribute on each card.
3. **Interest filter, client-side.** Choosing an interest hides the other cards already on the page and shows "Showing N of M · Show all". It never changes the next page request, so ranking is untouched. Those hidden cards were still served, so record the filter: a local, unreplicated `ui_events(user_id, event, value, query_id, created_at)` table, written by `POST /api/ui-event` through `navigator.sendBeacon`, with its own rate-limit rule.
4. **Layout on every interaction (T8).** Add `layout TEXT` to `interactions`, following T8 step by step: local migration, remote `ALTER TABLE` in `ensure_remote_schema()`, then the `cols` list. Script adds `layout=grid2` or `layout=list` to Save, Pass and Remove requests in `htmx:configRequest`, and `lay=` to a card's title link when it is clicked. `events.py` and `paper.py` pass it to `db.log_interaction()`. Feed exposures are logged on the server, which can't see the layout, so CURRENT-STATE says so.
5. **Figure warm-up (T2),** in `app/figure_prewarm.py`, owned by the app lifespan like `DiscoveryRuntime` and switched by `FIGURE_PREWARM`. Every 6 hours it makes one read-only Turso query for papers dated in the last 2 days, figure-allowed licences first (PR 2's `paper_licences`). It skips papers already saved, then runs `paper_source_svc.ensure()` and the first figure's thumbnail for each, through the shared pacer: at 1 request every 2 s, 1,000 papers take about 35 minutes. It stops when the breaker opens, handles at most 1,500 papers a run, holds a file lock so only one runner exists, and reports last run, done, skipped and failed in `/healthz/sources`.
6. Update CURRENT-STATE (Feed layout, Attribution, Persistence rows) and add the agreed doc 06 entry.

**Tests**

- Browser at 1440 px: for every pair of neighbouring cards, the later one's top edge is never above the earlier one's, which is the guarantee that sparse grid placement gives; no two cards overlap; after page 2 loads, the same holds and no paper repeats. At 1179 px there is one column, and at 1180 px two.
- The filter hides the right cards, shows the right counts, restores everything on Show all, sends one UI event, and leaves the next page request unchanged.
- Layout: a save at 1440 px stores `grid2`, a save at 390 px stores `list`, and a click-through stores the `lay` value. With the fake `_execute`: the new column is pushed, the remote `ALTER` runs once, a duplicate-column error is ignored, and any other error still surfaces.
- Warm-up, with a fake Turso and an `httpx.MockTransport`: allowed licences go first, the pacer is respected, the breaker stops the run, saved papers are skipped, an exception doesn't kill the loop, a second runner can't take the lock, and it is off by default.

**Done when**

- The owner has used it in `preview` for 3 days, then switches it `on`.
- For two weeks after, save rate, Pass share and seven-day return hold or improve against the baseline, feed p95 is unchanged, and at least 35% of new papers in the feed show a figure.
