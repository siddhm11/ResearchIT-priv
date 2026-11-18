"""
/trending/lasting, redesigned (doc 06, 2026-11-17): the lifeline, the story,
the quotes, "Back on HN this week", the Comebacks and Just joined views, and
the chart's data.

What must hold:
- the lifeline puts every paper on one axis (months since January 2007) and
  one height scale, so rows compare;
- a quote is cut word for word from a comment that links the paper, links
  removed, and is shown only when it explains or praises it: a bibliography
  line, a rude comment or the title alone is never shown, and no quote is
  better than a bad one; a missing comments table costs the quotes, not the page;
- a comeback is a link in the last 30 days after a year or more with none; a
  paper "joins" in its third month with someone new;
- "Back on HN this week" carries the §3.11 fields under its own policy and
  query_id, and its exposures are recorded with the list's;
- the chart's data is every listed lasting paper, stacked by first link, and
  its endpoint answers only with the switch on.
"""
import json
import sqlite3
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app import db, hn_svc, qdrant_svc, trending_shelf as ts, turso_svc, user_state as us
from tests.test_hn_svc import (DAY, META, NOW, ROWS, THREADS, VECTORS, client, outside_calls,  # noqa: F401
                               ts_of, write_tables)

# A comeback: lasting in 2013-14, then nothing until three days ago.
COMEBACK = [("1312.00001", f"80{i}", "comment", f"old{i}", ts_of(2013 + i // 2, 1 + 5 * (i % 2)), "950")
            for i in range(3)] + [("1312.00001", "810", "comment", "back", NOW - 3 * DAY, "951")]
# Joined lately: three people in three different recent months, one more yesterday.
JOINED = [("2607.00001", "820", "comment", "j1", NOW - 80 * DAY, "952"),
          ("2607.00001", "821", "comment", "j2", NOW - 50 * DAY, "952"),
          ("2607.00001", "822", "story", "j3", NOW - 20 * DAY, "953"),
          ("2607.00001", "823", "comment", "j4", NOW - 1 * DAY, "952")]
EXTRA_THREADS = {"950": ("Old thread about ideas", 60, 40, ts_of(2013, 1)),
                 "951": ("Ask HN: what are you reading?", 310, 200, NOW - 3 * DAY),
                 "952": ("Small models", 80, 50, NOW - 80 * DAY),
                 "953": ("Joined paper (arxiv.org)", 120, 30, NOW - 20 * DAY)}
META2 = {**META,
         "1312.00001": {"arxiv_id": "1312.00001", "title": "A Paper That Came Back", "authors": '["C. Author"]',
                        "year": 2013, "arxiv_categories": "math.CO", "citation_count": 3},
         "2607.00001": {"arxiv_id": "2607.00001", "title": "A Paper That Just Joined", "authors": '["D. Author"]',
                        "year": 2026, "arxiv_categories": "cs.LG", "citation_count": 0}}

GOOD = ("If you want to understand transformers, I highly recommend the original paper "
        "https://arxiv.org/abs/1706.03762 which shows that attention alone beats recurrent models on translation.")
BIBLIO = "Vaswani et al., Attention is all you need, arXiv preprint arXiv:1706.03762 (2017)."
RUDE = "Only an idiot would not have read https://arxiv.org/abs/1706.03762 by now, it is the whole field basically."
COMMENTS = [("101", "900", GOOD, 1), ("102", "901", BIBLIO, 1), ("103", "901", RUDE, 1)]


def write_comments(path, rows=COMMENTS):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE hn_comments (item_id TEXT PRIMARY KEY, story_id TEXT NOT NULL, text TEXT NOT NULL, "
                "papers INTEGER NOT NULL, checked_on TEXT NOT NULL)")
    con.executemany("INSERT INTO hn_comments VALUES (?, ?, ?, ?, 'd')", rows)
    con.executemany("INSERT INTO hn_threads VALUES (?, ?, '', ?, ?, ?, 'd')",
                    [(s, *v) for s, v in EXTRA_THREADS.items()])
    con.commit()
    con.close()


@pytest.fixture
async def world2(monkeypatch, tmp_path, outside_calls):  # noqa: F811
    path = tmp_path / "hn.sqlite"
    write_tables(path, rows=ROWS + COMEBACK + JOINED)
    write_comments(path)
    monkeypatch.setenv("HN_SQLITE", str(path))
    monkeypatch.setenv("HN_LASTING", "on")
    monkeypatch.setattr(hn_svc, "_state", {**hn_svc._state, "papers": {}, "threads": {}, "canon": [],
                                           "canon_rank": {}, "fresh": [], "meta": {}, "vectors": {},
                                           "loaded_at": None, "rows": 0, "cards": [], "quotes": {}})
    monkeypatch.setattr(turso_svc, "fetch_metadata_batch",
                        AsyncMock(side_effect=lambda ids: {i: META2[i] for i in ids if i in META2}))
    monkeypatch.setattr(qdrant_svc, "get_paper_vectors",
                        AsyncMock(side_effect=lambda ids, remember=True: {i: VECTORS[i] for i in ids if i in VECTORS}))
    await db.init_db()
    us._cache.clear()
    await hn_svc.refresh()
    yield path
    us._cache.clear()


def _agg(rows):
    threads = {s: {"title": v[0], "points": v[1], "comments": v[2], "created_at": v[3]}
               for s, v in {**THREADS, **EXTRA_THREADS}.items()}
    return hn_svc.aggregate(list(rows), threads, NOW), threads


# ── Counting: months, joining, this week, comebacks ──────────────────────────

def test_new_people_are_counted_per_month_and_a_paper_joins_in_its_third_such_month():
    papers, _ = _agg(ROWS)
    a = papers["1706.03762"]
    assert a["voices"] == {hn_svc.month_index(2017, 6): 1, hn_svc.month_index(2017, 9): 1,
                           hn_svc.month_index(2019, 1): 1, hn_svc.month_index(2019, 3): 1}
    assert a["joined"] == ts_of(2019, 1)                     # cy, the third month with someone new
    # One account every month adds one month and never makes the paper join.
    assert papers["2001.08361"]["joined"] == 0
    assert hn_svc.month_index(2007, 1) == 0 and hn_svc.month_index(2026, 11) == 238


def test_a_comeback_is_a_link_lately_after_a_year_or_more_without_one():
    papers, _ = _agg(ROWS + COMEBACK + JOINED)
    back, joined = papers["1312.00001"], papers["2607.00001"]
    assert back["quiet_days"] >= 12 * 365 and back["week_people"] == 1 and back["week_story"] == "951"
    assert 0 < joined["quiet_days"] < hn_svc.COMEBACK_DAYS        # linked a month before: not a comeback
    assert joined["week_people"] == 1 and joined["joined"] == NOW - 20 * DAY
    assert papers["1706.03762"]["quiet_days"] == 0 and papers["1706.03762"]["week_people"] == 0


# ── The lifeline and the story ───────────────────────────────────────────────

def test_the_lifeline_shares_one_axis_and_one_scale():
    papers, _ = _agg(ROWS)
    now_month = hn_svc.month_index(2026, 11)
    life = hn_svc.lifeline(papers["1706.03762"], now_month)
    assert life["months"] == now_month + 1
    assert [t["i"] for t in life["ticks"]] == sorted(papers["1706.03762"]["voices"])
    assert life["first"] == hn_svc.month_index(2017, 6) and life["last"] == hn_svc.month_index(2019, 4)
    # The scale is fixed, not the paper's own busiest month: one new person is
    # the same height on every row; LIFE_FULL or more is full height.
    one = {**papers["1706.03762"], "voices": {5: 1, 6: hn_svc.LIFE_FULL, 7: 40}}
    hs = [t["h"] for t in hn_svc.lifeline(one, now_month)["ticks"]]
    assert hs == [41, 100, 100] and [t["h"] for t in life["ticks"]] == [41] * 4


def test_the_story_is_first_link_largest_thread_and_latest_in_date_order():
    papers, threads = _agg(ROWS)
    beats = hn_svc.beats(papers["1706.03762"], threads)
    assert [b["k"] for b in beats] == ["First link", "Largest thread it came up in", "Latest link"]
    first, big, last = beats
    assert first["posted"] and first["points"] == 500 and first["when"] == "Jun 2017"
    assert big["title"] == "Ask HN: What papers changed how you work?" and big["points"] == 900
    assert last["title"] == "Transformers, explained" and not last["posted"] and last["points"] is None
    assert all(b["url"].startswith("https://news.ycombinator.com/item?id=") for b in beats)


def test_a_busiest_year_needs_enough_people():
    rows = [("1000.00001", f"9{i}", "comment", f"u{i}", ts_of(2020 if i < 6 else 2021, 1 + i % 12), "960")
            for i in range(8)]
    papers, threads = _agg(rows)
    busy = [b for b in hn_svc.beats(papers["1000.00001"], threads) if b["k"] == "Busiest year"]
    assert busy == [{"ts": busy[0]["ts"], "when": "2020", "k": "Busiest year", "people": 6, "year": 2020}]


# ── Quotes ───────────────────────────────────────────────────────────────────

def _cand(text, item="1", story="900", ts=ts_of(2020, 1)):
    return {"text": text, "item_id": item, "story_id": story, "created_at": ts}


def test_a_quote_is_cut_word_for_word_with_the_link_removed():
    q = hn_svc.pick_quote("1706.03762", "Attention Is All You Need", [_cand(GOOD, "101")])
    assert q["text"] == ("If you want to understand transformers, I highly recommend the original paper which "
                         "shows that attention alone beats recurrent models on translation.")
    assert q["url"] == "https://news.ycombinator.com/item?id=101" and q["when"] == "Jan 2020"
    for word in q["text"].rstrip(".").split():
        assert word in GOOD                                # nothing rewritten, only the link gone


@pytest.mark.parametrize("text", [
    BIBLIO, RUDE,
    "Attention Is All You Need https://arxiv.org/abs/1706.03762",           # the title and nothing else
    "> I highly recommend https://arxiv.org/abs/1706.03762 since it shows that attention wins.",  # quoting someone
    "Has anyone read https://arxiv.org/abs/1706.03762 and found that it holds up for long inputs?",  # a question
    "It shows that attention wins, see https://arxiv.org/abs/1706.03762 and also the follow up from",  # cut off
    "My 2c: the paper https://arxiv.org/abs/1706.03762 is about machine translation on WMT data sets mostly.",  # flat
    # the comment quoting the abstract, not speaking for itself
    'From the abstract https://arxiv.org/abs/1706.03762: "We propose a new simple network architecture, the '
    'Transformer, based solely on attention mechanisms, which the authors argue is a classic."',
    # a question, even a praising one
    "Isn't the classic here https://arxiv.org/abs/1706.03762 which shows that attention alone is enough? It is great.",
    # a question without its question mark (it sat after the link)
    "Wasn't there a classic paper https://arxiv.org/abs/1706.03762 which shows that attention alone is enough.",
    # a reading list: the links were the sentence
    "My favourite papers are https://arxiv.org/abs/1706.03762 and https://arxiv.org/abs/1810.04805 which show that "
    "pretraining works.",
    # the link was a word of the sentence
    "To see why, I highly recommend: https://arxiv.org/abs/1706.03762 and then the classic follow up on scaling laws.",
])
def test_weak_or_unsafe_passages_are_never_quoted(text):
    assert hn_svc.pick_quote("1706.03762", "Attention Is All You Need", [_cand(text)]) is None


def test_the_best_passage_wins_and_a_bracket_left_by_a_link_goes():
    weak = _cand("Fun fact, the classic paper (https://arxiv.org/abs/1706.03762) was rejected somewhere first.", "2")
    strong = _cand("This one (https://arxiv.org/abs/1706.03762 ) is a must-read: it shows that attention alone "
                   "is enough to beat recurrence.", "3")
    q = hn_svc.pick_quote("1706.03762", "Attention Is All You Need", [weak, strong])
    assert q["item_id"] == "3"
    assert q["text"] == "This one is a must-read: it shows that attention alone is enough to beat recurrence."


def test_a_long_passage_is_cut_at_a_sentence_end():
    text = ("I highly recommend https://arxiv.org/abs/1706.03762 because it shows that a model built only from "
            "attention can learn translation well. " + "It goes on to explain multi-head attention in detail " * 6 + ".")
    q = hn_svc.pick_quote("1706.03762", "Attention Is All You Need", [_cand(text)])
    assert q["text"].endswith("learn translation well.") and len(q["text"]) <= hn_svc.QUOTE_MAX


def test_brackets_left_open_or_closed_by_a_removed_link_are_dropped():
    assert hn_svc._clean("This paper (https://arxiv.org/abs/1 ) also touches on it.") == "This paper also touches on it."
    assert hn_svc._clean("See (https://x.org/a page 6) for examples.") == "See (page 6) for examples."
    assert hn_svc._clean("see (https://x.org/a and more") == "see and more"
    assert hn_svc._balanced("a) b [c] (d") == "a b [c] d"


async def test_quotes_come_from_the_comments_table_and_skip_reading_lists(world2):
    q = hn_svc._state["quotes"]
    assert set(q) == {"1706.03762"} and q["1706.03762"]["item_id"] == "101"
    assert q["1706.03762"]["thread"] == "Ask HN: What papers changed how you work?"
    # A comment that links many papers is a reading list, not a voice.
    cands = [{"text": GOOD, "item_id": "x", "story_id": "900", "created_at": NOW}]
    assert hn_svc.pick_quote("1706.03762", "Attention Is All You Need", cands)   # the rule itself accepts it ...
    write_comments_many = sqlite3.connect(world2)
    write_comments_many.execute("UPDATE hn_comments SET papers = 5 WHERE item_id = '101'")
    write_comments_many.execute("UPDATE hn_state SET value = 'later' WHERE key = 'loaded_at'")
    write_comments_many.commit()
    write_comments_many.close()
    await hn_svc.refresh()
    assert "1706.03762" not in hn_svc._state["quotes"]      # ... the load leaves reading lists out


# ── Views, this week and the chart ───────────────────────────────────────────

async def test_comebacks_and_just_joined_are_views_of_their_own(world2):
    cards = hn_svc.lasting_cards()
    assert [c["arxiv_id"] for c in hn_svc.browse(cards, "comebacks")] == ["1312.00001"]
    assert [c["arxiv_id"] for c in hn_svc.browse(cards, "joined")] == ["2607.00001"]
    by = {c["arxiv_id"]: c for c in cards}
    assert by["1312.00001"]["comeback_days"] >= 12 * 365 and not by["1312.00001"]["just_joined"]
    assert by["2607.00001"]["just_joined"] and by["2607.00001"]["comeback_days"] == 0
    assert by["1706.03762"]["joined_label"] == "Jan 2019" and not by["1706.03762"]["just_joined"]


async def test_this_week_puts_comebacks_first_and_leaves_out_passed_papers(world2):
    week = hn_svc.week_cards()
    assert [c["arxiv_id"] for c in week] == ["1312.00001", "2607.00001"]
    assert week[0]["week_thread"] == {"title": "Ask HN: what are you reading?", "points": 310, "submitted": False,
                                      "url": "https://news.ycombinator.com/item?id=951"}
    assert [c["arxiv_id"] for c in hn_svc.week_cards(exclude={"1312.00001"})] == ["2607.00001"]
    week[0]["query_id"] = "mine"
    assert "query_id" not in hn_svc.week_cards()[0]           # copies: per-request fields never reach the cache


async def test_the_chart_holds_every_listed_paper_stacked_by_first_link(world2):
    data = json.loads(hn_svc.strata_json())
    ids = [l[5] for l in data["lines"]]
    assert set(ids) == {c["arxiv_id"] for c in hn_svc.lasting_cards()} and "hep-th/9711200" not in ids
    assert [l[0] for l in data["lines"]] == sorted(l[0] for l in data["lines"])
    now = datetime.now(timezone.utc)
    assert data["months"] == hn_svc.month_index(now.year, now.month) + 1
    first = data["lines"][0]
    assert first[5] == "1312.00001" and first[3] == hn_svc._state["canon_rank"]["1312.00001"]
    assert first[2] == sorted(hn_svc._state["papers"]["1312.00001"]["voices"])


async def test_the_chart_endpoint_follows_the_switch(world2, monkeypatch):
    async with client() as c:
        on = await c.get("/trending/lasting/strata.json")
        monkeypatch.setenv("HN_LASTING", "off")
        off = await c.get("/trending/lasting/strata.json")
    assert on.status_code == 200 and on.headers["content-type"] == "application/json"
    assert "max-age" in on.headers["cache-control"] and len(on.json()["lines"]) == 3
    assert off.status_code == 404


# ── The page ─────────────────────────────────────────────────────────────────

async def test_the_page_shows_the_chart_this_week_quotes_and_the_story(world2, monkeypatch):
    record = AsyncMock()
    monkeypatch.setattr(db, "record_exposures", record)
    async with client() as c:
        html = (await c.get("/trending/lasting")).text
    assert 'class="hn-strata"' in html and 'data-src="/trending/lasting/strata.json"' in html
    assert "/static/lasting.js" in html
    week = html[html.index('id="hn-week-h"'):html.index('class="shelf hn-listing"')]
    assert "Back after 12 years" in week and "New to the list" in week
    assert "pol=hn_lasting_week_v1" in week and "sf=hn_lasting" in week
    assert "I highly recommend the original paper which shows that" in html
    assert 'href="https://news.ycombinator.com/item?id=101"' in html
    assert "<summary>The story · 3 threads</summary>" in html and "Largest thread it came up in" in html
    assert ">Comebacks <span class=\"n\">1</span>" in html and ">Just joined <span class=\"n\">1</span>" in html
    rows = record.await_args.args[0]
    lists = {e["policy_id"] for e in rows}
    assert lists == {"hn_lasting_list_v1", "hn_lasting_week_v1"}
    week_q = {e["query_id"] for e in rows if e["policy_id"] == "hn_lasting_week_v1"}
    list_q = {e["query_id"] for e in rows if e["policy_id"] == "hn_lasting_list_v1"}
    assert len(week_q) == 1 and len(list_q) == 1 and week_q != list_q
    assert [e["position"] for e in rows if e["policy_id"] == "hn_lasting_week_v1"] == [0, 1]


async def test_the_chart_and_this_week_stay_off_filtered_and_later_pages(world2):
    async with client() as c:
        physics = (await c.get("/trending/lasting?field=math")).text
        comebacks = (await c.get("/trending/lasting?view=comebacks")).text
        saved = (await c.get("/trending/lasting?view=saved")).text
        physics_only = (await c.get("/trending/lasting?field=physics")).text
    assert 'class="hn-strata"' not in physics and 'class="hn-strata"' not in comebacks
    assert "A Paper That Came Back" in physics and "Back on HN this week" in physics
    assert "back after 12" in comebacks and "pol=hn_lasting_comebacks_v1" in comebacks
    assert "Back on HN this week" not in saved
    assert "Back on HN this week" not in physics_only            # narrowed like the list: no physics paper this week


async def test_the_filters_are_three_menus_on_one_line_and_say_what_is_chosen(world2):
    async with client() as c:
        html = (await c.get("/trending/lasting?field=math&year=before-2015")).text
    assert html.count('<details class="hn-menu">') == 3
    assert '<span class="k">Field</span> Mathematics' in html and '<span class="k">Published</span> Before 2015' in html
    assert "A Paper That Came Back" in html and "Attention Is All You Need" not in html
    assert 'class="hn-clear" href="/trending/lasting"' in html


async def test_without_a_comments_table_the_page_has_no_quotes_but_works(world2):
    con = sqlite3.connect(world2)
    con.execute("DROP TABLE hn_comments")
    con.execute("UPDATE hn_state SET value = 'later' WHERE key = 'loaded_at'")
    con.commit()
    con.close()
    await hn_svc.refresh()
    async with client() as c:
        r = await c.get("/trending/lasting")
    assert r.status_code == 200 and "Attention Is All You Need" in r.text and 'class="hn-quote"' not in r.text
    assert hn_svc._state["quotes"] == {}


async def test_hn_lately_on_trending_keeps_its_plain_rows(world2, monkeypatch):
    monkeypatch.setenv("TRENDING_PAGE", "on")
    monkeypatch.setenv("HN_FRESH", "on")
    monkeypatch.setattr(ts, "_state", {**ts._state, "papers": {}, "vectors": {}, "code": {}})
    async with client() as c:
        html = (await c.get("/trending")).text
    section = html[html.index("On Hacker News lately"):]
    assert "in the last 30 days" in section and "hn-life-svg" not in section and "The story" not in section


def test_every_theme_block_defines_the_chart_colours():
    """The chart reads --hn-* from the stylesheet. A token missing from one of the three theme
    blocks (light, system dark, chosen dark) draws one theme's colours on the other's ground;
    this happened once while building the page (dark tokens landed twice in the media block)."""
    import re
    from pathlib import Path
    css = Path("app/static/styles.css").read_text()
    blocks = {"light": css[css.index(":root {"):css.index("@media (prefers-color-scheme: dark)")],
              "system dark": css[css.index("@media (prefers-color-scheme: dark)"):css.index(':root[data-theme="dark"] {')],
              "chosen dark": css[css.index(':root[data-theme="dark"] {'):]}
    blocks["chosen dark"] = blocks["chosen dark"][:blocks["chosen dark"].index("\n}\n")]
    for name, block in blocks.items():
        for token in ("--hn-line", "--hn-dot", "--hn-hot", "--hn-hot-bg"):
            assert len(re.findall(rf"{token}:", block)) == 1, (name, token)
    assert re.search(r"--hn-dot:\s*rgba\(127", blocks["chosen dark"])


async def test_a_failed_comments_read_costs_the_quotes_not_the_lists(world2, monkeypatch):
    """turso_svc.query raises httpx errors as well as RuntimeError; either must leave the lists loading."""
    import httpx
    real = hn_svc._paged

    async def flaky(sql):
        if "hn_comments" in sql:
            raise httpx.ConnectError("turso unreachable")
        return await real(sql)

    monkeypatch.setattr(hn_svc, "_paged", flaky)
    stats = await hn_svc.refresh(force=True)
    assert stats["error"] is None and stats["lasting"] == 3 and hn_svc._state["quotes"] == {}
    assert len(hn_svc.lasting_cards()) == 3 and json.loads(hn_svc.strata_json())["lines"]
