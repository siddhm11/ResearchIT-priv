"""
Hacker News on ResearchIT (app/hn_svc.py, doc 06 2026-11-03): the counting
rule, what each surface shows, and the three switches.

What must hold: a month counts only when someone new links the paper, so one
account cannot make a paper lasting; a deleted account is never a person; the
lists leave out papers the corpus lacks but count them; the counts load from
the tables (here a local copy via HN_SQLITE) and a failed load keeps the
previous ones; the paper block, /trending/lasting and the section on /trending
render only when their switch is on, carry the §3.11 fields, and never read
anything on the request path.
"""
import sqlite3
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import httpx
import numpy as np
import pytest

from app import db, hn_svc, http_client, qdrant_svc, trending_shelf as ts, turso_svc, user_state as us
from app.main import app

READER = "hn-reader"
NOW = int(time.time())
DAY = 86400


def ts_of(y, m, d=15):
    return int(datetime(y, m, d, 12, tzinfo=timezone.utc).timestamp())


def unit(axis: int) -> np.ndarray:
    v = np.zeros(1024, dtype=np.float32)
    v[axis] = 1.0
    return v


# (arxiv_id, item_id, kind, author, created_at, story_id)
ROWS = [
    # Lasting: four people, each first in a different month; submitted once at 500 points.
    ("1706.03762", "100", "story", "ann", ts_of(2017, 6), "100"),
    ("1706.03762", "101", "comment", "bob", ts_of(2017, 9), "900"),
    ("1706.03762", "102", "comment", "cy", ts_of(2019, 1), "901"),
    ("1706.03762", "103", "comment", "di", ts_of(2019, 3), "901"),
    ("1706.03762", "104", "comment", "", ts_of(2019, 4), "901"),         # deleted account
    # One account every month for a year, two others in one month: Karpathy's
    # rule passes (3 people, 12 months), the new-voice rule does not (2 months).
    *[("2001.08361", f"2{m:02d}", "comment", "eve", ts_of(2021, m), "902") for m in range(1, 13)],
    ("2001.08361", "300", "comment", "fay", ts_of(2021, 12, 20), "902"),
    ("2001.08361", "301", "comment", "gus", ts_of(2021, 12, 21), "902"),
    # Fresh by people: two people in the last fortnight.
    ("2610.00001", "400", "comment", "hal", NOW - 10 * DAY, "903"),
    ("2610.00001", "401", "comment", "ivy", NOW - 2 * DAY, "903"),
    # Fresh by a submission that reached 50 points.
    ("2610.00002", "500", "story", "jo", NOW - 3 * DAY, "500"),
    # Once, years ago.
    ("1802.00001", "600", "comment", "kim", ts_of(2018, 2), "904"),
    # Lasting but not in the corpus.
    *[("hep-th/9711200", f"7{i}", "comment", f"p{i}", ts_of(2010 + i, 1), "905") for i in range(3)],
]
THREADS = {
    "100": ("Attention is all you need (arxiv.org)", 500, 210, ts_of(2017, 6)),
    "900": ("Ask HN: What papers changed how you work?", 900, 700, ts_of(2017, 9)),
    "901": ("Transformers, explained", 40, 12, ts_of(2019, 1)),
    "902": ("Scaling laws", 30, 20, ts_of(2021, 1)),
    "903": ("A thread", 12, 30, NOW - 10 * DAY),
    "500": ("The fresh paper", 50, 9, NOW - 3 * DAY),
    "904": ("Old thread", 5, 2, ts_of(2018, 2)),
    "905": ("Strings", 8, 3, ts_of(2010, 1)),
}
META = {a: {"arxiv_id": a, "title": t, "authors": '["A. Author", "B. Author"]', "year": y,
            "arxiv_categories": c, "citation_count": 9}
        for a, t, y, c in [("1706.03762", "Attention Is All You Need", 2017, "cs.CL cs.LG"),
                           ("2001.08361", "Scaling Laws for Neural Language Models", 2020, "cs.LG"),
                           ("2610.00001", "A fresh paper people discuss", 2026, "cs.CL"),
                           ("2610.00002", "A fresh paper someone posted", 2026, "quant-ph"),
                           ("1802.00001", "Mentioned once", 2018, "cs.CV"),
                           ("2005.00001", "Never on Hacker News", 2020, "cs.CV")]}
VECTORS = {"1706.03762": unit(1), "2610.00001": unit(2), "2610.00002": unit(3), "2001.08361": unit(4)}


def write_tables(path, rows=ROWS, loaded_at="2026-11-03T08:00:00+00:00"):
    con = sqlite3.connect(path)
    con.executescript("""
        DROP TABLE IF EXISTS hn_mentions; DROP TABLE IF EXISTS hn_threads; DROP TABLE IF EXISTS hn_state;
        CREATE TABLE hn_mentions (item_id TEXT NOT NULL, arxiv_id TEXT NOT NULL, kind TEXT NOT NULL,
          author TEXT NOT NULL, created_at INTEGER NOT NULL, story_id TEXT NOT NULL, checked_on TEXT NOT NULL,
          PRIMARY KEY (item_id, arxiv_id));
        CREATE TABLE hn_threads (story_id TEXT PRIMARY KEY, title TEXT NOT NULL, url TEXT NOT NULL,
          points INTEGER, comments INTEGER, created_at INTEGER, checked_on TEXT NOT NULL);
        CREATE TABLE hn_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);""")
    con.executemany("INSERT INTO hn_mentions VALUES (?, ?, ?, ?, ?, ?, 'd')",
                    [(i, a, k, au, t, s) for a, i, k, au, t, s in rows])
    con.executemany("INSERT INTO hn_threads VALUES (?, ?, '', ?, ?, ?, 'd')",
                    [(s, *v) for s, v in THREADS.items()])
    con.execute("INSERT INTO hn_state VALUES ('loaded_at', ?)", (loaded_at,))
    con.commit()
    con.close()


@pytest.fixture
def outside_calls(monkeypatch):
    """Every outbound request through the shared client, refused and recorded."""
    calls: list[str] = []

    def refuse(request):
        calls.append(str(request.url))
        return httpx.Response(503)

    trap = httpx.AsyncClient(transport=httpx.MockTransport(refuse))
    monkeypatch.setattr(http_client, "get_client", lambda: trap)
    return calls


@pytest.fixture
async def world(monkeypatch, tmp_path, outside_calls):
    """The tables in a local file, metadata and vectors patched, nothing remote."""
    path = tmp_path / "hn.sqlite"
    write_tables(path)
    monkeypatch.setenv("HN_SQLITE", str(path))
    monkeypatch.setattr(hn_svc, "_state", {**hn_svc._state, "papers": {}, "threads": {}, "canon": [],
                                           "canon_rank": {}, "fresh": [], "meta": {}, "vectors": {},
                                           "loaded_at": None, "rows": 0, "cards": []})
    monkeypatch.setattr(turso_svc, "fetch_metadata_batch",
                        AsyncMock(side_effect=lambda ids: {i: META[i] for i in ids if i in META}))
    monkeypatch.setattr(qdrant_svc, "get_paper_vectors",
                        AsyncMock(side_effect=lambda ids, remember=True: {i: VECTORS[i] for i in ids if i in VECTORS}))
    await db.init_db()
    us._cache.clear()
    await hn_svc.refresh()
    yield path
    us._cache.clear()


def client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url="http://test", cookies={"arxiv_user_id": READER})


# ── The counting rule (pure) ─────────────────────────────────────────────────

def _papers():
    threads = {s: {"title": v[0], "points": v[1], "comments": v[2], "created_at": v[3]} for s, v in THREADS.items()}
    return hn_svc.aggregate(list(ROWS), threads, NOW), threads


def test_a_month_counts_only_when_someone_new_links_the_paper():
    papers, _ = _papers()
    scaling = papers["2001.08361"]
    assert (scaling["people"], scaling["any_months"], scaling["months"]) == (3, 12, 2)
    assert not scaling["lasting"]
    attention = papers["1706.03762"]
    assert (attention["people"], attention["months"], attention["mentions"]) == (4, 4, 5)
    assert attention["lasting"]


def test_a_deleted_account_is_a_mention_but_never_a_person():
    papers, _ = _papers()
    assert papers["1706.03762"]["years"] == {2017: 2, 2019: 2}


def test_fresh_needs_two_people_or_an_upvoted_submission():
    papers, _ = _papers()
    assert hn_svc.is_fresh(papers["2610.00001"]) and hn_svc.is_fresh(papers["2610.00002"])
    assert not hn_svc.is_fresh(papers["1706.03762"])
    one_quiet = hn_svc.aggregate([("2610.00009", "1", "comment", "zed", NOW - DAY, "9")], {}, NOW)
    assert not hn_svc.is_fresh(one_quiet["2610.00009"])


def test_lasting_order_is_months_then_people():
    papers, _ = _papers()
    assert hn_svc.canon_order(papers) == ["1706.03762", "hep-th/9711200"]


def test_threads_put_noticed_submissions_first_and_quiet_ones_last():
    p = {"per_thread": {"1": 1, "2": 3, "3": 1, "4": 1}, "submitted": {"1", "4"}}
    threads = {"1": {"title": "Posted, 500 points", "points": 500, "created_at": ts_of(2020, 1)},
               "2": {"title": "Linked by three comments", "points": 40, "created_at": ts_of(2020, 1)},
               "3": {"title": "Linked once", "points": 4000, "created_at": ts_of(2020, 1)},
               "4": {"title": "Posted, nobody came", "points": 2, "created_at": ts_of(2020, 1)}}
    got = hn_svc.top_threads(p, threads, n=4)
    assert [t["story_id"] for t in got] == ["1", "2", "3", "4"]
    assert got[0]["url"] == "https://news.ycombinator.com/item?id=1" and got[0]["when"] == "Jan 2020"


def test_the_year_strip_fills_empty_years():
    papers, _ = _papers()
    strip = hn_svc.year_strip(papers["1706.03762"])
    assert [(y["year"], y["people"]) for y in strip] == [(2017, 2), (2018, 0), (2019, 2)]
    assert strip[0]["h"] == 100 and strip[1]["h"] == 0


def test_rank_can_take_people_as_its_popularity():
    near = {"arxiv_id": "a", "people": 1000, "upvotes": 0, "categories": []}
    far = {"arxiv_id": "b", "people": 1, "upvotes": 0, "categories": []}
    third = {"arxiv_id": "c", "people": 10, "upvotes": 0, "categories": []}
    interest = [{"cluster_idx": 0, "medoid": unit(1), "label": "x", "tone": 0}]
    vecs = {"a": unit(1) * .8 + unit(5) * .6, "b": unit(1) * .8 + unit(6) * .6, "c": unit(1) * .8 + unit(7) * .6}
    got = ts.rank([far, third, near], vecs, interest, 0.6, set(), votes_key="people")
    assert [p["arxiv_id"] for p in got] == ["a", "c", "b"]


# ── Loading ──────────────────────────────────────────────────────────────────

async def test_the_lists_count_but_leave_out_papers_the_corpus_lacks(world):
    s = hn_svc.stats()
    assert s["lasting"] == 1 and s["lasting_all"] == 2 and s["mentions"] == len(ROWS)
    assert hn_svc._state["canon"] == ["1706.03762"]
    assert hn_svc._state["fresh"] == ["2610.00001", "2610.00002"]
    assert set(hn_svc.vectors()) == {"1706.03762", "2610.00001", "2610.00002"}


async def test_an_unchanged_harvest_is_not_read_again(world):
    write_tables(world, rows=ROWS[:1])                       # same loaded_at
    await hn_svc.refresh()
    assert hn_svc.stats()["mentions"] == len(ROWS)
    write_tables(world, rows=ROWS[:1], loaded_at="2026-11-04T08:00:00+00:00")
    await hn_svc.refresh()
    assert hn_svc.stats()["mentions"] == 1


async def test_a_failed_load_keeps_the_counts(world, monkeypatch):
    monkeypatch.setenv("HN_SQLITE", str(world.parent / "missing.sqlite"))
    with pytest.raises(RuntimeError):
        await hn_svc.refresh(force=True)
    assert hn_svc.stats()["mentions"] == len(ROWS)


async def test_turso_reads_are_paged_by_rowid(monkeypatch):
    seen = []

    async def query(sql, args=(), timeout=30):
        seen.append((sql, args))
        after, limit = args
        rows = [{"rid": i} for i in range(after + 1, min(after + limit, 12) + 1)]
        return rows

    monkeypatch.delenv("HN_SQLITE", raising=False)
    monkeypatch.setattr(turso_svc, "query", query)
    monkeypatch.setattr(hn_svc, "PAGE_ROWS", 5)
    got = await hn_svc._paged("SELECT rowid AS rid FROM hn_mentions WHERE rowid > ? ORDER BY rowid LIMIT ?")
    assert [r["rid"] for r in got] == list(range(1, 13))
    assert [a for _s, a in seen] == [(0, 5), (5, 5), (10, 5)]


# ── The paper page ───────────────────────────────────────────────────────────

@pytest.fixture
def paper_page(monkeypatch):
    monkeypatch.setenv("PAPER_PAGE_V2", "on")

    async def meta(ids):
        return {i: {**META[i], "abstract": "An abstract.", "category": "cs.CL", "published": "2017-06-12"}
                for i in ids if i in META}
    monkeypatch.setattr(turso_svc, "fetch_metadata_batch", AsyncMock(side_effect=meta))
    monkeypatch.setattr(qdrant_svc, "search_by_vector", AsyncMock(return_value=[]))


async def test_the_paper_page_shows_where_hn_discussed_it(world, paper_page, monkeypatch, outside_calls):
    monkeypatch.setenv("HN_PAPER", "on")
    async with client() as c:
        html = (await c.get("/p/1706.03762")).text
    assert outside_calls == []
    assert 'id="sec-hn"' in html and 'href="#sec-hn"' in html
    assert "Linked by <b>4 people</b>" in html and "4 separate months" in html and "2017–2019" in html
    assert 'href="https://news.ycombinator.com/item?id=100"' in html and "the paper was the link" in html
    assert "Lasting · #1 of 1" in html
    assert 'aria-label="People who linked it each year: 2017 2, 2018 0, 2019 2"' in html
    assert "/trending/lasting" not in html                    # its switch is off


async def test_once_is_said_as_once(world, paper_page, monkeypatch):
    monkeypatch.setenv("HN_PAPER", "on")
    async with client() as c:
        html = (await c.get("/p/1802.00001")).text
    assert "Linked by one person, 2018." in html and "hn-strip" not in html and "Lasting ·" not in html


async def test_no_block_with_the_switch_off_or_without_mentions(world, paper_page, monkeypatch, outside_calls):
    async with client() as c:
        assert 'id="sec-hn"' not in (await c.get("/p/1706.03762")).text
        monkeypatch.setenv("HN_PAPER", "on")
        never = (await c.get("/p/2005.00001")).text
    assert "Never on Hacker News" in never and 'id="sec-hn"' not in never
    assert outside_calls == []


# ── /trending/lasting and the section on /trending ───────────────────────────

async def test_lasting_is_not_found_with_its_switch_off(world):
    async with client() as c:
        assert (await c.get("/trending/lasting")).status_code == 404


async def test_the_lasting_list_ranks_carries_the_fields_and_records_exposures(world, monkeypatch):
    monkeypatch.setenv("HN_LASTING", "on")
    record = AsyncMock()
    monkeypatch.setattr(db, "record_exposures", record)
    async with client() as c:
        r = await c.get("/trending/lasting")
    html = r.text
    assert r.status_code == 200 and "Attention Is All You Need" in html
    assert '<span class="hn-rank" title="Place on the lasting list">1</span>' in html
    assert "pol=hn_lasting_list_v1" in html and "src=hn_lasting" in html and "sf=hn_lasting" in html
    assert "One lasting paper isn&#39;t in ResearchIT" in html or "One lasting paper isn't in ResearchIT" in html
    assert "Closest to you" not in html                     # no interests, no areas
    rows = record.await_args.args[0]
    assert [(e["paper_id"], e["policy_id"], e["propensity"]) for e in rows] == [
        ("1706.03762", "hn_lasting_list_v1", 1.0)]


async def test_a_reader_with_interests_gets_the_closest_first(world, monkeypatch):
    monkeypatch.setenv("HN_LASTING", "on")
    interest = {"cluster_idx": 0, "medoid": unit(1), "label": "Transformers", "tone": 0}
    monkeypatch.setattr(ts, "reader_groups", AsyncMock(return_value=([interest], 0.6, [])))
    monkeypatch.setattr(ts, "MIN_SHOWN", 1)
    async with client() as c:
        you = (await c.get("/trending/lasting")).text
        everything = (await c.get("/trending/lasting?view=all")).text
    assert "Closest to your interests" in you and "pol=hn_lasting_for_you_v1" in you
    assert 'class="interest' not in you                     # one interest's name on every row says nothing
    assert ">Most lasting</h2>" in everything and "pol=hn_lasting_list_v1" in everything
    assert 'href="/trending/lasting"' in everything             # the way back to "Closest to you"


async def test_fields_and_years_filter_the_list_and_links_keep_them(world, monkeypatch):
    monkeypatch.setenv("HN_LASTING", "on")
    async with client() as c:
        ai = (await c.get("/trending/lasting?field=ai&year=2015-2019")).text
        physics = (await c.get("/trending/lasting?field=physics")).text
        bad = await c.get("/trending/lasting?field=nonsense&sort=nonsense&year=1066")
    assert "Attention Is All You Need" in ai and "1 paper" in ai
    assert 'href="/trending/lasting?sort=people&amp;year=2015-2019&amp;field=ai"' in ai
    assert "Attention Is All You Need" not in physics and "No lasting paper matches these filters." in physics
    assert bad.status_code == 200 and "Attention Is All You Need" in bad.text


async def test_each_view_says_what_it_counts(world, monkeypatch):
    monkeypatch.setenv("HN_LASTING", "on")
    async with client() as c:
        gems = (await c.get("/trending/lasting?view=gems")).text
        running = (await c.get("/trending/lasting?view=running")).text
        recent = (await c.get("/trending/lasting?view=recent")).text
        saved = (await c.get("/trending/lasting?view=saved")).text
    assert "cited 9 times" in gems and "pol=hn_lasting_gems_v1" in gems
    assert "linked over 1.8 years" in running
    assert "last linked 15 Apr 2019" in recent
    assert "None of the papers you saved is on the list" in saved


async def test_rows_carry_the_year_record_and_dated_threads(world, monkeypatch):
    monkeypatch.setenv("HN_LASTING", "on")
    async with client() as c:
        html = (await c.get("/trending/lasting")).text
    assert 'class="hn-spark"' in html and 'aria-label="People who linked it each year: 2017 2, 2019 2"' in html
    assert "<summary>3 threads</summary>" in html
    assert html.index("Attention is all you need (arxiv.org)") < html.index("Transformers, explained")


def _c(aid, field="ai", year=2017, people=5, threads=5, span=1.0, last=0, cites=10, rank=1):
    return {"arxiv_id": aid, "field": field, "year": year, "hn": {"people": people, "threads": threads},
            "span_years": span, "last": last, "citation_count": cites, "rank": rank}


def test_browse_views_sorts_and_filters():
    cards = [_c("a", rank=1, people=9, threads=1, span=2, last=5, cites=0),
             _c("b", field="physics", year=2010, rank=2, people=3, threads=9, span=9, last=9, cites=20),
             _c("c", year=2025, rank=3, people=5, threads=5, span=1, last=1, cites=10)]
    ids = lambda xs: [x["arxiv_id"] for x in xs]
    assert ids(hn_svc.browse(cards, this_year=2026)) == ["a", "b", "c"]
    assert ids(hn_svc.browse(cards, "running", this_year=2026)) == ["b", "a", "c"]
    assert ids(hn_svc.browse(cards, "recent", this_year=2026)) == ["b", "a", "c"]
    assert ids(hn_svc.browse(cards, "gems", this_year=2026)) == ["b"]        # a: no count; c: too new
    assert ids(hn_svc.browse(cards, "saved", saved=frozenset({"c"}), this_year=2026)) == ["c"]
    assert ids(hn_svc.browse(cards, sort="people", this_year=2026)) == ["a", "c", "b"]
    assert ids(hn_svc.browse(cards, sort="threads", this_year=2026)) == ["b", "c", "a"]
    assert ids(hn_svc.browse(cards, sort="newest", this_year=2026)) == ["c", "a", "b"]
    assert ids(hn_svc.browse(cards, field="physics", this_year=2026)) == ["b"]
    assert ids(hn_svc.browse(cards, year="2025", this_year=2026)) == ["c"]
    assert ids(hn_svc.browse(cards, year="2015-2019", this_year=2026)) == ["a"]
    assert ids(hn_svc.browse(cards, year="before-2015", this_year=2026)) == ["b"]


def test_fields_and_year_buckets_match_dairs_groups():
    assert [hn_svc.field_of([c]) for c in ("cs.LG", "cs.CR", "math.CO", "quant-ph", "q-bio.NC", "stat.ML", "stat.ME")] == \
        ["ai", "cs", "math", "physics", "bio", "ai", "bio"]
    assert hn_svc.field_of([], "hep-th/9711200") == "physics"
    assert [label for _k, label, *_ in hn_svc.year_buckets(2026)] == \
        ["2026", "2025", "2024", "2023", "2020–2022", "2015–2019", "Before 2015"]


async def test_trending_shows_hn_lately_only_with_its_switch(world, monkeypatch):
    monkeypatch.setenv("TRENDING_PAGE", "on")
    monkeypatch.setattr(ts, "_state", {**ts._state, "papers": {}, "vectors": {}, "code": {}})
    async with client() as c:
        off = (await c.get("/trending")).text
        monkeypatch.setenv("HN_FRESH", "on")
        on = (await c.get("/trending")).text
    assert "On Hacker News lately" not in off and "trend-views" not in off
    assert "On Hacker News lately" in on
    assert on.index("A fresh paper people discuss") < on.index("A fresh paper someone posted")
    assert "<b>2 people</b> in the last 30 days" in on and "pol=hn_fresh_page_v1" in on
    monkeypatch.setenv("HN_LASTING", "on")
    async with client() as c:
        both = (await c.get("/trending")).text
    assert 'class="trend-views"' in both and 'aria-current="page">This week' in both


def test_a_list_shows_the_year_a_paper_first_appeared():
    assert hn_svc.id_year("1706.03762") == 2017
    assert hn_svc.id_year("hep-th/9711200") == 1997
    assert hn_svc.id_year("cs/0101001") == 2001


def test_dated_threads_are_the_biggest_in_date_order_without_removed_ones():
    p = {"per_thread": {str(i): 1 for i in range(1, 13)}, "submitted": set()}
    threads = {str(i): {"title": f"T{i}", "points": i * 10, "created_at": ts_of(2010 + i, 1)} for i in range(1, 12)}
    threads["12"] = {"title": "[dead]", "points": 9999, "created_at": ts_of(2024, 1)}
    got = hn_svc.dated_threads(p, threads, n=3)
    assert [t["story_id"] for t in got] == ["9", "10", "11"]
    # The paper's own post comes before bigger threads that only mention it.
    p2 = {"per_thread": {"1": 1, "2": 1, "3": 1}, "submitted": {"1"}}
    t2 = {"1": {"title": "Posted", "points": 80, "created_at": ts_of(2020, 1)},
          "2": {"title": "Huge, one passing link", "points": 4000, "created_at": ts_of(2024, 1)},
          "3": {"title": "Bigger still", "points": 5000, "created_at": ts_of(2022, 1)}}
    assert [t["story_id"] for t in hn_svc.dated_threads(p2, t2, n=2)] == ["1", "3"]
    assert [t["story_id"] for t in hn_svc.top_threads(p, threads, n=1)] == ["11"]
