"""scripts/hn_harvest.py: which papers an HN item links, fetching past
Algolia's 1,000-hit ceiling, and what the daily job stores."""
import importlib.util
import re
import sqlite3
import sys
from pathlib import Path

import httpx
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.append(str(_SCRIPTS))
_spec = importlib.util.spec_from_file_location("hn_harvest", _SCRIPTS / "hn_harvest.py")
hn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hn)

YEAR = 2026


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(hn.time, "sleep", lambda s: None)


# ── Which papers an item links ───────────────────────────────────────────────

def test_ids_from_links_of_every_kind_without_versions():
    text = ("Attention: https://arxiv.org/abs/1706.03762v5, the PDF arxiv.org/pdf/2401.00001.pdf, "
            "https://www.alphaxiv.org/abs/2502.12345 and https://huggingface.co/papers/2510.01234 "
            "and ar5iv.labs.arxiv.org/html/1512.03385")
    assert hn.arxiv_ids(text, YEAR) == ["1706.03762", "2401.00001", "2502.12345", "2510.01234", "1512.03385"]


def test_ids_from_escaped_html_and_citations():
    html_text = ('<a href="https:&#x2F;&#x2F;arxiv.org&#x2F;abs&#x2F;2310.06825" rel="nofollow">'
                 'https:&#x2F;&#x2F;arxiv.org&#x2F;abs&#x2F;2310.06825</a> and arXiv:2005.14165')
    assert hn.arxiv_ids(html_text, YEAR) == ["2310.06825", "2005.14165"]


def test_old_style_ids_take_the_corpus_form():
    text = "arxiv.org/abs/hep-th/9711200 and https://arxiv.org/abs/math.GT/0309136 and arxiv.org/pdf/cs/0101001"
    assert hn.arxiv_ids(text, YEAR) == ["hep-th/9711200", "math/0309136", "cs/0101001"]


def test_impossible_ids_and_bare_numbers_are_ignored():
    assert hn.arxiv_ids("arxiv.org/abs/2413.00001", YEAR) == []        # month 13
    assert hn.arxiv_ids("arxiv.org/abs/2701.00001", YEAR) == []        # next year
    assert hn.arxiv_ids("arxiv.org/abs/1501.0001", YEAR) == []         # 2015 ids have five digits
    assert hn.arxiv_ids("arxiv.org/abs/0612.0001", YEAR) == []         # before April 2007's scheme
    assert hn.arxiv_ids("the number 1706.03762 alone", YEAR) == []
    assert hn.arxiv_ids("arxiv.org/abs/1706.03762 arxiv.org/pdf/1706.03762v2", YEAR) == ["1706.03762"]
    assert hn.arxiv_ids(None, YEAR) == []


def test_plain_keeps_paragraphs_and_decodes_entities():
    assert hn.plain("First &amp; foremost<p>See <a href=\"x\">this</a> &#x2F; <i>that</i>") == \
        "First & foremost\n\nSee this / that"


def _story(item="100", url="https://arxiv.org/abs/2401.00001", ts=1_790_000_000, points=120):
    return {"objectID": item, "author": "alice", "created_at_i": ts, "title": "A  paper", "url": url,
            "points": points, "num_comments": 40}


def _comment(item="200", text="see arxiv.org/abs/2401.00001 and arxiv.org/abs/1706.03762", story=100,
             ts=1_790_000_100, author="bob"):
    return {"objectID": item, "author": author, "created_at_i": ts, "story_id": story,
            "comment_text": text, "story_title": "Some thread", "story_url": "https://example.com"}


def test_a_story_is_a_submission_with_its_thread():
    r = hn.rows_from_hit("story", _story(), "2026-11-03", YEAR)
    assert r["mentions"] == [("100", "2401.00001", "story", "alice", 1_790_000_000, "100", "2026-11-03")]
    assert r["thread"] == ("100", "A paper", "https://arxiv.org/abs/2401.00001", 120, 40, 1_790_000_000,
                           "2026-11-03")
    assert r["comment"] is None
    assert hn.rows_from_hit("story", _story(url="https://example.com/arxiv-is-great"), "d", YEAR)["mentions"] == []


def test_a_comment_names_every_paper_and_its_thread():
    r = hn.rows_from_hit("comment", _comment(), "2026-11-03", YEAR)
    assert [m[1] for m in r["mentions"]] == ["2401.00001", "1706.03762"]
    assert all(m[2] == "comment" and m[5] == "100" for m in r["mentions"])
    assert r["comment"][0] == "200" and r["comment"][3] == 2
    assert r["fallback"] == ("100", "Some thread", "https://example.com", None, None, None, "2026-11-03")
    assert hn.rows_from_hit("comment", _comment(text="I love arxiv"), "d", YEAR)["comment"] is None
    assert hn.rows_from_hit("comment", {**_comment(), "story_id": None}, "d", YEAR)["mentions"] == []


# ── Fetching ─────────────────────────────────────────────────────────────────

def _index_handler(items, calls):
    """A fake search_by_date over `items` (dicts with created_at_i), honouring
    the numeric filter and the 1,000-hit ceiling, like Algolia."""
    def handler(request):
        calls.append(dict(request.url.params))
        lo, hi = map(int, re.findall(r"created_at_i[<>]=?(\d+)", request.url.params["numericFilters"]))
        hits = [h for h in items if lo <= h["created_at_i"] < hi]
        per = int(request.url.params["hitsPerPage"])
        return httpx.Response(200, json={"nbHits": len(hits), "hits": hits[:min(per, 1000)]})
    return handler


def test_a_window_past_the_ceiling_is_split_until_every_hit_is_fetched():
    items = [{"objectID": str(i), "created_at_i": 1_000_000 + i * 300} for i in range(2500)]
    calls: list = []
    client = httpx.Client(transport=httpx.MockTransport(_index_handler(items, calls)))
    out: list = []
    stats = {"requests": 0, "lost": 0}
    hn.fetch_window(client, {"query": "arxiv"}, 1_000_000, 1_000_000 + 2500 * 300, out, stats)
    assert sorted(int(h["objectID"]) for h in out) == list(range(2500))
    assert stats["lost"] == 0 and stats["requests"] == len(calls)


def test_long_windows_are_counted_before_being_downloaded():
    items = [{"objectID": "1", "created_at_i": 5_000_000}]
    calls: list = []
    client = httpx.Client(transport=httpx.MockTransport(_index_handler(items, calls)))
    out: list = []
    hn.fetch_window(client, {"query": "arxiv"}, 0, 100 * 86400, out, {"requests": 0, "lost": 0})
    assert [c["hitsPerPage"] for c in calls] == ["0", "1000"]
    calls.clear()
    hn.fetch_window(client, {"query": "arxiv"}, 60 * 86400, 100 * 86400, [], {"requests": 0, "lost": 0})
    assert [c["hitsPerPage"] for c in calls] == ["0"]                 # empty: nothing downloaded


def test_year_ranges_cut_at_new_year():
    lo = 1_703_980_800          # 2023-12-31
    hi = 1_735_776_000          # 2025-01-02
    got = hn._year_ranges(lo, hi)
    assert got[0][0] == lo and got[-1][1] == hi and len(got) == 3
    assert all(a < b for a, b in got) and all(got[i][1] == got[i + 1][0] for i in range(len(got) - 1))


def test_thread_lookup_asks_by_id_and_leaves_out_unknown_ids():
    seen = []

    def handler(request):
        tags = request.url.params["tags"]
        seen.append(tags)
        ids = re.findall(r"story_(\d+)", tags)
        hits = [{"objectID": i, "title": f"T{i}", "url": "", "points": 7, "num_comments": 3,
                 "created_at_i": 1_700_000_000} for i in ids if i != "999"]
        return httpx.Response(200, json={"hits": hits})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    got = hn.lookup_threads(client, ["1", "2", "999"], "2026-11-03")
    assert set(got) == {"1", "2"} and got["1"][3] == 7
    assert seen == ["story,(story_1,story_2,story_999)"]


# ── Storing ──────────────────────────────────────────────────────────────────

def _run(conn, stmt):
    def val(a):
        if a["type"] == "null":
            return None
        return int(a["value"]) if a["type"] == "integer" else a["value"]
    conn.execute(stmt["sql"], [val(a) for a in stmt.get("args", [])])


class FakeStores:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")

    def t(self, sql, args=()):
        return [list(r) for r in self.conn.execute(sql, args).fetchall()]

    def t_many(self, stmts):
        for s in stmts:
            _run(self.conn, s)


def test_a_title_only_row_never_wipes_a_looked_up_thread():
    st = FakeStores()
    for sql in hn.DDL:
        st.t(sql)
    looked = ("100", "A paper", "https://arxiv.org/abs/2401.00001", 120, 40, 1_790_000_000, "2026-11-02")
    for tx in hn.statements([], [], [looked]):
        st.t_many(tx)
    fallback = ("100", "", "", None, None, None, "2026-11-03")
    for tx in hn.statements([], [], [fallback]):
        st.t_many(tx)
    assert st.t("SELECT title, url, points, comments, created_at, checked_on FROM hn_threads") == [
        ["A paper", "https://arxiv.org/abs/2401.00001", 120, 40, 1_790_000_000, "2026-11-03"]]


def test_writes_are_batched():
    mentions = [(str(i), "2401.00001", "comment", "a", 1, "9", "d") for i in range(hn.WRITE_BATCH + 1)]
    txs = hn.statements(mentions, [], [])
    assert len(txs) == 2 and txs[0][0]["sql"].count("(?, ?, ?, ?, ?, ?, ?)") == hn.WRITE_BATCH


def _api(items_by_kind, fail=False):
    """search_by_date answering per kind (comment or story), plus the thread lookup."""
    def handler(request):
        if fail:
            return httpx.Response(503)
        p = request.url.params
        if request.url.path.endswith("/search"):
            ids = re.findall(r"story_(\d+)", p["tags"])
            return httpx.Response(200, json={"hits": [
                {"objectID": i, "title": "Thread " + i, "url": "", "points": 55, "num_comments": 9,
                 "created_at_i": 1_790_000_000} for i in ids]})
        lo, hi = map(int, re.findall(r"created_at_i[<>]=?(\d+)", p["numericFilters"]))
        kind = "story" if p["tags"] == "story" else "comment"
        mine = items_by_kind.get((kind, p["query"]), [])
        hits = [h for h in mine if lo <= h["created_at_i"] < hi]
        return httpx.Response(200, json={"nbHits": len(hits), "hits": hits[:int(p["hitsPerPage"])]})
    return handler


def test_harvest_stores_mentions_threads_and_the_watermark():
    st = FakeStores()
    now = 1_790_100_000
    items = {("story", "arxiv"): [_story(ts=now - 5000)],
             ("comment", "arxiv"): [_comment(ts=now - 4000), _comment(item="201", story=300, ts=now - 3000,
                                                                       author="carol")]}
    client = httpx.Client(transport=httpx.MockTransport(_api(items)))
    got = hn.harvest(st, now, client=client)
    assert "error" not in got
    assert got["mentions"] == 5 and got["papers"] == 2 and got["comments"] == 2 and got["threads"] == 2
    assert st.t("SELECT value FROM hn_state WHERE key = 'fetched_to'") == [[str(now)]]
    # The story came with its points; thread 300 was looked up.
    assert st.t("SELECT story_id, points FROM hn_threads ORDER BY story_id") == [["100", 120], ["300", 55]]
    assert st.t("SELECT COUNT(DISTINCT author) FROM hn_mentions WHERE arxiv_id = '2401.00001'") == [[3]]

    # The next run starts OVERLAP_S before the watermark and adds nothing twice.
    later = now + 86400
    client = httpx.Client(transport=httpx.MockTransport(_api(items)))
    again = hn.harvest(st, later, client=client)
    assert again["from"].startswith("2026") and again["mentions"] == 5
    assert st.t("SELECT COUNT(*) FROM hn_mentions") == [[5]]


def test_a_failed_harvest_reports_and_keeps_the_watermark():
    st = FakeStores()
    for sql in hn.DDL:
        st.t(sql)
    st.t("INSERT INTO hn_state (key, value) VALUES ('fetched_to', '1790000000')")
    client = httpx.Client(transport=httpx.MockTransport(_api({}, fail=True)))
    got = hn.harvest(st, 1_790_100_000, client=client)
    assert got["error"].startswith("HTTPStatusError")
    assert st.t("SELECT value FROM hn_state WHERE key = 'fetched_to'") == [["1790000000"]]
