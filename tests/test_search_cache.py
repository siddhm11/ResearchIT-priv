"""
Recent search results kept in memory (app/search_cache.py), so going back to a
search doesn't run it again (asked for 2026-10-21): no encode, no Groq rewrite,
no retrieval, no second search_log row, and the reader keeps the query_id they
were shown, so clicks stay tied to the search that showed them.
"""
import asyncio
import re
from unittest.mock import patch

import aiosqlite
import pytest
from fastapi.testclient import TestClient

from app import analytics, arxiv_svc, config, hybrid_search_svc, search_cache, turso_svc

IDS = ["2609.00001", "2609.00002", "2609.00003"]


# ── the cache itself ─────────────────────────────────────────────────────────

@pytest.fixture
def clock(monkeypatch):
    t = {"now": 1000.0}
    monkeypatch.setattr(search_cache, "_now", lambda: t["now"])
    return t


def papers(*ids):
    return [{"arxiv_id": i, "title": f"T {i}"} for i in ids]


def test_case_and_spacing_share_an_entry(clock):
    search_cache.put("Drone  Navigation ", papers("a"), {"retrieval_mode": "hybrid"})
    assert search_cache.get("drone navigation") is not None
    assert search_cache.key("  Drone\tNavigation ") == "drone navigation"


@pytest.mark.parametrize("mode,lives", [("hybrid", 600), ("keyword", 120), ("semantic", 120),
                                        ("arxiv_keyword", 120)])
def test_a_full_search_lives_ten_minutes_one_missing_an_arm_two(clock, mode, lives):
    search_cache.put("q", papers("a"), {"retrieval_mode": mode})
    clock["now"] += lives - 1
    assert search_cache.get("q") is not None
    clock["now"] += 2
    assert search_cache.get("q") is None


def test_nothing_is_kept_for_no_results_or_when_turned_off(clock, monkeypatch):
    assert search_cache.put("q", [], {"retrieval_mode": "hybrid"}) is None
    monkeypatch.setattr(config, "SEARCH_CACHE_TTL_S", 0)
    assert search_cache.put("q", papers("a"), {"retrieval_mode": "hybrid"}) is None
    assert search_cache.get("q") is None


def test_callers_get_copies(clock):
    search_cache.put("q", papers("a"), {"retrieval_mode": "hybrid"})
    got, meta = search_cache.get("q").copies()
    got[0]["saved"], meta["cached"] = True, True
    again, meta2 = search_cache.get("q").copies()
    assert "saved" not in again[0] and "cached" not in meta2


def test_the_oldest_entry_goes_first(clock, monkeypatch):
    monkeypatch.setattr(search_cache, "MAX_ENTRIES", 2)
    for q in ("one", "two"):
        search_cache.put(q, papers("a"), {"retrieval_mode": "hybrid"})
    search_cache.get("one")                                   # used, so "two" is now the oldest
    search_cache.put("three", papers("a"), {"retrieval_mode": "hybrid"})
    assert search_cache.get("two") is None and search_cache.get("one") and search_cache.get("three")


def test_a_repeat_is_the_same_reader_on_the_same_entry(clock):
    entry = search_cache.put("q", papers("a"), {"retrieval_mode": "hybrid"})
    search_cache.remember("alice", "Q", entry, "qid-1")
    assert search_cache.repeat_query_id("alice", "q", entry) == "qid-1"
    assert search_cache.repeat_query_id("bob", "q", entry) is None
    assert search_cache.repeat_query_id(None, "q", entry) is None
    newer = search_cache.put("q", papers("b"), {"retrieval_mode": "hybrid"})   # results changed
    assert search_cache.repeat_query_id("alice", "q", newer) is None


# ── the search route ─────────────────────────────────────────────────────────

@pytest.fixture
def site(monkeypatch):
    from app.main import app
    calls = {"search": 0, "meta": 0, "arxiv": 0}
    state = {"ids": IDS, "mode": "hybrid"}

    async def fake_search(q, limit=20, return_meta=False):
        calls["search"] += 1
        return list(state["ids"]), {"retrieval_mode": state["mode"], "rewritten_query": None, "total_time_ms": 900}

    async def meta(ids):
        calls["meta"] += 1
        return {i: {"arxiv_id": i, "title": f"Paper {i}", "abstract": "A.", "authors": "[]"} for i in ids}

    async def no_arxiv(q, *a, **k):
        calls["arxiv"] += 1
        return []

    monkeypatch.setattr(analytics, "_readers", {})
    with patch.object(hybrid_search_svc, "search", side_effect=fake_search), \
            patch.object(turso_svc, "fetch_metadata_batch", side_effect=meta), \
            patch.object(arxiv_svc, "search", side_effect=no_arxiv), TestClient(app) as c:
        yield c, calls, state


def get(client, q, reader, hx=False):
    client.cookies.clear()                    # the server sets the cookie too; keep only this reader's
    client.cookies.set(config.COOKIE_NAME, reader)
    return client.get("/search", params={"q": q}, headers={"HX-Request": "true"} if hx else {})


def qids(html):
    return set(re.findall(r"qid=([0-9a-f-]{36})", html))


def logged():
    async def go():
        async with aiosqlite.connect(config.DB_PATH) as conn:
            return await (await conn.execute("SELECT query, query_id FROM search_log ORDER BY id")).fetchall()
    return asyncio.run(go())


def searches_counted(reader):
    return sum(v.get("searches", 0) for (_day, u), v in analytics._readers.items() if u == reader)


def test_going_back_to_a_search_runs_nothing_again(site):
    client, calls, _ = site
    first = get(client, "drone navigation", "alice", hx=True)       # typed into the search box
    back = get(client, "drone navigation", "alice")                  # Back: the browser asks for the page
    assert calls == {"search": 1, "meta": 1, "arxiv": 0}
    assert "data-cached" not in first.text and "data-cached" in back.text and "<title>" in back.text
    assert len(qids(first.text)) == 1 and qids(back.text) == qids(first.text)   # same query_id
    assert len(logged()) == 1 and searches_counted("alice") == 1


def test_another_reader_shares_the_results_but_not_the_search(site):
    client, calls, _ = site
    a = get(client, "drone navigation", "alice")
    b = get(client, "Drone Navigation", "bob")
    assert calls["search"] == 1 and "data-cached" in b.text
    assert qids(a.text) != qids(b.text) and len(logged()) == 2
    assert searches_counted("alice") == 1 and searches_counted("bob") == 1


def test_one_reader_s_saves_never_show_on_another_reader_s_results(site):
    client, _, _ = site
    async def save():
        async with aiosqlite.connect(config.DB_PATH) as conn:
            await conn.execute("INSERT INTO interactions (user_id, paper_id, event_type, source) "
                               "VALUES ('alice', ?, 'save', 'search')", (IDS[0],))
            await conn.commit()
    asyncio.run(save())
    alice = get(client, "drone navigation", "alice")
    bob = get(client, "drone navigation", "bob")
    assert "btn-saved" in alice.text and "btn-saved" not in bob.text


def test_after_ten_minutes_the_search_runs_again_and_counts_again(site, clock):
    client, calls, _ = site
    first = get(client, "drone navigation", "alice")
    clock["now"] += 601
    again = get(client, "drone navigation", "alice")
    assert calls["search"] == 2 and "data-cached" not in again.text
    assert qids(again.text) != qids(first.text) and len(logged()) == 2


def test_a_search_missing_an_arm_is_run_again_after_two_minutes(site, clock):
    client, calls, state = site
    state["mode"] = "keyword"
    get(client, "drone navigation", "alice")
    clock["now"] += 60
    get(client, "drone navigation", "alice")
    assert calls["search"] == 1
    clock["now"] += 61
    get(client, "drone navigation", "alice")
    assert calls["search"] == 2


def test_a_search_with_no_results_is_not_kept(site):
    client, calls, state = site
    state["ids"] = []
    get(client, "nothing here", "alice")
    get(client, "nothing here", "alice")
    assert calls["search"] == 2 and calls["arxiv"] == 2


def test_the_cache_can_be_turned_off(site, monkeypatch):
    client, calls, _ = site
    monkeypatch.setattr(config, "SEARCH_CACHE_TTL_S", 0)
    get(client, "drone navigation", "alice")
    get(client, "drone navigation", "alice")
    assert calls["search"] == 2 and len(logged()) == 2
