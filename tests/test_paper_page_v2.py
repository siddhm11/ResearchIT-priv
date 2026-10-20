"""
The redesigned paper page (PAPER_PAGE_V2) and its code chip.

Metadata and the vector store are patched as in tests/test_paper_page.py.
Saved arXiv sources are written straight into the test database. Every test
that renders the page also proves it made no outside call: the shared HTTP
client is replaced by one that fails the test on any request.
"""
import asyncio
import json
from unittest.mock import patch

import aiosqlite
import httpx
import pytest
from fastapi.testclient import TestClient

from app import config, features, http_client, qdrant_svc, turso_svc
from app.main import app

PAPER = {
    "arxiv_id": "2609.31507",
    "title": "SatNav: A Scalable Benchmark for Long-Horizon UAV Vision-Language Navigation",
    "abstract": "Urban UAV navigation needs long-term memory. We introduce SatNav with $118$K episodes.",
    "authors": '["Jiajun Jiang", "Chunliang Hua", "Zichun Chen"]',
    "category": "cs.CV",
    "published": "2026-09-25",
    "year": 2026,
    "citation_count": 3,
}
RELATED = {"2609.27176": {"arxiv_id": "2609.27176", "title": "LeakScale", "abstract": "x",
                          "authors": '["Divyansh Singh"]', "year": 2026}}
FIG_SRC = "2609.31507v1/overview-final.png"


@pytest.fixture
def page(monkeypatch):
    """Patched metadata and neighbours, v2 on, and a client that refuses outside calls."""
    monkeypatch.setenv("PAPER_PAGE_V2", "on")
    calls = []

    def refuse(request):
        calls.append(str(request.url))
        raise AssertionError(f"outside call from the paper page: {request.url}")

    trap = httpx.AsyncClient(transport=httpx.MockTransport(refuse))
    monkeypatch.setattr(http_client, "get_client", lambda: trap)

    async def fake_meta(ids):
        out = {i: PAPER for i in ids if i == PAPER["arxiv_id"]}
        out.update({i: RELATED[i] for i in ids if i in RELATED})
        return out

    async def fake_vectors(ids):
        import numpy as np
        return {i: np.zeros(1024, dtype=np.float32) for i in ids}

    async def fake_search(**kwargs):
        return list(RELATED)

    with patch.object(turso_svc, "fetch_metadata_batch", side_effect=fake_meta), \
            patch.object(qdrant_svc, "get_paper_vectors", side_effect=fake_vectors), \
            patch.object(qdrant_svc, "search_by_vector", side_effect=fake_search):
        yield calls


def save_source(licence="CC-BY-4.0", figures=2, **extra):
    figs = [{"anchor": f"S1.F{i + 1}", "label": f"Figure {i + 1}", "caption": f"Caption {i + 1}.",
             "src": FIG_SRC.replace("overview", f"overview{i}"), "w": 476, "h": 235} for i in range(figures)]
    row = {"arxiv_id": PAPER["arxiv_id"], "status": "ok", "version": "v1", "licence": licence,
           "licence_text": "CC BY 4.0", "figures_json": json.dumps(figs), "n_figures": figures + 3,
           "parser_version": 1, **extra}

    async def go():
        async with aiosqlite.connect(config.DB_PATH) as conn:
            cols = ", ".join(row)
            await conn.execute(f"INSERT OR REPLACE INTO paper_sources ({cols}) VALUES ({', '.join('?' * len(row))})",
                               list(row.values()))
            await conn.commit()
    asyncio.run(go())


def get(path, cookies=None):
    with TestClient(app) as c:
        for k, v in (cookies or {}).items():
            c.cookies.set(k, v)
        return c.get(path)


def test_switch_off_serves_todays_page(page, monkeypatch):
    monkeypatch.setenv("PAPER_PAGE_V2", "off")
    html = get("/p/2609.31507").text
    assert 'class="paper-abstract"' in html and "data-sumbox" not in html


def test_switch_on_serves_the_new_page(page):
    html = get("/p/2609.31507").text
    assert "data-sumbox" in html and 'class="paper-abstract"' not in html
    assert 'class="page-wide"' in html
    assert "/static/paper.js?v=" in html and "katex.min.js?v=" in html


def test_preview_shows_only_to_the_owner_cookie(page, monkeypatch):
    monkeypatch.setenv("PAPER_PAGE_V2", "preview")
    monkeypatch.setattr(config, "PREVIEW_TOKEN", "s3cret")
    assert "data-sumbox" not in get("/p/2609.31507").text
    owner = {features.PREVIEW_COOKIE: features._cookie_value()}
    assert "data-sumbox" in get("/p/2609.31507", owner).text


def test_the_page_makes_no_outside_call(page, monkeypatch):
    monkeypatch.setenv("FIGURES", "on")
    save_source()
    assert get("/p/2609.31507").status_code == 200
    assert page == []


def test_without_a_saved_source_the_outline_lists_only_what_renders(page):
    html = get("/p/2609.31507").text
    assert 'href="#sec-sum"' in html and 'href="#sec-near"' in html
    assert 'href="#sec-fig"' not in html and 'id="sec-figs"' not in html
    assert "/fig/" not in html


def test_saved_figures_render_with_size_alt_text_and_credit(page, monkeypatch):
    monkeypatch.setenv("FIGURES", "on")
    save_source(figures=3)
    html = get("/p/2609.31507").text
    assert 'id="sec-fig"' in html and 'id="sec-figs"' in html
    assert 'width="476" height="235"' in html
    assert 'alt="Figure 1: Caption 1."' in html
    assert "Jiang et al. 2026, CC BY 4.0" in html
    assert "3 of 6" in html                         # 3 saved of 6 in the paper
    assert html.count('<img src="/fig/2609.31507/') == 4   # key figure + 3 in the grid


def test_figures_follow_the_licence_policy(page, monkeypatch):
    monkeypatch.setenv("FIGURES", "on")
    monkeypatch.setenv("FIGURE_LICENCES", "CC-BY-4.0")
    save_source(licence="ARXIV-NONEXCLUSIVE-1.0")
    assert "/fig/" not in get("/p/2609.31507").text


def test_figures_stay_hidden_with_their_switch_off(page):
    save_source()
    assert "/fig/" not in get("/p/2609.31507").text


def test_one_figure_shows_no_grid(page, monkeypatch):
    monkeypatch.setenv("FIGURES", "on")
    save_source(figures=1)
    html = get("/p/2609.31507").text
    assert 'id="sec-fig"' in html and 'id="sec-figs"' not in html


@pytest.mark.parametrize("field", ["title", "abstract", "authors"])
def test_paper_text_is_escaped(page, field):
    evil = '<script>alert(1)</script>"><img src=x onerror=alert(1)>'
    value = json.dumps([evil]) if field == "authors" else evil
    PAPER[field], old = value, PAPER[field]
    try:
        html = get("/p/2609.31507").text
    finally:
        PAPER[field] = old
    assert "<script>alert(1)</script>" not in html and "onerror=alert(1)>" not in html
    assert "&lt;script&gt;" in html


def test_captions_are_escaped(page, monkeypatch):
    monkeypatch.setenv("FIGURES", "on")
    figs = [{"anchor": "S1.F1", "label": "Figure 1", "caption": '<img src=x onerror=alert(1)>',
             "src": FIG_SRC, "w": 10, "h": 10}]

    async def go():
        async with aiosqlite.connect(config.DB_PATH) as conn:
            await conn.execute("INSERT INTO paper_sources (arxiv_id, status, version, licence, figures_json, "
                               "parser_version) VALUES (?, 'ok', 'v1', 'CC-BY-4.0', ?, 1)",
                               (PAPER["arxiv_id"], json.dumps(figs)))
            await conn.commit()
    asyncio.run(go())
    html = get("/p/2609.31507").text
    assert "<img src=x onerror" not in html


def test_the_back_link_appears_only_when_coming_from_the_feed(page):
    assert "← Your feed" not in get("/p/2609.31507").text
    assert "← Your feed" in get("/p/2609.31507?qid=q1&sf=recommendation&pos=0").text


def test_saves_from_the_new_page_carry_the_paper_page_surface(page):
    html = get("/p/2609.31507").text
    assert '&#34;source&#34;: &#34;paper_page&#34;' in html or '"source": "paper_page"' in html


def test_explain_v2_returns_only_the_summary_body(page, monkeypatch):
    from app import groq_svc

    async def fake(title, abstract):
        return "A plain summary with <b>tags</b>."

    monkeypatch.setattr(groq_svc, "explain_paper", fake)
    body = get("/api/papers/2609.31507/explain?v=2").text
    assert 'class="tx-body tex"' in body and "explain-label" not in body
    assert "&lt;b&gt;tags&lt;/b&gt;" in body
    classic = get("/api/papers/2609.31507/explain").text
    assert "In plain terms" in classic


# ── the code chip ────────────────────────────────────────────────────────────

def hf(monkeypatch, handler):
    calls = []

    def wrapped(request):
        calls.append(str(request.url))
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(wrapped))
    monkeypatch.setattr(http_client, "get_client", lambda: client)
    return calls


def test_the_code_chip_links_a_github_repo_and_remembers_it(page, monkeypatch):
    calls = hf(monkeypatch, lambda r: httpx.Response(200, json={"githubRepo": "https://github.com/Eku127/SatNav",
                                                                "githubStars": 15}))
    first = get("/api/papers/2609.31507/code").text
    second = get("/api/papers/2609.31507/code").text
    assert 'href="https://github.com/Eku127/SatNav"' in first and "★ 15" in first
    assert second == first
    assert calls == ["https://huggingface.co/api/papers/2609.31507"]


def test_a_repo_that_is_not_github_is_ignored(page, monkeypatch):
    hf(monkeypatch, lambda r: httpx.Response(200, json={"githubRepo": "https://evil.example/x/y"}))
    assert "code-chip" not in get("/api/papers/2609.31507/code").text


def test_no_hugging_face_page_is_remembered_for_a_week(page, monkeypatch):
    calls = hf(monkeypatch, lambda r: httpx.Response(404))
    assert get("/api/papers/2609.31507/code").text.strip() == ""
    get("/api/papers/2609.31507/code")
    assert len(calls) == 1


def test_a_failed_lookup_shows_nothing_and_is_not_cached(page, monkeypatch):
    def boom(request):
        raise httpx.ConnectTimeout("slow")

    calls = hf(monkeypatch, boom)
    assert get("/api/papers/2609.31507/code").text.strip() == ""
    get("/api/papers/2609.31507/code")
    assert len(calls) == 2


def test_an_unknown_paper_is_never_looked_up(page, monkeypatch):
    calls = hf(monkeypatch, lambda r: httpx.Response(200, json={"githubRepo": "https://github.com/a/b"}))
    assert get("/api/papers/2609.99999/code").text.strip() == ""
    assert calls == []


def test_the_code_chip_needs_the_new_page(page, monkeypatch):
    monkeypatch.setenv("PAPER_PAGE_V2", "off")
    calls = hf(monkeypatch, lambda r: httpx.Response(200, json={"githubRepo": "https://github.com/a/b"}))
    assert get("/api/papers/2609.31507/code").text.strip() == ""
    assert calls == []


def test_the_full_caption_is_used_when_this_container_has_the_full_text(page, monkeypatch):
    monkeypatch.setenv("FIGURES", "on")
    long_cut = "A" * 300
    figs = [{"anchor": "S1.F1", "label": "Figure 1", "caption": long_cut, "src": FIG_SRC, "w": 10, "h": 10}]

    async def go(with_fulltext):
        async with aiosqlite.connect(config.DB_PATH) as conn:
            await conn.execute("INSERT OR REPLACE INTO paper_sources (arxiv_id, status, version, licence, "
                               "figures_json, parser_version) VALUES (?, 'ok', 'v1', 'CC-BY-4.0', ?, 1)",
                               (PAPER["arxiv_id"], json.dumps(figs)))
            await conn.execute("DELETE FROM paper_fulltext")
            if with_fulltext:
                await conn.execute("INSERT INTO paper_fulltext (arxiv_id, figures_json) VALUES (?, ?)",
                                   (PAPER["arxiv_id"], json.dumps([{"anchor": "S1.F1", "caption": long_cut + " ends here."}])))
            await conn.commit()

    asyncio.run(go(False))
    assert long_cut + "…" in get("/p/2609.31507").text
    asyncio.run(go(True))
    assert long_cut + " ends here." in get("/p/2609.31507").text
