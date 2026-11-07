"""Titles named in a search overview link to their papers (2026-11-08).

The overview for "popular paper about bidirectional encoder" named "BERT:
Pre-training of Deep Bidirectional Transformers for Language Understanding"
as plain text. app/overview_links.py links the titles of the five papers the
model was shown, after escaping, so model text never becomes markup.
"""
import html
import re

import pytest

from app.overview_links import link_titles

BERT = {"arxiv_id": "1810.04805",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"}
MODERN = {"arxiv_id": "2412.13663",
          "title": "Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, "
                   "Memory Efficient, and Long Context Finetuning and Inference"}
SURVEY = {"arxiv_id": "2001.00001", "title": "Deep Learning: A Survey of Methods"}
OLD = {"arxiv_id": "hep-th/9901001", "title": "Large N Limit of Superconformal Field Theories"}
PAPERS = [BERT, MODERN, SURVEY, OLD]


def _escaped(text: str) -> str:
    """What generate_search_summary hands over: escaped, then **bold**."""
    return re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html.escape(text, quote=False))


def _links(out: str) -> list[tuple[str, str]]:
    return re.findall(r'<a class="overview-paper" href="([^"]+)">(.*?)</a>', out, re.S)


def test_a_full_title_links_to_its_paper():
    out = link_titles(_escaped("The seminal **BERT: Pre-training of Deep Bidirectional "
                               "Transformers for Language Understanding** paper."), PAPERS)
    assert _links(out) == [("/p/1810.04805", BERT["title"])]
    assert out.startswith("The seminal <strong><a ")


def test_case_and_line_breaks_do_not_matter():
    out = link_titles(_escaped("see large n limit of\nsuperconformal field theories."), PAPERS)
    assert _links(out) == [("/p/hep-th/9901001", "large n limit of\nsuperconformal field theories")]


def test_the_name_before_the_colon_links_and_only_its_first_mention():
    out = link_titles(_escaped("BERT reads both ways. BERT was later extended."), PAPERS)
    assert _links(out) == [("/p/1810.04805", "BERT")]
    assert out.endswith("BERT was later extended.")


def test_a_name_inside_a_longer_word_is_left_alone():
    assert _links(link_titles(_escaped("ModernBERT and BERT-style models"), PAPERS)) == []


def test_lowercase_or_generic_names_are_not_linked():
    out = link_titles(_escaped("Deep Learning keeps improving; bert is a word too."), PAPERS)
    assert _links(out) == []


def test_a_name_two_papers_share_links_neither():
    twin = {"arxiv_id": "1907.11692", "title": "BERT: A Robustly Optimized Pretraining Approach"}
    assert _links(link_titles(_escaped("BERT is popular."), [BERT, twin])) == []


def test_a_title_the_overview_does_not_name_is_not_linked():
    assert _links(link_titles(_escaped("Encoders read text in both directions."), PAPERS)) == []


def test_model_text_never_becomes_markup():
    evil = '<a href="https://evil.example">BERT</a> <script>alert(1)</script>'
    out = link_titles(_escaped(evil), PAPERS)
    assert "<script>" not in out and '<a href="https://evil' not in out
    assert _links(out) == [("/p/1810.04805", "BERT")]


def test_titles_with_markup_characters_still_match():
    odd = {"arxiv_id": "2101.00001", "title": "Q&A <Models>: Answering Questions at Scale"}
    out = link_titles(_escaped("We cite Q&A <Models>: Answering Questions at Scale here."), [odd])
    assert _links(out) == [("/p/2101.00001", "Q&amp;A &lt;Models&gt;: Answering Questions at Scale")]


def test_a_search_link_carries_its_provenance():
    out = link_titles(_escaped("BERT"), PAPERS, qid="q-1", positions={"1810.04805": 2})
    assert _links(out) == [("/p/1810.04805?qid=q-1&amp;sf=search_overview&amp;pos=2", "BERT")]


# ── The route ────────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    import asyncio

    from fastapi.testclient import TestClient

    import app.config as cfg
    import app.db as db_mod
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    from app.main import app
    asyncio.run(db_mod.init_db())
    with TestClient(app) as c:
        yield c


def _meta(monkeypatch):
    import app.turso_svc as turso

    async def fake_meta(ids):
        return {p["arxiv_id"]: {**p, "abstract": "An abstract."} for p in (BERT, MODERN)}
    monkeypatch.setattr(turso, "fetch_metadata_batch", fake_meta)


def test_the_overview_route_links_titles(client, monkeypatch):
    import app.groq_svc as groq
    _meta(monkeypatch)

    async def summary(query, papers):
        return _escaped("**BERT** set the pattern; ModernBERT extends it.")
    monkeypatch.setattr(groq, "generate_search_summary", summary)

    resp = client.get("/api/search/summary?q=bert&ids=2412.13663,1810.04805&qid=abc")
    assert resp.status_code == 200
    assert '<a class="overview-paper" href="/p/1810.04805?qid=abc&amp;sf=search_overview&amp;pos=1">BERT</a>' in resp.text


def test_a_cached_overview_gets_links_and_the_cache_keeps_none(client, monkeypatch):
    import asyncio

    import app.groq_svc as groq
    from app import analytics
    _meta(monkeypatch)
    ids = ["1810.04805", "2412.13663"]
    asyncio.run(analytics.save_overview("bert", ids, _escaped("BERT came first.")))

    async def never(query, papers):
        raise AssertionError("a cached overview must not call the model")
    monkeypatch.setattr(groq, "generate_search_summary", never)

    resp = client.get("/api/search/summary?q=bert&ids=1810.04805,2412.13663&qid=abc")
    assert 'href="/p/1810.04805?qid=abc' in resp.text
    assert asyncio.run(analytics.cached_overview("bert", ids)) == "BERT came first."
