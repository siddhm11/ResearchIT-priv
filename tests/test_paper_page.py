"""
The paper page — the site's owned destination for a single paper.

Until this existed, no URL on the site was ABOUT a paper: every card sent its
title straight to arxiv.org. Three structural consequences, not cosmetic ones:
nothing was shareable (the Open Graph tags could only ever describe the feed),
there was nowhere to put comprehension features, and every card was an exit.
"""
import re
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import turso_svc, qdrant_svc
from app.main import app

PAPER = {
    "arxiv_id": "1706.03762",
    "title": "Attention Is All You Need",
    "abstract": "The dominant sequence transduction models are based on "
                "complex recurrent or convolutional neural networks. " * 2,
    "authors": '["A. Vaswani", "N. Shazeer"]',
    "category": "cs.CL",
    "published": "2017-06-12",
    "year": 2017,
    "citation_count": 100000,
}


def _client_with(paper=PAPER, related=None):
    async def fake_meta(ids):
        out = {}
        for i in ids:
            if paper and i == paper["arxiv_id"]:
                out[i] = paper
            elif related and i in related:
                out[i] = related[i]
        return out

    async def fake_search(**kwargs):
        return list(related or {})

    return (
        patch.object(turso_svc, "fetch_metadata_batch", side_effect=fake_meta),
        patch.object(qdrant_svc, "search_by_vector", side_effect=fake_search),
    )


def test_paper_page_renders_the_paper():
    meta, search = _client_with()
    with meta, search, patch.object(qdrant_svc, "get_paper_vectors", return_value={}):
        with TestClient(app) as c:
            r = c.get("/p/1706.03762")
    assert r.status_code == 200
    assert "Attention Is All You Need" in r.text
    assert "A. Vaswani" in r.text


def test_the_link_preview_names_the_paper_not_the_site():
    """The whole reason this page exists: a shared link that says something."""
    meta, search = _client_with()
    with meta, search, patch.object(qdrant_svc, "get_paper_vectors", return_value={}):
        with TestClient(app) as c:
            r = c.get("/p/1706.03762")

    title = re.search(r'<meta property="og:title" content="([^"]*)"', r.text)
    assert title and "Attention Is All You Need" in title.group(1)

    otype = re.search(r'<meta property="og:type" content="([^"]*)"', r.text)
    assert otype and otype.group(1) == "article"

    desc = re.search(r'<meta property="og:description" content="([^"]*)"', r.text)
    assert desc and "sequence transduction" in desc.group(1)


def test_link_preview_description_is_cut_on_a_word_boundary():
    from app.routers.paper import _summary_for_card
    long_paper = {**PAPER, "abstract": "word " * 200}
    out = _summary_for_card(long_paper)
    assert len(out) <= 205
    assert not out.rstrip("…").endswith("wor"), "cut mid-word"


def test_unknown_paper_returns_an_html_404():
    async def empty(ids):
        return {}
    with patch.object(turso_svc, "fetch_metadata_batch", side_effect=empty), \
         patch.object(turso_svc, "fetch_metadata", return_value=None):
        from app import arxiv_svc
        with patch.object(arxiv_svc, "fetch_metadata_batch", side_effect=empty):
            with TestClient(app) as c:
                r = c.get("/p/9999.99999")
    assert r.status_code == 404
    assert "text/html" in r.headers.get("content-type", "")


def test_related_papers_render_when_available():
    related = {
        "1810.04805": {**PAPER, "arxiv_id": "1810.04805", "title": "BERT"},
        "2005.14165": {**PAPER, "arxiv_id": "2005.14165", "title": "GPT-3"},
    }
    meta, search = _client_with(related=related)
    import numpy as np
    with meta, search, patch.object(
            qdrant_svc, "get_paper_vectors",
            return_value={"1706.03762": np.zeros(1024, dtype=np.float32)}):
        with TestClient(app) as c:
            r = c.get("/p/1706.03762")
    assert "What sits near this" in r.text
    assert "BERT" in r.text and "GPT-3" in r.text


def test_page_survives_a_dead_vector_store():
    """Related is best-effort; the page is worth serving without it."""
    async def boom(**kwargs):
        raise RuntimeError("qdrant down")
    meta, _ = _client_with()
    with meta, patch.object(qdrant_svc, "get_paper_vectors", side_effect=boom):
        with TestClient(app) as c:
            r = c.get("/p/1706.03762")
    assert r.status_code == 200
    assert "Attention Is All You Need" in r.text
    assert "What sits near this" not in r.text


def test_the_card_title_links_to_our_page():
    """Without this the destination is unreachable and the work is dead code."""
    import pathlib
    card = pathlib.Path("app/templates/partials/paper_card.html").read_text()
    assert 'class="card-title" href="/p/{{ paper.arxiv_id }}"' in card, (
        "card title still exits straight to arxiv.org")
    # …and the arXiv id in the foot still offers the source directly.
    assert 'class="card-id" href="https://arxiv.org/abs/' in card


def test_paper_page_carries_no_fabricated_ranking_instrumentation():
    """CLAUDE.md §3.11: a save from here has no query_id or propensity.

    Inventing one would corrupt exactly the IPS/SNIPS analysis those fields
    exist for, so the page must leave them empty rather than fill them in.
    """
    meta, search = _client_with()
    with meta, search, patch.object(qdrant_svc, "get_paper_vectors", return_value={}):
        with TestClient(app) as c:
            r = c.get("/p/1706.03762")
    vals = re.findall(r"hx-vals='([^']*)'", r.text)
    assert vals, "no action buttons rendered"
    for v in vals:
        assert '"query_id": ""' in v or '"query_id":""' in v or "query_id" not in v
