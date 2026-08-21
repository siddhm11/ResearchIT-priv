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
    assert 'href="/p/{{ paper.arxiv_id }}' in card, (
        "card title still exits straight to arxiv.org")
    assert 'class="card-title"' in card
    assert 'card-title"\n     href="https://arxiv.org' not in card
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


# ── Click-through ────────────────────────────────────────────────────────────
#
# app/db.py declared `click` as an event_type since the schema was written and
# nothing ever wrote one. The system knew which papers were saved and which
# were dismissed, but not which were OPENED — the difference between "scrolled
# past" and "read", and the densest engagement signal a feed produces.

def _open(path):
    import numpy as np
    meta, search = _client_with()
    with meta, search, patch.object(
            qdrant_svc, "get_paper_vectors", return_value={}):
        with TestClient(app) as c:
            c.cookies.set("arxiv_user_id", "click-user")
            return c.get(path)


def _clicks(user="click-user"):
    import sqlite3
    from app import config
    conn = sqlite3.connect(config.DB_PATH)
    return conn.execute(
        "SELECT paper_id, query_id, position, propensity, policy_id "
        "FROM interactions WHERE user_id = ? AND event_type = 'click'",
        (user,)).fetchall()


def test_a_click_from_the_feed_is_logged_with_its_provenance():
    r = _open("/p/1706.03762?qid=q-abc&pos=3&src=cluster_1&prop=0.25&pol=v9.1")
    assert r.status_code == 200

    rows = _clicks()
    assert rows, "no click event written"
    paper_id, query_id, position, propensity, policy_id = rows[-1]
    assert paper_id == "1706.03762"
    assert query_id == "q-abc", "click cannot be tied back to its feed request"
    assert position == 3
    assert propensity == pytest.approx(0.25)
    assert policy_id == "v9.1"


def test_a_bare_visit_writes_no_click():
    """A shared link, a bookmark or a crawler has no propensity and no policy.

    Recording one as though it did would corrupt the §3.11 contract rather
    than honour it.
    """
    before = len(_clicks("bare-user"))
    import numpy as np
    meta, search = _client_with()
    with meta, search, patch.object(qdrant_svc, "get_paper_vectors", return_value={}):
        with TestClient(app) as c:
            c.cookies.set("arxiv_user_id", "bare-user")
            c.get("/p/1706.03762")
    assert len(_clicks("bare-user")) == before


def test_the_card_link_carries_the_feed_provenance():
    """Without the query string on the link, nothing above can happen."""
    import pathlib
    card = pathlib.Path("app/templates/partials/paper_card.html").read_text()
    for key in ("qid=", "pos=", "src=", "prop=", "pol="):
        assert key in card, f"card title link drops {key}"


def test_position_zero_is_recorded_not_discarded():
    """Rank 0 is the top of the feed and the slot CTR analysis cares about most.

    `position or None` recorded it as NULL — indistinguishable from "no
    position at all" — so the most important rank was missing from the data
    entirely. A -1 sentinel distinguishes the two.
    """
    _open("/p/1706.03762?qid=q-zero&pos=0&src=cluster_0&prop=1.0&pol=v9.1")
    rows = [r for r in _clicks() if r[1] == "q-zero"]
    assert rows, "no click logged for the top card"
    assert rows[-1][2] == 0, f"position 0 was stored as {rows[-1][2]!r}"


def test_a_genuinely_absent_position_is_still_null():
    from app.routers.events import _position, _NO_POSITION
    assert _position(_NO_POSITION) is None
    assert _position(None) is None
    assert _position(0) == 0
    assert _position(7) == 7
