"""
Share cards (app/share_cards.py, app/routers/share.py): the 1200x630 picture a
link to a paper, topic, reading list, Lasting or Trending shows when pasted
into Slack, X, LinkedIn or WhatsApp. Built from data already held: no model
call, no arXiv fetch, no cookie; any failure answers with the site's picture.
"""
import io
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import share_cards

META = {"1706.03762": {"arxiv_id": "1706.03762", "title": "Attention Is All You Need",
                       "abstract": "We propose a new simple network architecture, the Transformer, based "
                                   "solely on attention mechanisms. More text follows here.",
                       "authors": '["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"]',
                       "category": "cs.CL", "published": "2017-06-12", "year": 2017, "citation_count": 194564}}


def _size(data: bytes) -> tuple[int, int]:
    return Image.open(io.BytesIO(data)).size


# ── drawing ─────────────────────────────────────────────────────────────────
def test_cards_are_1200_by_630_png():
    cards = [
        share_cards.paper_card(title="Attention Is All You Need", authors=["A", "B"], category="cs.CL",
                               year="2017", cited=10, arxiv_id="1706.03762", hook="A sentence."),
        share_cards.topic_card(name="Agent Harnesses", label="Rising topic", from_share=0.012,
                               to_share=0.029, rolling=[0.01, 0.02, 0.03], papers=["One", "Two"]),
        share_cards.list_card(title="Diffusion models", kicker="Reading list", blurb="Start here.",
                              items=["DDPM", "Score SDE"]),
    ]
    for data in cards:
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        assert _size(data) == (1200, 630)


def test_a_figure_goes_in_the_card():
    buf = io.BytesIO()
    Image.new("RGB", (800, 400), (200, 30, 30)).save(buf, "WEBP")
    with_fig = share_cards.paper_card(title="T", authors=["A"], figure=buf.getvalue())
    img = Image.open(io.BytesIO(with_fig)).convert("RGB")
    r, g, b = img.getpixel((930, 330))                      # the figure, in the right-hand column
    assert r > 190 and g < 40 and b < 40


def test_a_broken_figure_still_gives_a_card():
    data = share_cards.paper_card(title="T", authors=["A"], figure=b"not an image")
    assert _size(data) == (1200, 630)


def test_wrap_keeps_to_the_line_count_and_ends_in_an_ellipsis():
    long = "word " * 200
    lines = share_cards.wrap(long, 40, 500, 3)
    assert len(lines) == 3 and lines[-1].endswith("…")
    assert all(share_cards.text_w(line, 40) <= 500 for line in lines)
    assert share_cards.wrap("Short title", 40, 500, 3) == ["Short title"]


def test_clean_unwraps_latex_and_drops_what_no_font_draws():
    assert share_cards.clean(r"$\alpha$-Divergence and $O(n \log n)$") == "α-Divergence and O(n log n)"
    # No bundled font has accented Latin: the base letter, never an empty box.
    assert share_cards.clean("Jürgen Müller, Hervé") == "Jurgen Muller, Herve"
    assert share_cards.clean("emoji 🚀 gone") == "emoji gone"


def test_authors_line_shortens_long_lists():
    assert share_cards.authors_line(["A", "B"]) == "A, B"
    assert share_cards.authors_line(["A", "B", "C", "D", "E"]) == "A, B and 3 others"


def test_the_cache_keeps_the_most_recent():
    c = share_cards.Cache(size=2)
    c.put("a", b"1"); c.put("b", b"2"); c.get("a"); c.put("c", b"3")
    assert c.get("a") == b"1" and c.get("b") is None and c.get("c") == b"3"


# ── routes ──────────────────────────────────────────────────────────────────
@pytest.fixture
def client(tmp_path, monkeypatch):
    import asyncio

    import app.config as cfg
    import app.db as db_mod
    import app.user_state as us
    from app.main import app
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    us._cache.clear()
    share_cards.cache._d.clear()
    asyncio.run(db_mod.init_db())
    with TestClient(app) as c:
        yield c


@pytest.fixture
def meta(monkeypatch):
    import app.turso_svc as turso
    fetch = AsyncMock(side_effect=lambda ids: {i: dict(META[i]) for i in ids if i in META})
    monkeypatch.setattr(turso, "fetch_metadata_batch", fetch)
    return fetch


def test_paper_card_route(client, meta, monkeypatch):
    import app.arxiv_svc as arxiv
    import app.groq_svc as groq
    never = AsyncMock(side_effect=AssertionError("no outside call for a card"))
    monkeypatch.setattr(arxiv, "fetch_metadata_batch", never)
    monkeypatch.setattr(groq, "explain_paper", never, raising=False)
    r = client.get("/og/p/1706.03762.png", headers={"user-agent": "Slackbot-LinkExpanding 1.0"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert _size(r.content) == (1200, 630)
    assert "set-cookie" not in r.headers
    assert "max-age" in r.headers["cache-control"]


def test_unknown_paper_gets_the_site_picture(client, meta):
    r = client.get("/og/p/2101.99999.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    with open("app/static/og.png", "rb") as f:
        assert r.content == f.read()
    assert client.get("/og/p/not-an-id.png").content == r.content


def test_the_paper_page_names_its_card(client, meta, monkeypatch):
    import app.qdrant_svc as q
    monkeypatch.setattr(q, "search_by_vector", AsyncMock(return_value=[]), raising=False)
    html = client.get("/p/1706.03762", headers={"user-agent": "Twitterbot/1.0"}).text
    assert 'property="og:image" content="' in html
    assert "/og/p/1706.03762.png?v=" in html
    assert 'name="twitter:image" content="' in html and html.count("/og/p/1706.03762.png") >= 2


def test_list_and_unknown_topic_cards(client, meta):
    from app import collections_svc
    cols = collections_svc.load_all()
    if cols:
        slug = cols[0]["slug"]
        r = client.get(f"/og/list/{slug}.png")
        assert r.status_code == 200 and _size(r.content) == (1200, 630)
    assert client.get("/og/list/no-such-list.png").headers["content-type"] == "image/png"
    assert client.get("/og/topic/no-such-topic.png").headers["content-type"] == "image/png"


def test_lasting_and_trending_cards_fall_back_when_empty(client):
    for path in ("/og/lasting.png", "/og/trending.png"):
        r = client.get(path)
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
