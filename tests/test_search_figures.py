"""
Search thumbnails (switch SEARCH_FIGURES): a small figure beside a search
result whose paper already has one saved.

What matters: without a figure the classic card is byte-for-byte unchanged
(it is used on every page), the thumbnail carries the same §3.11 link as the
title, and nothing changes with the switch off.
"""
import json
import re
from unittest.mock import patch

import aiosqlite
from fastapi.testclient import TestClient

from app import config, hybrid_search_svc, turso_svc
from app.templates_env import templates

PAPER = {
    "arxiv_id": "2609.31507", "title": "SatNav", "abstract": "Urban UAV navigation needs memory. We build SatNav.",
    "authors": '["Jiajun Jiang", "Chunliang Hua"]', "year": 2026, "category": "cs.CV",
    "query_id": "q-1", "position": 0, "propensity": 1.0, "policy_id": "search_v2_independent_lexical",
}
THUMB = {"url": "/fig/2609.31507/0-69f7df71.webp", "w": 476, "h": 235, "label": "Figure 1",
         "caption": "Overview.", "credit": "Jiang et al. 2026, CC BY 4.0"}


def render(paper):
    return templates.env.get_template("partials/paper_card.html").render(paper=paper, source="search")


def test_without_a_figure_the_card_is_unchanged():
    assert render(PAPER) == render({**PAPER, "card_figure": None})
    assert "card-split" not in render(PAPER) and "card-thumb" not in render(PAPER)


def test_the_thumbnail_links_like_the_title_and_stays_out_of_the_way():
    html = render({**PAPER, "card_figure": THUMB})
    title = re.search(r'class="card-title"\s+href="([^"]+)"', html).group(1)
    thumb = re.search(r'<a class="card-thumb" href="([^"]+)"', html).group(1)
    assert thumb == title + "#sec-fig" and "qid=q-1" in title and "sf=search" in title
    tag = re.search(r'<a class="card-thumb"[^>]*>', html, re.S).group(0)
    assert 'tabindex="-1"' in tag and 'aria-hidden="true"' in tag and "CC BY 4.0" in tag
    assert f'src="{THUMB["url"]}"' in html and 'alt=""' in html


async def save_source():
    figs = [{"anchor": "S1.F1", "label": "Figure 1", "caption": "Overview.",
             "src": "2609.31507v1/overview-final.png", "w": 476, "h": 235}]
    async with aiosqlite.connect(config.DB_PATH) as conn:
        await conn.execute(
            "INSERT OR REPLACE INTO paper_sources (arxiv_id, status, version, licence, figures_json, "
            "parser_version) VALUES ('2609.31507', 'ok', 'v1', 'CC-BY-4.0', ?, 2)", (json.dumps(figs),))
        await conn.commit()


def search_html(monkeypatch, mode):
    from app.main import app
    import asyncio
    monkeypatch.setenv("SEARCH_FIGURES", mode)
    monkeypatch.setenv("FIGURES", "on")
    asyncio.run(save_source())

    async def fake_search(q, limit=20, return_meta=False):
        return ["2609.31507", "2609.00002"], {}

    async def meta(ids):
        return {i: {**PAPER, "arxiv_id": i, "title": f"Paper {i}"} for i in ids}

    with patch.object(hybrid_search_svc, "search", side_effect=fake_search), \
            patch.object(turso_svc, "fetch_metadata_batch", side_effect=meta), TestClient(app) as c:
        return c.get("/search?q=drones", headers={"HX-Request": "true"}).text


def test_search_shows_a_thumbnail_only_with_the_switch_on(monkeypatch):
    on = search_html(monkeypatch, "on")
    assert on.count('class="card-thumb"') == 1                     # only the paper with a saved figure
    assert "/fig/2609.31507/0-" in on
    assert [m for m in re.findall(r'data-arxiv-id="([^"]+)"', on)] == ["2609.31507", "2609.00002"]
    off = search_html(monkeypatch, "off")
    assert "card-thumb" not in off and "/fig/" not in off
