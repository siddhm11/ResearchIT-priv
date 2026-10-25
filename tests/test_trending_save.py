"""
Saving from a trending card (shelf, /trending, topic pages).

Cards matched by area, the week's top and every topic-page card have no
interest, so their cluster_id is None. hx-vals turned that into JSON null, and
htmx sends a null as the text "null"; the save handler's int("null") then
failed with a 500 and the Save button did nothing. Pass worked, because
app.js leaves nulls out of its form.

These tests post exactly what the browser posts.
"""
import html as htmlmod
import json
import re

import pytest

from app import db, user_state as us
from tests.test_trending_shelf import HF, READER, client, shelf_world  # noqa: F401  (fixture)


def htmx_form(vals: dict) -> dict:
    """What htmx sends for hx-vals: every value as a string, null as "null"."""
    return {k: ("null" if v is None else str(v)) for k, v in vals.items()}


def save_buttons(html: str) -> list[tuple[str, dict]]:
    out = []
    for m in re.finditer(r'hx-post="/api/papers/([^/]+)/save"[^>]*?hx-vals=\'([^\']*)\'', html, re.S):
        out.append((m.group(1), json.loads(htmlmod.unescape(m.group(2)))))
    return out


async def test_saving_from_an_area_matched_shelf_card_works(shelf_world):
    await db.save_user_clusters(READER, [])
    await db.save_onboarding_categories(READER, ["robotics"])
    from app import trending_shelf as ts
    for aid in HF:
        ts._state["papers"][aid]["categories"] = ["cs.RO"]
    async with client() as c:
        shelf = (await c.get("/api/trending-shelf")).text
        buttons = save_buttons(shelf)
        assert buttons, "the shelf has Save buttons"
        aid, vals = buttons[0]
        assert None not in vals.values(), "no null reaches the browser"
        r = await c.post(f"/api/papers/{aid}/save", data=htmx_form(vals))
    assert r.status_code == 200 and "Saved" in r.text
    us._cache.clear()
    state = await us.ensure_loaded(READER)
    assert aid in state.positive_list


async def test_saving_from_the_weeks_top_works(shelf_world):
    await db.save_user_clusters(READER, [])
    async with client() as c:
        shelf = (await c.get("/api/trending-shelf")).text
        assert "Trending this week" in shelf
        aid, vals = save_buttons(shelf)[0]
        r = await c.post(f"/api/papers/{aid}/save", data=htmx_form(vals))
    assert r.status_code == 200 and "Saved" in r.text


@pytest.mark.parametrize("cluster_id", ["null", "None", "undefined", "abc", ""])
async def test_a_bad_cluster_id_never_loses_the_save(shelf_world, cluster_id):
    """Pages already open in a browser still send "null"; the save must land."""
    async with client() as c:
        r = await c.post(f"/api/papers/{HF[0]}/save",
                         data={"source": "trending", "cluster_id": cluster_id, "position": "0"})
    assert r.status_code == 200 and "Saved" in r.text
    import aiosqlite
    async with aiosqlite.connect(db.DB_PATH) as conn:
        row = await (await conn.execute(
            "SELECT event_type, cluster_id FROM interactions WHERE user_id = ? AND paper_id = ?",
            (READER, HF[0]))).fetchone()
    assert row == ("save", None)


async def test_a_real_cluster_id_is_still_kept(shelf_world):
    async with client() as c:
        await c.post(f"/api/papers/{HF[1]}/save", data={"source": "trending", "cluster_id": "1"})
    import aiosqlite
    async with aiosqlite.connect(db.DB_PATH) as conn:
        row = await (await conn.execute(
            "SELECT cluster_id FROM interactions WHERE paper_id = ?", (HF[1],))).fetchone()
    assert row == (1,)
