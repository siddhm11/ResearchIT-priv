"""
The area pickers (onboarding and /interests) show a drawing for each area.

An area's name alone often means nothing to someone outside the field, so each
card shows the field doing its typical thing. These tests keep every area in
CATEGORY_GROUPS drawn, and keep the drawings decorative and themeable.
"""
import re

import pytest
from fastapi.testclient import TestClient

from app.config import CATEGORY_GROUPS
from app.templates_env import templates


def _area_art():
    return templates.env.get_template("partials/area_art.html").module.area_art


def test_every_area_has_a_drawing_of_its_own():
    """A new area added without a drawing falls back to a blank paper."""
    area_art = _area_art()
    fallback = str(area_art("no-such-area"))
    drawings = {key: str(area_art(key)) for key in CATEGORY_GROUPS}
    assert [k for k, d in drawings.items() if d == fallback] == []
    assert len(set(drawings.values())) == len(drawings)


def test_drawings_are_decorative_and_take_the_page_colours():
    """Hidden from screen readers (the name and description carry the meaning),
    and coloured only through classes, so both themes apply."""
    for key in CATEGORY_GROUPS:
        svg = str(_area_art()(key))
        assert svg.count("<svg") == 1 and 'aria-hidden="true"' in svg
        assert not re.search(r'(fill|stroke)="#', svg), key
        # A redrawn stroke needs pathLength, or the dash pattern breaks it up.
        for el in re.findall(r"<[^>]*\bdraw\b[^>]*>", svg):
            assert 'pathLength="1"' in el and el.startswith("<path"), (key, el)


@pytest.fixture
def client(tmp_path, monkeypatch):
    import asyncio

    import app.config as cfg
    import app.db as db_mod
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    from app.main import app
    asyncio.run(db_mod.init_db())
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("path", ["/onboarding", "/interests"])
def test_both_pickers_draw_every_area(client, path):
    html = client.get(path).text
    assert html.count('class="cat-art"') == len(CATEGORY_GROUPS)
    assert "cat-grid--art" in html
