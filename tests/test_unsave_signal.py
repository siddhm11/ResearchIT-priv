"""
Un-saving must not be recorded as a dislike.

"Remove" on an already-saved card posted to /not-interested, because that was
the only unwind path that existed. So correcting a misclick — or tidying the
library — was logged as a `not_interested` interaction, added to the negative
deque, and folded into the negative EWMA profile that heuristic_score subtracts
at 0.15.

Those are different acts. "I did not mean to save this" is not "show me less
like this", and conflating them poisons both the only signal the system has for
genuine dislike and the interaction log any future ranker trains on.
"""
import pathlib
import re

from fastapi.testclient import TestClient

from app import db, user_state as us
from app.main import app


def test_unsave_removes_the_paper_without_recording_a_dislike():
    uid = "unsave-user"
    with TestClient(app) as c:
        c.cookies.set("arxiv_user_id", uid)
        c.post("/api/papers/2401.00001/save", data={"source": "search"})
        assert "2401.00001" in us.get_user_state(uid).positive_list

        r = c.post("/api/papers/2401.00001/unsave", data={"source": "saved"})
        assert r.status_code == 200

    state = us.get_user_state(uid)
    assert "2401.00001" not in state.positive_list, "paper stayed in the library"
    assert "2401.00001" not in state.negative_list, (
        "un-saving was recorded as a dislike")


def test_dismiss_still_records_a_dislike():
    """The fix must not disarm genuine negative signal."""
    uid = "dismiss-user"
    with TestClient(app) as c:
        c.cookies.set("arxiv_user_id", uid)
        c.post("/api/papers/2401.00002/not-interested", data={"source": "search"})

    assert "2401.00002" in us.get_user_state(uid).negative_list


def test_unsave_is_logged_under_its_own_event_type():
    """An undo and a dislike must be distinguishable in the training data."""
    uid = "unsave-log-user"
    with TestClient(app) as c:
        c.cookies.set("arxiv_user_id", uid)
        c.post("/api/papers/2401.00003/save", data={"source": "search"})
        c.post("/api/papers/2401.00003/unsave", data={"source": "saved"})

    import sqlite3
    from app import config
    conn = sqlite3.connect(config.DB_PATH)
    rows = conn.execute(
        "SELECT event_type FROM interactions WHERE user_id = ? AND paper_id = ?",
        (uid, "2401.00003")).fetchall()
    kinds = {r[0] for r in rows}
    assert "unsave" in kinds
    assert "not_interested" not in kinds


def test_the_remove_button_names_the_action_it_performs():
    html = pathlib.Path("app/templates/partials/action_buttons.html").read_text()
    saved_block = html.split("{% if is_saved %}")[1].split("{% else %}")[0]
    assert 'data-action="unsave"' in saved_block, (
        "Remove on a saved card does not declare itself an unsave, so app.js "
        "falls back to /not-interested")


def test_the_client_routes_unsave_separately():
    js = pathlib.Path("app/static/app.js").read_text()
    assert "'/unsave'" in js or "'unsave'" in js
    assert re.search(r"function endpointFor\(id, action\)", js), (
        "endpointFor still hardcodes a single endpoint")


def test_loading_indicators_can_actually_appear():
    """htmx puts `htmx-request` on the indicator ELEMENT, not an ancestor.

    Every hx-indicator in this codebase points at a separate element, so the
    node ends up with both classes. A descendant selector cannot match that,
    and search plus the onboarding seed step ran with no loading state at all.
    """
    css = pathlib.Path("app/static/styles.css").read_text()
    assert ".htmx-indicator.htmx-request" in css, (
        "no compound selector — the spinner can never show")
