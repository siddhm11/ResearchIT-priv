"""
Feature switches for the redesign.

Each feature reads one Space variable, so a bad change is undone by setting it
to `off` rather than by a code change. Modes:

    off      the feature does nothing (the default)
    shadow   it does its work, such as fetching or generating, but shows nothing
    preview  it shows only to the owner's browser (see below)
    on       everyone sees it

`preview` exists so the owner can use a feature on the live site before anyone
else does. Visiting any page with ?preview=<PREVIEW_TOKEN> sets a cookie on that
browser; ?preview=off clears it. The cookie holds a hash of the token rather
than the token itself, and with no PREVIEW_TOKEN configured nobody can preview.

Values are read at call time, not import time, so a test can set the
environment and a restart is the only thing a change needs.
"""
from __future__ import annotations

import hashlib
import hmac
import os

from app import config

MODES = ("off", "shadow", "preview", "on")
PREVIEW_COOKIE = "rit_preview"
PREVIEW_MAX_AGE = 30 * 24 * 3600

_warned: set[str] = set()


def mode(name: str) -> str:
    """The configured mode of switch `name`; anything unknown is `off`."""
    raw = os.getenv(name, "off").strip().lower()
    if raw in ("", "0", "false", "no"):
        return "off"
    if raw in ("1", "true", "yes"):
        return "on"
    if raw not in MODES:
        if name not in _warned:
            _warned.add(name)
            print(f"[features] WARNING: {name}={raw!r} is not one of {MODES}; treating it as off")
        return "off"
    return raw


def active(name: str) -> bool:
    """True when the feature should do its work, shown or not."""
    return mode(name) != "off"


def _cookie_value() -> str:
    token = config.PREVIEW_TOKEN
    if not token:
        return ""
    return hashlib.sha256(b"researchit-preview:" + token.encode()).hexdigest()[:32]


def has_preview(request) -> bool:
    expected = _cookie_value()
    if not expected or request is None:
        return False
    got = request.cookies.get(PREVIEW_COOKIE, "")
    return hmac.compare_digest(got, expected)


def visible(name: str, request=None) -> bool:
    """True when this request should see the feature."""
    m = mode(name)
    return m == "on" or (m == "preview" and has_preview(request))


def preview_cookie_action(query_value: str | None) -> str | None:
    """What a ?preview= value asks for: "set", "clear", or None (ignore it)."""
    if query_value is None:
        return None
    if query_value == "off":
        return "clear"
    token = config.PREVIEW_TOKEN
    if token and hmac.compare_digest(query_value, token):
        return "set"
    return None


def apply_preview_cookie(request, response) -> None:
    """Set or clear the preview cookie when the URL asks for it."""
    action = preview_cookie_action(request.query_params.get("preview"))
    if action == "set":
        response.set_cookie(PREVIEW_COOKIE, _cookie_value(), max_age=PREVIEW_MAX_AGE,
                            httponly=True, samesite="lax")
    elif action == "clear":
        response.delete_cookie(PREVIEW_COOKIE)
