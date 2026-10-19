"""app/features.py: the switches every redesign feature hides behind."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import config, features


def req(cookies=None):
    return SimpleNamespace(cookies=cookies or {})


@pytest.mark.parametrize("raw,expected", [
    ("off", "off"), ("shadow", "shadow"), ("preview", "preview"), ("on", "on"),
    ("ON", "on"), (" on ", "on"), ("1", "on"), ("true", "on"), ("0", "off"), ("", "off"),
])
def test_modes(monkeypatch, raw, expected):
    monkeypatch.setenv("SOME_FEATURE", raw)
    assert features.mode("SOME_FEATURE") == expected


def test_an_unset_switch_is_off(monkeypatch):
    monkeypatch.delenv("NEVER_SET_FEATURE", raising=False)
    assert features.mode("NEVER_SET_FEATURE") == "off"
    assert not features.active("NEVER_SET_FEATURE")


def test_an_unknown_value_is_off_with_one_warning(monkeypatch, capsys):
    monkeypatch.setenv("TYPO_FEATURE", "onn")
    features._warned.discard("TYPO_FEATURE")
    assert features.mode("TYPO_FEATURE") == "off"
    assert features.mode("TYPO_FEATURE") == "off"
    assert capsys.readouterr().out.count("TYPO_FEATURE") == 1


def test_shadow_is_active_but_not_visible(monkeypatch):
    monkeypatch.setenv("F", "shadow")
    assert features.active("F")
    assert not features.visible("F", req())


def test_preview_is_visible_only_with_the_owner_cookie(monkeypatch):
    monkeypatch.setattr(config, "PREVIEW_TOKEN", "s3cret")
    monkeypatch.setenv("F", "preview")
    assert not features.visible("F", req())
    assert not features.visible("F", req({features.PREVIEW_COOKIE: "s3cret"}))   # the raw token is not the cookie
    assert features.visible("F", req({features.PREVIEW_COOKIE: features._cookie_value()}))


def test_without_a_token_nobody_can_preview(monkeypatch):
    monkeypatch.setattr(config, "PREVIEW_TOKEN", "")
    monkeypatch.setenv("F", "preview")
    assert features._cookie_value() == ""
    assert not features.visible("F", req({features.PREVIEW_COOKIE: ""}))


def test_on_is_visible_to_everyone(monkeypatch):
    monkeypatch.setenv("F", "on")
    assert features.visible("F", None)


def test_the_right_token_sets_the_cookie_and_off_clears_it(monkeypatch):
    from app.main import app
    monkeypatch.setattr(config, "PREVIEW_TOKEN", "s3cret")
    with TestClient(app) as c:
        r = c.get("/healthz/client?preview=wrong")
        assert features.PREVIEW_COOKIE not in r.cookies
        r = c.get("/healthz/client?preview=s3cret")
        assert r.cookies.get(features.PREVIEW_COOKIE) == features._cookie_value()
        r = c.get("/healthz/client?preview=off")
        assert 'rit_preview=""' in r.headers.get("set-cookie", "") or "Max-Age=0" in r.headers.get("set-cookie", "")


def test_templates_can_ask_whether_a_feature_is_visible(monkeypatch):
    from app.templates_env import templates
    monkeypatch.setenv("F", "on")
    out = templates.env.from_string("{{ 'yes' if feature_visible('F', None) else 'no' }}").render()
    assert out == "yes"


@pytest.mark.parametrize("env,expected", [
    ({"RESEARCHIT_ENV": "test", "ARXIV_API_URL": "http://127.0.0.1:8799/api"}, "http://127.0.0.1:8799/api"),
    ({"RESEARCHIT_ENV": "", "ARXIV_API_URL": "http://evil.example/api"}, "https://export.arxiv.org/api/query"),
    ({"RESEARCHIT_ENV": "test", "ARXIV_HTML_BASE": "http://127.0.0.1:8799"}, None),
])
def test_arxiv_addresses_can_only_be_redirected_in_the_test_world(monkeypatch, env, expected):
    import importlib
    from app import config
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    try:
        fresh = importlib.reload(config)
        if expected:
            assert fresh.ARXIV_API_URL == expected
        else:
            assert fresh.ARXIV_HTML_BASE == "http://127.0.0.1:8799"
    finally:
        monkeypatch.undo()
        importlib.reload(config)

