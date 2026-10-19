"""app/licences.py: one code per licence, whichever way arXiv describes it."""
import pytest

from app import licences


@pytest.mark.parametrize("url,code", [
    ("http://creativecommons.org/licenses/by/4.0/", "CC-BY-4.0"),
    ("https://creativecommons.org/licenses/by/4.0", "CC-BY-4.0"),
    ("http://creativecommons.org/licenses/by-sa/4.0/", "CC-BY-SA-4.0"),
    ("http://creativecommons.org/licenses/by-nc-sa/4.0/", "CC-BY-NC-SA-4.0"),
    ("http://creativecommons.org/licenses/by-nc-nd/4.0/", "CC-BY-NC-ND-4.0"),
    ("http://creativecommons.org/licenses/by/3.0/", "CC-BY-3.0"),
    ("http://creativecommons.org/publicdomain/zero/1.0/", "CC0-1.0"),
    ("http://arxiv.org/licenses/nonexclusive-distrib/1.0/", "ARXIV-NONEXCLUSIVE-1.0"),
    ("", ""), (None, ""), ("https://example.org/licence", ""),
])
def test_codes_from_urls(url, code):
    assert licences.code_from_url(url) == code


@pytest.mark.parametrize("text,code", [
    ("License: CC BY 4.0", "CC-BY-4.0"),
    ("CC BY 4.0", "CC-BY-4.0"),
    ("License: CC BY-SA 4.0", "CC-BY-SA-4.0"),
    ("License: CC BY-NC-SA 4.0", "CC-BY-NC-SA-4.0"),
    ("License: CC BY-NC-ND 4.0", "CC-BY-NC-ND-4.0"),
    ("License: CC Zero", "CC0-1.0"),
    ("License: arXiv.org perpetual non-exclusive license", "ARXIV-NONEXCLUSIVE-1.0"),
    ("", ""), ("License: something new", ""),
])
def test_codes_from_page_labels(text, code):
    assert licences.code_from_text(text) == code


def test_labels():
    assert licences.label("CC-BY-4.0") == "CC BY 4.0"
    assert licences.label("ARXIV-NONEXCLUSIVE-1.0") == "arXiv licence"
    assert licences.label("") == "licence unknown"


def test_every_licence_shows_figures_by_default(monkeypatch):
    """The owner's choice on 2026-10-19 while ResearchIT is a personal project."""
    monkeypatch.delenv("FIGURE_LICENCES", raising=False)
    for code in ("CC-BY-4.0", "CC-BY-NC-ND-4.0", "ARXIV-NONEXCLUSIVE-1.0", ""):
        assert licences.allows_figures(code)


def test_the_policy_can_be_narrowed(monkeypatch):
    monkeypatch.setenv("FIGURE_LICENCES", "CC-BY-4.0, CC-BY-SA-4.0,CC0-1.0")
    assert licences.allows_figures("CC-BY-4.0")
    assert licences.allows_figures("CC0-1.0")
    assert not licences.allows_figures("CC-BY-NC-SA-4.0")
    assert not licences.allows_figures("ARXIV-NONEXCLUSIVE-1.0")
    assert not licences.allows_figures("")
