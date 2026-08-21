"""
Comprehension surfaces on the paper page: reading density and the
plain-language explanation.

doc 01 names paper difficulty ratings as "a complete gap" nothing fills, and it
also says why the obvious approach fails — Flesch-Kincaid and friends are
calibrated on general prose and mislead badly on academic writing. So the
density signal deliberately does NOT claim to measure how hard a paper is to
UNDERSTAND; it measures how dense the abstract is to READ, which is honestly
computable from text alone.
"""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import db, groq_svc, readability, turso_svc
from app.main import app

PLAIN = ("We show that a simple change to how we train the model makes it work "
         "much better on new data. We tested this on three tasks. The results "
         "are consistent across all of them and easy to reproduce by anyone.")

DENSE = (r"Let $\mathcal{H}$ be a separable Hilbert space and $T \in "
         r"\mathcal{B}(\mathcal{H})$. We prove $\sigma(T) \subseteq "
         r"\overline{W(T)}$ where $W(T)$ denotes the numerical range, and "
         r"derive $\|T^n\| \leq C\rho^n$ for $\rho > r(T)$ via the "
         r"$\Gamma$-functional calculus and the ACRONYM SU(2) LMO framework.")


# ── Reading density ──────────────────────────────────────────────────────────

def test_notation_heavy_text_scores_above_prose():
    assert readability.score(DENSE) > readability.score(PLAIN)


def test_levels_are_ordered():
    assert readability.level(PLAIN) == "accessible"
    assert readability.level(DENSE) == "specialist"


def test_too_short_to_judge_returns_none():
    """Better to say nothing than to extrapolate from a fragment."""
    assert readability.level("Short.") is None
    assert readability.assess("") is None
    assert readability.assess(None) is None


def test_bands_actually_discriminate_on_the_real_corpus():
    """A label where most papers land in one band costs attention and returns
    nothing. The thresholds are corpus terciles for exactly this reason — an
    earlier guess of 0.34/0.50 put 82.5% in a single band.
    """
    import sqlite3
    from collections import Counter
    from app import config

    conn = sqlite3.connect(config.DB_PATH)
    abstracts = [r[0] for r in conn.execute(
        "SELECT abstract FROM paper_metadata "
        "WHERE abstract IS NOT NULL AND length(abstract) > 200 LIMIT 500")]
    if len(abstracts) < 100:
        pytest.skip("not enough cached abstracts to check the distribution")

    dist = Counter(readability.level(a) for a in abstracts)
    n = sum(dist.values())
    for lvl in readability.LEVELS:
        share = dist[lvl] / n
        assert 0.15 < share < 0.55, (
            f"{lvl} holds {share:.0%} of the corpus — the band does not "
            f"discriminate: {dict(dist)}")


def test_drivers_explain_the_score():
    """The badge must be explainable, not oracular."""
    assessed = readability.assess(DENSE)
    assert assessed["drivers"], "no reason given for a specialist rating"
    assert "notation" in assessed["drivers"]


def test_wording_is_comparative_not_absolute():
    """Terciles support only a relative claim about this corpus."""
    for lvl in readability.LEVELS:
        blurb = readability._BLURB[lvl]
        assert "arXiv abstract" in blurb, (
            f"{lvl} blurb makes an absolute claim rather than a comparison "
            f"against the corpus: {blurb!r}")


# ── Explanation caching ──────────────────────────────────────────────────────

def test_cache_key_changes_when_the_abstract_is_backfilled():
    """Keyed on the TEXT, so a repaired abstract is a new entry.

    Otherwise fixing the 500-char truncation would keep serving explanations
    generated from stumps.
    """
    stump = groq_svc.explain_cache_key("1706.03762", "a" * 500)
    full = groq_svc.explain_cache_key("1706.03762", "a" * 500 + " and the rest")
    assert stump != full


def test_cache_key_changes_with_prompt_version(monkeypatch):
    before = groq_svc.explain_cache_key("x", "abstract")
    monkeypatch.setattr(groq_svc, "_EXPLAIN_PROMPT_VERSION", "v2")
    assert groq_svc.explain_cache_key("x", "abstract") != before


async def test_explanation_roundtrips_through_the_cache():
    await db.init_db()
    await db.save_explanation("k-test", "1706.03762", "It explains attention.", "m")
    assert await db.get_explanation("k-test") == "It explains attention."
    assert await db.get_explanation("absent") is None


# ── The endpoint ─────────────────────────────────────────────────────────────
#
# The explanation cache is a real table in a persistent DB, and it is keyed on
# CONTENT — so two tests using the same fixture abstract share a cache entry and
# the second reads back the first's text. Clearing it per test is what makes
# "did this generate or did it hit the cache?" answerable at all.

@pytest.fixture(autouse=True)
async def _clear_explanation_cache():
    import aiosqlite
    from app import config
    await db.init_db()
    async with aiosqlite.connect(config.DB_PATH) as conn:
        await conn.execute("DELETE FROM paper_explanations")
        await conn.commit()
    yield

PAPER = {
    "arxiv_id": "1706.03762", "title": "Attention Is All You Need",
    "abstract": PLAIN * 2, "authors": '["A. Vaswani"]', "category": "cs.CL",
    "published": "2017-06-12", "year": 2017, "citation_count": 100,
}


async def _meta(ids):
    return {i: PAPER for i in ids if i == PAPER["arxiv_id"]}


def test_explain_generates_then_serves_from_cache():
    calls = []

    async def fake_explain(title, abstract):
        calls.append(title)
        return "This paper shows that attention alone is enough."

    with patch.object(turso_svc, "fetch_metadata_batch", side_effect=_meta), \
         patch.object(groq_svc, "explain_paper", side_effect=fake_explain):
        with TestClient(app) as c:
            first = c.get("/api/papers/1706.03762/explain")
            second = c.get("/api/papers/1706.03762/explain")

    assert "attention alone is enough" in first.text
    assert "attention alone is enough" in second.text
    assert len(calls) == 1, (
        f"generated {len(calls)} times — the cache is not being used")


def test_explain_renders_nothing_when_it_cannot_be_done_honestly():
    """A truncated abstract, no API key, a timeout — all return empty.

    An empty slot is quieter than an apology the reader cannot act on.
    """
    async def refuse(title, abstract):
        return None

    with patch.object(turso_svc, "fetch_metadata_batch", side_effect=_meta), \
         patch.object(groq_svc, "explain_paper", side_effect=refuse), \
         patch.object(db, "get_explanation", return_value=None):
        with TestClient(app) as c:
            r = c.get("/api/papers/1706.03762/explain")

    assert r.status_code == 200
    assert r.text.strip() == ""


def test_explanation_is_labelled_as_generated():
    """Never presented as the paper's own words."""
    async def gen(title, abstract):
        return "A plain summary."

    with patch.object(turso_svc, "fetch_metadata_batch", side_effect=_meta), \
         patch.object(groq_svc, "explain_paper", side_effect=gen), \
         patch.object(db, "get_explanation", return_value=None):
        with TestClient(app) as c:
            r = c.get("/api/papers/1706.03762/explain")

    assert "Generated from the abstract" in r.text


def test_the_paper_page_does_not_wait_on_generation():
    """Loaded by htmx after paint. Inline would put an 8s network ceiling in
    front of an abstract the reader already has."""
    import pathlib
    tpl = pathlib.Path("app/templates/paper.html").read_text()
    assert 'hx-get="/api/papers/{{ paper.arxiv_id }}/explain"' in tpl
    assert 'hx-trigger="load"' in tpl
