"""
The card must not misrepresent the abstract it was given.

Two defects, both measured against the 509 real abstracts cached in
interactions.db:

  * 79% of stored abstracts are cut at the 500-character ingest cap and 99.3%
    of those stop without terminal punctuation — one ends "...we study a
    parameter". The card rendered that stump with no ellipsis and no marker, so
    a reader could not tell the text had been cut, and "Read more" expanded to
    reveal only more stump.
  * `_lead_split` returns ("", whole) when it finds no confident sentence
    boundary — 1.6% of real abstracts. The card then emitted
    `<p class="card-abstract is-clamped">` containing only `<span class="rest">`,
    and `.is-clamped .rest` is `display:none`, so the abstract rendered as an
    entirely blank paragraph with a "Read more" button underneath.

Repairing the DATA is a PHASE7 arXiv backfill. These tests pin the honesty of
the rendering, which is fixable now.
"""
import re

from app.templates_env import _is_truncated, _lead_split, templates


# ── the detector ─────────────────────────────────────────────────────────────

def test_short_complete_abstract_is_not_truncated():
    assert _is_truncated("We prove a theorem about graphs.") is False


def test_capped_midword_abstract_is_truncated():
    assert _is_truncated("x" * 499 + "r") is True


def test_cap_is_measured_before_stripping():
    """The cut often lands on a space.

    402 of the 406 capped rows in the local cache strip down to 499 characters,
    so stripping before the length check would let almost every real truncation
    through untested.
    """
    assert _is_truncated("x" * 490 + " parameter ") is True


def test_abstract_ending_in_punctuation_at_the_cap_is_not_flagged():
    assert _is_truncated("x" * 499 + ".") is False


def test_detector_tolerates_empty_input():
    assert _is_truncated(None) is False
    assert _is_truncated("") is False
    assert _is_truncated("   ") is False


# ── the rendering ────────────────────────────────────────────────────────────

def _render(abstract: str) -> str:
    tpl = templates.env.get_template("partials/paper_card.html")
    return tpl.render(
        paper={
            "arxiv_id": "2401.00001", "title": "A Paper", "abstract": abstract,
            "authors": '["A. Author"]', "category": "cs.LG", "published": "2024-01-01",
            "citation_count": 3, "saved": False,
        },
        source="recommendation", position=0,
    )


def test_truncated_abstract_is_marked_and_offers_the_full_text():
    html = _render("We study a hard problem" + "x" * 480 + " and we study a parameter")
    assert "abstract-cut" in html, "no ellipsis marking the cut"
    assert "Full abstract on arXiv" in html, "no route to the untruncated text"


def test_complete_abstract_is_not_marked():
    html = _render("We prove a short theorem. It is correct and complete.")
    assert "abstract-cut" not in html
    assert "Full abstract on arXiv" not in html


def test_abstract_with_no_sentence_boundary_still_renders_its_text():
    """The blank-card bug: clamping with no lead hides the only content."""
    # Long, and with no ". " boundary, so _lead_split yields ("", whole).
    abstract = "we study " + "a very long unpunctuated clause about graphs " * 12
    assert _lead_split(abstract)[0] == "", "fixture no longer exercises the bug"

    html = _render(abstract)
    body = re.search(r'<p class="card-abstract[^"]*"[^>]*>(.*?)</p>', html, re.S)
    assert body, "no abstract paragraph rendered at all"
    assert "is-clamped" not in body.group(0), (
        "clamped with no lead sentence — .is-clamped .rest is display:none, so "
        "this renders an empty paragraph"
    )
    assert "graphs" in body.group(1), "abstract text is missing from the card"


def test_normal_abstract_still_clamps_and_offers_read_more():
    """The clamp must survive for the ordinary case it exists to serve."""
    abstract = ("We introduce a new method for graph learning. " +
                "It works well in practice and we evaluate it broadly. " * 4)
    lead, rest = _lead_split(abstract)
    assert lead and rest, "fixture should split into lead + remainder"

    html = _render(abstract)
    assert "is-clamped" in html
    assert "Read more" in html
