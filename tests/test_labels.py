"""
Tests for app/recommend/labels.py — human names for interest clusters.

The property that matters most is UNIQUENESS. The feature exists to show that
the multi-interest pipeline keeps several distinct interests alive; rendering
two of them under the same name would misrepresent exactly the thing it is
meant to demonstrate.
"""
from app.recommend.labels import (
    N_TONES,
    cluster_tone,
    label_clusters,
    _distinctive_term,
    _label_one,
    _name_for,
)


def paper(category: str, *, also: str = "", title: str = "") -> dict:
    """Metadata dict shaped like turso_svc.fetch_metadata_batch returns."""
    return {
        "category": category,
        "arxiv_categories": f"{category} {also}".strip(),
        "title": title,
    }


# ── Category name resolution ─────────────────────────────────────────────────

def test_known_code_resolves_to_its_name():
    assert _name_for("cs.CV") == "Computer vision"
    assert _name_for("math.CO") == "Combinatorics"
    assert _name_for("quant-ph") == "Quantum physics"


def test_unknown_code_falls_back_to_its_archive_family():
    # cs.XX does not exist; the archive prefix still names it usefully.
    assert _name_for("cs.XX") == "Computer science"
    assert _name_for("astro-ph.GA") == "Astrophysics"


def test_unrecognisable_code_returns_empty():
    assert _name_for("") == ""
    assert _name_for(None) == ""
    assert _name_for("wingdings.ZZ") == ""


# ── Single-cluster labelling ─────────────────────────────────────────────────

def test_dominant_category_stands_alone():
    papers = [paper("cs.CV") for _ in range(5)]
    assert _label_one(papers) == "Computer vision"


def test_two_comparable_categories_are_joined():
    papers = [paper("math.CO"), paper("math.CO"), paper("math.LO"), paper("math.LO")]
    label = _label_one(papers)
    assert label == "Combinatorics & logic"


def test_second_category_mapping_to_the_same_name_is_skipped():
    # cs.LO and math.LO are both "Logic" — the pair must not read "Logic & logic".
    papers = [paper("cs.LO"), paper("cs.LO"), paper("math.LO"), paper("math.LO")]
    assert _label_one(papers) == "Logic"


def test_cross_listings_count_toward_the_label():
    # Primary is cs.LG throughout, but every paper cross-lists cs.CL, so the
    # cluster is really a language-modelling interest and should say so.
    # cs.LG + cs.CL is the standard pairing for language-model work, and at a
    # 2:1 primary weighting it lands at exactly 0.667 — the reason the
    # stands-alone threshold sits at 0.70 rather than 0.55.
    papers = [paper("cs.LG", also="cs.CL") for _ in range(4)]
    assert _label_one(papers) == "Machine learning & language"


def test_compound_names_are_reduced_to_one_ampersand():
    # "Language & NLP" paired whole would read "… & language & nlp".
    papers = [paper("cs.CL"), paper("cs.CL"), paper("cs.CV"), paper("cs.CV")]
    label = _label_one(papers)
    assert label.count("&") == 1, label
    assert label == "Language & computer vision"


def test_acronyms_keep_their_capitals_on_the_right_hand_side():
    # "AI & reasoning" reduces to head "AI", which must not become "ai".
    papers = [paper("cs.CV"), paper("cs.CV"), paper("cs.AI"), paper("cs.AI")]
    assert _label_one(papers) == "Computer vision & AI"


def test_no_recognisable_category_degrades_gracefully():
    assert _label_one([paper("zzz.QQ")]) == "Mixed interests"
    assert _label_one([]) == "Mixed interests"


def test_missing_keys_are_tolerated():
    assert _label_one([{}, {"title": "x"}]) == "Mixed interests"


# ── Uniqueness across clusters ───────────────────────────────────────────────

def test_distinct_clusters_keep_their_plain_names():
    out = label_clusters({
        0: [paper("cs.CV") for _ in range(4)],
        1: [paper("cs.CL") for _ in range(4)],
    })
    assert out == {0: "Computer vision", 1: "Language & NLP"}


def test_colliding_labels_are_separated_by_a_recurring_title_term():
    out = label_clusters({
        0: [paper("cs.CV", title=f"Diffusion sampling {i}") for i in range(4)],
        1: [paper("cs.CV", title=f"Pose estimation study {i}") for i in range(4)],
    })
    assert len(set(out.values())) == 2, out
    # BOTH are qualified. Leaving one plain would read as "the general one"
    # next to "the special case", which two sibling clusters are not.
    assert out[0].startswith("Computer vision · ")
    assert out[1].startswith("Computer vision · ")


def test_collision_falls_back_to_an_ordinal_when_no_term_recurs():
    out = label_clusters({
        0: [paper("cs.CV", title="Alpha"), paper("cs.CV", title="Beta")],
        1: [paper("cs.CV", title="Gamma"), paper("cs.CV", title="Delta")],
    })
    assert len(set(out.values())) == 2, out


def test_three_way_collision_still_yields_three_names():
    out = label_clusters({
        0: [paper("cs.LG", title=f"Optimization landscape {i}") for i in range(3)],
        1: [paper("cs.LG", title=f"Generalization bounds {i}") for i in range(3)],
        2: [paper("cs.LG", title=f"Quantization tradeoffs {i}") for i in range(3)],
    })
    assert len(set(out.values())) == 3, out


def test_labelling_is_deterministic():
    corpus = {
        0: [paper("cs.CV", title=f"Diffusion model {i}") for i in range(4)],
        1: [paper("cs.CV", title=f"Segmentation mask {i}") for i in range(4)],
        2: [paper("math.CO", title=f"Graph colouring {i}") for i in range(4)],
    }
    assert label_clusters(corpus) == label_clusters(corpus)


def test_empty_input():
    assert label_clusters({}) == {}


# ── Title-term extraction ────────────────────────────────────────────────────

def test_distinctive_term_ignores_stopwords():
    titles = ["Learning a model of the world", "Learning a model for control"]
    term = _distinctive_term(titles, avoid=set())
    # "learning", "model", "a", "the", "of", "for" are all stopwords.
    assert term in ("world", "control", "")


def test_distinctive_term_needs_repetition():
    # "quasar" appears once across four titles — below the one-third floor.
    titles = ["quasar spectra", "alpha", "beta", "gamma"]
    assert _distinctive_term(titles, avoid=set()) == ""


def test_distinctive_term_respects_avoid_set():
    titles = ["diffusion sampling one", "diffusion sampling two"]
    assert _distinctive_term(titles, avoid={"diffusion"}) == "sampling"


def test_distinctive_term_needs_at_least_two_titles():
    assert _distinctive_term(["a single lonely title"], avoid=set()) == ""


# ── Tones ────────────────────────────────────────────────────────────────────

def test_tone_is_stable_and_in_range():
    for idx in range(20):
        assert 0 <= cluster_tone(idx) < N_TONES
    assert cluster_tone(3) == cluster_tone(3 + N_TONES)


def test_tone_survives_bad_input():
    assert cluster_tone(None) == 0
    assert cluster_tone("nope") == 0
