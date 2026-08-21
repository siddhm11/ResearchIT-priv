"""
How dense is this abstract to read?

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
doc 01 names paper difficulty ratings as "a complete gap" that nothing fills,
and it also says why the obvious approach fails: standard readability formulas
(Flesch-Kincaid, FORCAST) are calibrated on general prose and are inadequate for
academic writing full of domain jargon and mathematical notation. A paper can be
conceptually simple and score as unreadable, or conceptually brutal and score as
plain.

So this deliberately does NOT claim to measure how hard a paper is to
UNDERSTAND. That would need a model trained on researcher judgements, which
does not exist here. It measures how dense the abstract is to READ — notation,
sentence length, jargon rate — which is a real and separate thing, is honestly
computable from text alone, and is what a reader deciding whether to open
something actually wants to know.

The UI wording follows from that: "dense with notation", not "hard".

CALIBRATION
-----------
The band thresholds are the terciles of the real corpus, not round numbers
chosen by eye. A scale where 90% of papers land in one band tells a reader
nothing; terciles guarantee the label discriminates. See
`scripts/calibrate_readability.py` for the derivation and
tests/test_readability.py for the distribution assertion.

No LLM, no network, no dependencies. It runs in microseconds, so it can be
computed inline on any surface.
"""
from __future__ import annotations

import re

# ── Signals ──────────────────────────────────────────────────────────────────

# Mathematical notation: LaTeX delimiters, sub/superscripts, operators, and the
# Greek letters that survive into plain-text abstracts.
_MATH = re.compile(r"[$\\^_{}]|\\[a-zA-Z]+|[=<>≤≥≈∈∀∃∑∏∫∇×·±→←↔⊗⊕]|"
                   r"[αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ]")

# An acronym: two or more capitals in a row, optionally with digits (BERT, GAN,
# SU(2), LSTM). These are the single clearest marker of assumed background.
_ACRONYM = re.compile(r"\b[A-Z]{2,}[0-9]*\b")

_SENTENCE = re.compile(r"[.!?](?:\s|$)")
_WORD = re.compile(r"[A-Za-z][A-Za-z\-']*")

# Long words stand in for morphological complexity — "convolutional",
# "regularisation", "heteroskedasticity". Cheaper than syllable counting and
# more robust on technical vocabulary, where syllable heuristics do badly.
_LONG_WORD_CHARS = 11

# Terciles of the real corpus (n=525 cached abstracts), not round numbers.
# My first guess at these was 0.34 / 0.50, which put 82.5% of papers in a
# single band — a label that never discriminates is worse than no label, since
# it costs the reader attention and returns nothing.
_TECHNICAL_AT = 0.181
_SPECIALIST_AT = 0.259

LEVELS = ("accessible", "technical", "specialist")

_LABEL = {
    "accessible": "Plainly written",
    "technical": "Technical",
    "specialist": "Dense with notation",
}

# Wording is deliberately COMPARATIVE. The bands are terciles of this corpus,
# so the only claim they can honestly support is "relative to other arXiv
# abstracts" — not an absolute statement about difficulty.
_BLURB = {
    "accessible": "Lighter on notation and acronyms than most arXiv abstracts.",
    "technical": "About as dense as a typical arXiv abstract.",
    "specialist": "Denser than most arXiv abstracts — heavy notation or "
                  "assumed vocabulary.",
}


def signals(text: str) -> dict:
    """The four raw measurements, each normalised to roughly 0-1."""
    text = (text or "").strip()
    words = _WORD.findall(text)
    n_words = len(words)
    if n_words < 20:
        # Too little text to say anything. Returning zeros rather than
        # extrapolating from a fragment keeps the caller honest.
        return {"math": 0.0, "acronym": 0.0, "sentence": 0.0, "long_word": 0.0,
                "n_words": n_words}

    n_sentences = max(1, len(_SENTENCE.findall(text)))
    long_words = sum(1 for w in words if len(w) >= _LONG_WORD_CHARS)

    return {
        # Notation marks per word, saturating at 1 mark every 5 words.
        "math": min(1.0, len(_MATH.findall(text)) / n_words * 5),
        # Acronyms per word, saturating at 1 in 12.
        "acronym": min(1.0, len(_ACRONYM.findall(text)) / n_words * 12),
        # Mean sentence length, mapped so 12 words -> 0 and 40 -> 1.
        "sentence": min(1.0, max(0.0, (n_words / n_sentences - 12) / 28)),
        # Long-word share, saturating at 25%.
        "long_word": min(1.0, long_words / n_words * 4),
        "n_words": n_words,
    }


def score(text: str) -> float:
    """A single 0-1 density score.

    Weights are a judgement, not a fit: notation and assumed vocabulary are what
    actually stop a reader who lacks the background, while long sentences are
    merely tiring. There is no labelled data to fit against — saying so is more
    useful than implying a precision this does not have.
    """
    s = signals(text)
    if s["n_words"] < 20:
        return 0.0
    return (
        0.35 * s["math"]
        + 0.30 * s["acronym"]
        + 0.20 * s["long_word"]
        + 0.15 * s["sentence"]
    )


def level(text: str) -> str | None:
    """One of LEVELS, or None when there is too little text to judge."""
    s = signals(text)
    if s["n_words"] < 20:
        return None
    v = score(text)
    if v >= _SPECIALIST_AT:
        return "specialist"
    if v >= _TECHNICAL_AT:
        return "technical"
    return "accessible"


def assess(text: str) -> dict | None:
    """Everything the UI needs, or None when the text is too short to judge."""
    lvl = level(text)
    if lvl is None:
        return None
    s = signals(text)
    return {
        "level": lvl,
        "label": _LABEL[lvl],
        "blurb": _BLURB[lvl],
        "score": round(score(text), 3),
        # What actually drove it, so the badge is explainable rather than
        # oracular. Only the signals that are genuinely elevated.
        "drivers": [
            name for name, threshold in (
                ("notation", 0.35), ("acronyms", 0.35),
                ("long words", 0.5), ("long sentences", 0.5),
            )
            if s[{"notation": "math", "acronyms": "acronym",
                  "long words": "long_word",
                  "long sentences": "sentence"}[name]] >= threshold
        ],
    }
