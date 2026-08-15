"""
Shared Jinja2 environment with custom filters.
Import `templates` from here instead of creating it per-router.
"""
import json
import re

from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="app/templates")


def _tojson_parse(value: str | None) -> list:
    """Parse a JSON-encoded string into a Python list. Returns [] on error."""
    if not value:
        return []
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except (ValueError, TypeError):
        return []


# ── Abstract lead-sentence split ─────────────────────────────────────────────
#
# The paper card emphasises the abstract's first sentence and dims the rest.
# Two pieces of evidence drive this:
#
#   * Scholar Inbox (ACL 2025 Demo) highlights "the sentence most related to
#     the idea of the research paper" specifically to enable skim-reading on
#     mobile.  We have no relevance model over sentences, but an abstract's
#     opening sentence carries its claim almost always, so it is a cheap and
#     honest approximation of the same affordance.
#   * NN/g's F-pattern eyetracking finds attention decays sharply after the
#     first couple of lines, so the highest-signal text has to live there
#     rather than at character 400.
#
# Splitting on ". " naively breaks on the things abstracts are full of:
# decimals (0.95), initials (J. Smith), and abbreviations (e.g., et al.,
# Fig. 2).  The guard below handles those rather than shipping a lead that
# reads "We evaluate on MS."

_ABBREV = {
    "e.g", "i.e", "et al", "al", "cf", "vs", "resp", "approx", "ca",
    "fig", "figs", "eq", "eqs", "ref", "refs", "sec", "secs", "tab", "tabs",
    "no", "nos", "vol", "pp", "ch", "dr", "prof", "mr", "mrs", "ms", "st",
    "inc", "ltd", "etc", "vs", "w.r.t", "s.t", "i.i.d",
}

# A sentence end: '.', '!' or '?' then whitespace then something that starts a
# new sentence (capital, digit, quote, or an opening bracket).
_SENT_END = re.compile(r'([.!?])\s+(?=[A-Z0-9"“(\[])')

_MIN_LEAD = 40    # below this the "sentence" is a fragment, not a claim
_MAX_LEAD = 340   # above this it stops being a scannable lead


def _is_false_stop(text: str, dot_index: int) -> bool:
    """True when the '.' at dot_index does not actually end a sentence."""
    before = text[:dot_index]

    # Decimal number: "0.95", "1.5x"
    if before and before[-1].isdigit() and dot_index + 1 < len(text) \
            and text[dot_index + 1: dot_index + 2].isdigit():
        return True

    # Trailing token, lowercased, without its dots — "e.g", "et al", "fig"
    token = re.split(r"[\s(\[]", before)[-1].rstrip(".").lower()
    if token in _ABBREV:
        return True

    # Single-letter initial: "J. Smith", "A. Vaswani"
    if len(token) == 1 and token.isalpha():
        return True

    return False


def _lead_split(abstract: str | None) -> tuple[str, str]:
    """
    Split an abstract into (lead_sentence, remainder).

    Returns ("", "") for empty input, and (whole, "") when no confident
    sentence boundary is found — callers then render the whole abstract
    unemphasised rather than guessing.
    """
    text = (abstract or "").strip()
    if not text:
        return "", ""

    for match in _SENT_END.finditer(text):
        end = match.end(1)          # index just past the '.'/'!'/'?'
        if _is_false_stop(text, match.start(1)):
            continue
        if end < _MIN_LEAD:
            continue
        if end > _MAX_LEAD:
            break
        return text[:end].strip(), text[end:].strip()

    # No usable boundary. If the abstract is short enough to read whole, treat
    # it as all-lead; otherwise return it unsplit so nothing is emphasised.
    if len(text) <= _MAX_LEAD:
        return text, ""
    return "", text


def _lead(abstract: str | None) -> str:
    return _lead_split(abstract)[0]


def _lead_rest(abstract: str | None) -> str:
    return _lead_split(abstract)[1]


# ── arXiv category → chip class ──────────────────────────────────────────────
#
# Maps a primary arXiv code onto one of the ten chip hues in styles.css.
# Lives here rather than in the template because the template was carrying a
# ten-branch if/elif chain that had to be duplicated wherever a card appeared.
# Green is deliberately unused — it is reserved for Save.

_CAT_RULES: list[tuple[tuple[str, ...], str]] = [
    (("cs.CL", "cs.IR"), "cat-cl"),
    (("cs.CV",), "cat-cv"),
    (("cs.LG", "stat.ML", "stat."), "cat-lg"),
    (("cs.AI", "cs.NE", "cs.MA"), "cat-ai"),
    (("cs.RO", "cs.SY", "eess.SY"), "cat-ro"),
    (("astro-ph",), "cat-astro"),
    (("hep", "quant-ph", "physics", "cond-mat", "nucl", "gr-qc", "nlin"), "cat-phys"),
    (("math", "cs.GT", "cs.DM", "cs.CC", "econ", "q-fin"), "cat-math"),
    (("q-bio",), "cat-bio"),
    (("eess", "cs.SD", "cs.SE", "cs.PL", "cs.CR", "cs.DC",
      "cs.NI", "cs.HC", "cs.DB", "cs.OS"), "cat-eng"),
]


def _cat_class(category: str | None) -> str:
    """Return the chip class for an arXiv category code."""
    code = (category or "").strip()
    if not code:
        return "cat"
    for prefixes, cls in _CAT_RULES:
        for p in prefixes:
            if code.startswith(p):
                return cls
    return "cat"


# ── "Why am I seeing this?" ──────────────────────────────────────────────────
#
# The serving tier already records WHY a paper was retrieved, per candidate, in
# `candidate_source` — it just never reached the user, who instead saw a
# "Why this paper? 🔒 Coming Soon" badge on every card. This turns the existing
# value into a sentence.
#
# Deliberately vague about cluster identity: clusters are numbered, not named
# (labelling them is Phase 8), so claiming more than "one of your recurring
# interests" would be inventing detail the system does not have.

_WHY = {
    "exploration": "Outside your usual areas — included to keep your feed from narrowing.",
    "short_term_supplement": "Based on what you have been reading in this session.",
    "ewma_longterm": "Close to the overall profile built from your saved papers.",
    "qdrant_recommend": "Similar to papers already in your library.",
    "trending_category_fallback": "Widely cited recently in the areas you picked during setup.",
}


def _why_shown(candidate_source: str | None) -> str:
    """Human-readable reason a paper appeared, or '' when there isn't one."""
    src = (candidate_source or "").strip()
    if not src:
        return ""
    if src.startswith("cluster_"):
        return "Matches one of the recurring interests in your library."
    return _WHY.get(src, "")


# ── Match meter ──────────────────────────────────────────────────────────────
#
# Scholar Inbox — 23k users, a 1,233-participant study — renders each paper's
# relevance as a coloured card header, i.e. as a visible quantity rather than a
# hidden one. This pipeline computes the same thing (cosine against the cluster
# medoid that retrieved the paper, reranker feature 0) and then dropped it at
# render time.
#
# The number shown is the RAW cosine, unmassaged. Only the bar's fill is
# rescaled, because BGE-M3 cosines over this corpus occupy roughly [.35, .90]
# and a bar drawn on the full 0–1 range would sit between a third and nine
# tenths full for every paper ever retrieved — visually identical, therefore
# useless. Clamping to the band the data actually occupies is what makes the
# comparison between two cards legible.

_MATCH_FLOOR = 0.35
_MATCH_CEIL = 0.90


def _match_pct(value) -> int:
    """Cosine -> 0..100 bar fill across the band real scores occupy."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0
    if score <= 0:
        return 0
    span = _MATCH_CEIL - _MATCH_FLOOR
    pct = (score - _MATCH_FLOOR) / span * 100.0
    return int(max(0.0, min(100.0, pct)))


def _commafy(value) -> str:
    """1204 -> '1,204'. Returns '' for anything non-numeric."""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return ""


templates.env.filters["tojson_parse"] = _tojson_parse
templates.env.filters["lead"] = _lead
templates.env.filters["lead_rest"] = _lead_rest
templates.env.filters["cat_class"] = _cat_class
templates.env.filters["why_shown"] = _why_shown
templates.env.filters["commafy"] = _commafy
templates.env.filters["match_pct"] = _match_pct


# ── 3D map links (from the Space build) ──────────────────────────────────────
#
# Exposed to every template so the map links do not have to be threaded through
# each route's context dict. Both are inert when SPACE_APP_URL is unset: the
# templates test them and render nothing, so an undeployed map leaves no dead
# links behind.
from app import config  # noqa: E402  (after `templates` exists, avoids a cycle)

templates.env.globals["space_app_url"] = config.SPACE_APP_URL
templates.env.globals["space_paper_url"] = config.space_paper_url
