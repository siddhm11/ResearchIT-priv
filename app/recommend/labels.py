"""
Human-readable names for interest clusters.

Ward clustering produces numbered clusters. A number is correct and useless:
the whole point of the multi-interest pipeline — quota fusion with a floor,
K medoid queries, Hungarian ID stabilisation — is that a user's minority
interests survive into the feed, and the user cannot see that happening if
every cluster renders identically.

This module turns a cluster's member papers into a short label like
"Combinatorics & logic". It is deliberately **deterministic and offline**:

  * no LLM call, so it costs nothing in the hot path and cannot fail open
    into a wrong-but-confident name (doc 06 defers LLM cluster summaries to
    Phase 8; this is the cheap version that does not block on it)
  * derived from arXiv category codes, which are curated by the authors, not
    from free text, which is noisy

Title terms are used for ONE job only: breaking ties when two clusters would
otherwise get the same name. A user with two distinct cs.CV interests must not
see "Computer vision" twice — that would actively misrepresent the clustering.

`_why_shown()` in templates_env.py stays as the fallback sentence for papers
with no cluster (exploration, EWMA tiers); this covers the Tier 1 case where a
real cluster identity exists.
"""
from __future__ import annotations

import re
from collections import Counter

# ── arXiv category → human name ──────────────────────────────────────────────
#
# Exact codes first. Anything unmatched falls through to _FAMILY_NAMES on its
# archive prefix, so a code we have never seen still produces a sane family
# name rather than the raw code.

_CATEGORY_NAMES: dict[str, str] = {
    # Computer science
    "cs.AI": "AI & reasoning",
    "cs.CL": "Language & NLP",
    "cs.CV": "Computer vision",
    "cs.LG": "Machine learning",
    "cs.NE": "Neural computation",
    "cs.RO": "Robotics",
    "cs.IR": "Information retrieval",
    "cs.HC": "Human-computer interaction",
    "cs.CR": "Security & cryptography",
    "cs.DS": "Algorithms",
    "cs.DM": "Discrete mathematics",
    "cs.CC": "Computational complexity",
    "cs.LO": "Logic",
    "cs.PL": "Programming languages",
    "cs.SE": "Software engineering",
    "cs.DB": "Databases",
    "cs.DC": "Distributed computing",
    "cs.NI": "Networking",
    "cs.SY": "Systems & control",
    "cs.GT": "Game theory",
    "cs.CG": "Computational geometry",
    "cs.MA": "Multi-agent systems",
    "cs.SD": "Audio & speech",
    "cs.MM": "Multimedia",
    "cs.GR": "Graphics",
    "cs.AR": "Computer architecture",
    "cs.OS": "Operating systems",
    "cs.IT": "Information theory",
    "cs.SI": "Social networks",
    "cs.CY": "Computers & society",
    "cs.FL": "Formal languages",
    "cs.NA": "Numerical analysis",
    "cs.MS": "Mathematical software",
    "cs.SC": "Symbolic computation",
    "cs.ET": "Emerging technologies",
    "cs.PF": "Performance",
    "cs.DL": "Digital libraries",
    "cs.CE": "Computational science",
    # Statistics
    "stat.ML": "Statistical learning",
    "stat.ME": "Statistical methodology",
    "stat.TH": "Statistical theory",
    "stat.AP": "Applied statistics",
    "stat.CO": "Computational statistics",
    # Mathematics
    "math.CO": "Combinatorics",
    "math.LO": "Logic",
    "math.OC": "Optimization",
    "math.PR": "Probability",
    "math.ST": "Statistical theory",
    "math.NA": "Numerical analysis",
    "math.AG": "Algebraic geometry",
    "math.AT": "Algebraic topology",
    "math.NT": "Number theory",
    "math.DG": "Differential geometry",
    "math.GT": "Geometry & topology",
    "math.RT": "Representation theory",
    "math.GR": "Group theory",
    "math.RA": "Rings & algebras",
    "math.AC": "Commutative algebra",
    "math.CT": "Category theory",
    "math.DS": "Dynamical systems",
    "math.FA": "Functional analysis",
    "math.AP": "Differential equations",
    "math.CA": "Classical analysis",
    "math.MP": "Mathematical physics",
    "math.QA": "Quantum algebra",
    "math.SG": "Symplectic geometry",
    "math.SP": "Spectral theory",
    "math.IT": "Information theory",
    "math.HO": "History of mathematics",
    # Physics
    "quant-ph": "Quantum physics",
    "gr-qc": "Relativity & gravitation",
    "hep-th": "Theoretical high-energy",
    "hep-ph": "Particle phenomenology",
    "hep-ex": "Particle experiment",
    "hep-lat": "Lattice field theory",
    "nucl-th": "Nuclear theory",
    "nucl-ex": "Nuclear experiment",
    "math-ph": "Mathematical physics",
    # Quantitative biology / finance / economics
    "q-bio.NC": "Neuroscience",
    "q-bio.QM": "Quantitative methods",
    "q-bio.GN": "Genomics",
    "q-bio.PE": "Populations & evolution",
    "q-bio.BM": "Biomolecules",
    "q-bio.MN": "Molecular networks",
    "q-fin.PM": "Portfolio management",
    "q-fin.TR": "Trading & microstructure",
    "q-fin.ST": "Statistical finance",
    "econ.EM": "Econometrics",
    "econ.TH": "Economic theory",
    # Electrical engineering
    "eess.IV": "Image & video processing",
    "eess.SP": "Signal processing",
    "eess.AS": "Audio & speech",
    "eess.SY": "Systems & control",
}

# Archive-level fallback, matched on the part before the dot (or on the whole
# code for dashed archives like `cond-mat`).
_FAMILY_NAMES: dict[str, str] = {
    "cs": "Computer science",
    "math": "Mathematics",
    "stat": "Statistics",
    "physics": "Physics",
    "cond-mat": "Condensed matter",
    "astro-ph": "Astrophysics",
    "nlin": "Nonlinear systems",
    "q-bio": "Quantitative biology",
    "q-fin": "Quantitative finance",
    "econ": "Economics",
    "eess": "Electrical engineering",
}


def _head(name: str) -> str:
    """
    The leading segment of a category name: "Language & NLP" -> "Language".

    Only used when joining two categories. Several names are already compound,
    and pairing them whole produces "Machine learning & language & nlp" — a
    three-way string for a two-way fact. Reducing both sides to their head
    keeps every pair to one ampersand.
    """
    return (name or "").split(" & ", 1)[0]


def _as_tail(name: str) -> str:
    """
    Cased for the right-hand side of a pair: "Computer vision" -> "computer
    vision", but "AI" and "NLP" keep their capitals — an acronym lowercased is
    just a typo.
    """
    return name if name == name.upper() else name.lower()


def _name_for(code: str) -> str:
    """Human name for one arXiv category code, or '' if it is unrecognisable."""
    code = (code or "").strip()
    if not code:
        return ""
    if code in _CATEGORY_NAMES:
        return _CATEGORY_NAMES[code]
    archive = code.split(".", 1)[0]
    return _FAMILY_NAMES.get(archive, "")


# ── Title-term extraction (tie-breaking only) ────────────────────────────────
#
# Only ever used to separate two clusters that produced identical category
# labels. Held to a high bar: the term must recur across the cluster, so a
# one-off word from a single title can never become an interest name.

_STOPWORDS = frozenset("""
a an and are as at be by for from has have how in into is it its of on or over
that the their this to via with without using use used toward towards through
we our can new novel more most than then when where which while what who whom
approach approaches method methods model models framework frameworks system
systems study studies analysis based learning learn learned network networks
neural deep large small fast efficient effective robust general generalized
improved improving improvement performance results result paper towards toward
case cases problem problems solution solutions data set sets multi single
two three one first second third high low better best simple simplified
evaluation evaluating benchmark benchmarks survey review empirical
""".split())

_WORD_RE = re.compile(r"[a-z][a-z0-9-]{2,}")


def _distinctive_term(titles: list[str], avoid: set[str]) -> str:
    """
    The most repeated meaningful word across `titles`, or ''.

    Must appear in at least a third of the titles (and at least twice) to
    count — a term that shows up once is a coincidence, not an interest.
    """
    if len(titles) < 2:
        return ""
    seen_in: Counter[str] = Counter()
    for title in titles:
        words = {
            w for w in _WORD_RE.findall((title or "").lower())
            if w not in _STOPWORDS and w not in avoid
        }
        seen_in.update(words)

    threshold = max(2, (len(titles) + 2) // 3)
    for word, count in seen_in.most_common():
        if count < threshold:
            break
        return word
    return ""


# ── Public API ───────────────────────────────────────────────────────────────

# One tone per cluster slot. Cluster indices are stabilised by Hungarian
# matching across reclusterings (clustering.stabilize_cluster_ids), so a tone
# assigned from the index stays with the same interest between feeds — the
# colour is a real identity, not a per-request accident.
#
# MAX_CLUSTERS is 7, so seven tones cover every case; the modulo is a guard,
# not an expectation. Green is absent on purpose: it is reserved for Save.
N_TONES = 7


def cluster_tone(cluster_idx: int) -> int:
    """Palette slot (0..N_TONES-1) for a cluster index."""
    try:
        return int(cluster_idx) % N_TONES
    except (TypeError, ValueError):
        return 0


def _label_one(papers: list[dict]) -> str:
    """
    Category-derived name for a single cluster's papers.

    One dominant category gives its own name. Two comparable categories are
    joined ("Combinatorics & logic") — that pattern is common and real: a
    cluster is often the intersection of two fields rather than one.
    """
    counts: Counter[str] = Counter()
    for p in papers:
        # `category` is the paper's primary arXiv code (turso_svc sets it from
        # the first entry of `categories`); `arxiv_categories` holds them all.
        # Cross-listings say what a paper is *about* as much as its primary
        # does, so both are counted, with the primary weighted higher.
        primary = (p.get("category") or "").strip()
        if primary:
            counts[primary] += 2
        for code in (p.get("arxiv_categories") or "").split():
            if code and code != primary:
                counts[code] += 1

    named = [(code, n) for code, n in counts.most_common() if _name_for(code)]
    if not named:
        return "Mixed interests"

    total = sum(n for _, n in named)
    top_code, top_n = named[0]
    top_name = _name_for(top_code)

    # A clear majority stands alone.
    #
    # The threshold has to clear 2/3 to be useful. Primaries are weighted 2 and
    # cross-listings 1, so a cluster where every paper carries the SAME
    # cross-listing puts its primary at exactly 8/12 = 0.667 — and a universal
    # cross-listing is the most informative signal there is (cs.LG + cs.CL is
    # the standard pairing for language-model work). At 0.55 that pairing could
    # never surface and every such cluster reduced to "Machine learning".
    if total and top_n / total >= 0.70:
        return top_name

    # Otherwise pair the two leaders, skipping a second category that reduces
    # to the same head as the first (cs.LO and math.LO are both "Logic").
    top_head = _head(top_name)
    for code, _ in named[1:]:
        second_head = _head(_name_for(code))
        if second_head and second_head != top_head:
            return f"{top_head} & {_as_tail(second_head)}"

    return top_name


def label_clusters(cluster_papers: dict[int, list[dict]]) -> dict[int, str]:
    """
    Name each interest cluster from the papers that belong to it.

    Args:
        cluster_papers: cluster_idx -> metadata dicts for that cluster's
            papers. Each dict may carry `category`, `arxiv_categories` and
            `title`; missing keys are tolerated.

    Returns:
        cluster_idx -> label. **Labels are unique within the returned dict** —
        two clusters that reduce to the same category name are BOTH qualified
        by a recurring term from their titles ("Computer vision · diffusion"
        and "Computer vision · pose"), falling back to an ordinal only when no
        term recurs. Qualifying both rather than letting one keep the plain
        name is deliberate: "Computer vision" sitting next to "Computer vision
        · pose" reads as a general bucket and a special case, which is not what
        two sibling clusters are. Showing one name twice would misrepresent the
        clustering, which is the one thing this feature exists to convey.
    """
    if not cluster_papers:
        return {}

    base = {idx: _label_one(papers) for idx, papers in cluster_papers.items()}

    # Resolve collisions in a stable order so the same input always produces
    # the same output: the lowest cluster index keeps the plain name.
    by_label: dict[str, list[int]] = {}
    for idx in sorted(base):
        by_label.setdefault(base[idx], []).append(idx)

    out: dict[int, str] = {}
    for label, idxs in by_label.items():
        if len(idxs) == 1:
            out[idxs[0]] = label
            continue
        avoid = {w for w in label.lower().replace("&", " ").split() if w}
        used: set[str] = set()
        for idx in idxs:
            titles = [p.get("title") or "" for p in cluster_papers[idx]]
            term = _distinctive_term(titles, avoid | used)
            if term:
                used.add(term)
                out[idx] = f"{label} · {term}"
            else:
                out[idx] = f"{label} {idxs.index(idx) + 1}"
    return out
