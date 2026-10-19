"""
arXiv licences as short codes, and which of them show figures.

Two sources describe a paper's licence: the OAI-PMH harvest gives a URL
(http://creativecommons.org/licenses/by/4.0/) and every arXiv HTML page states
a label ("License: CC BY 4.0"). Both become one code, such as CC-BY-4.0.

Figure policy (FIGURE_LICENCES): the owner chose on 2026-10-19 to show figures
for every licence while ResearchIT is a personal project. arXiv's default
licence grants arXiv the right to distribute a paper, not anyone else, so set
FIGURE_LICENCES to a comma-separated list of codes (for example
"CC-BY-4.0,CC-BY-SA-4.0,CC0-1.0") to narrow it. Every displayed figure carries
a credit line either way.

Standard library only, so scripts can import it without the app's stack.
"""
from __future__ import annotations

import os
import re

ARXIV_DEFAULT = "ARXIV-NONEXCLUSIVE-1.0"

_LABELS = {
    "CC-BY-4.0": "CC BY 4.0", "CC-BY-SA-4.0": "CC BY-SA 4.0", "CC-BY-NC-4.0": "CC BY-NC 4.0",
    "CC-BY-NC-SA-4.0": "CC BY-NC-SA 4.0", "CC-BY-NC-ND-4.0": "CC BY-NC-ND 4.0",
    "CC-BY-ND-4.0": "CC BY-ND 4.0", "CC0-1.0": "CC0 1.0", ARXIV_DEFAULT: "arXiv licence",
}

_CC_URL = re.compile(r"creativecommons\.org/licenses/(by(?:-nc)?(?:-sa|-nd)?)/(\d\.\d)", re.I)
_CC_TEXT = re.compile(r"\bCC[\s-]+(BY(?:-NC)?(?:-SA|-ND)?)\s+(\d\.\d)\b", re.I)


def code_from_url(url: str | None) -> str:
    """OAI-PMH licence URL -> code; "" when missing or unrecognised."""
    u = (url or "").strip()
    if not u:
        return ""
    m = _CC_URL.search(u)
    if m:
        return f"CC-{m.group(1).upper()}-{m.group(2)}"
    if "publicdomain/zero" in u.lower():
        return "CC0-1.0"
    if "arxiv.org/licenses/nonexclusive-distrib" in u.lower():
        return ARXIV_DEFAULT
    return ""


def code_from_text(text: str | None) -> str:
    """The label on an arXiv HTML page -> code; "" when unrecognised."""
    t = re.sub(r"^\s*License:\s*", "", text or "", flags=re.I).strip()
    if not t:
        return ""
    m = _CC_TEXT.search(t)
    if m:
        return f"CC-{m.group(1).upper()}-{m.group(2)}"
    low = t.lower()
    if re.search(r"\bcc\s*0\b|cc zero|public domain", low):
        return "CC0-1.0"
    if "arxiv" in low and ("non-exclusive" in low or "nonexclusive" in low):
        return ARXIV_DEFAULT
    return ""


def label(code: str) -> str:
    """Display text for a code, such as "CC BY 4.0"."""
    return _LABELS.get(code, code.replace("-", " ") if code else "licence unknown")


def is_open(code: str) -> bool:
    """True for Creative Commons and public-domain licences."""
    return code.startswith("CC")


def figure_policy() -> str:
    return os.getenv("FIGURE_LICENCES", "all").strip()


def allows_figures(code: str) -> bool:
    """Whether figures of a paper under this licence may be shown."""
    policy = figure_policy()
    if policy.lower() == "all":
        return True
    allowed = {c.strip() for c in policy.split(",") if c.strip()}
    return code in allowed
