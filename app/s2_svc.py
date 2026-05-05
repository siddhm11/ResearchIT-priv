"""
Semantic Scholar service — Phase 5.1 (author import for onboarding).

Accepts an S2 author URL, a raw S2 author ID, or an ORCID, then
fetches that author's papers and returns arXiv IDs for auto-saving.

API docs: https://api.semanticscholar.org/api-docs/graph
"""
from __future__ import annotations

import re
import httpx
from app.config import S2_API_KEY

_BASE = "https://api.semanticscholar.org/graph/v1"
_TIMEOUT = 15.0  # seconds

# ── Patterns ──────────────────────────────────────────────────────────────────
#   URL:   https://www.semanticscholar.org/author/Yoshua-Bengio/1751762
#   Raw:   1751762
#   ORCID: 0000-0003-3394-6622
_S2_URL_RE = re.compile(
    r"semanticscholar\.org/author/[^/]+/(\d+)", re.IGNORECASE
)
_ORCID_RE = re.compile(r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]")
_RAW_ID_RE = re.compile(r"^\d{3,}$")  # 3+ digits = plausible S2 author ID


def _headers() -> dict[str, str]:
    """Build request headers, including API key if available."""
    h: dict[str, str] = {"Accept": "application/json"}
    if S2_API_KEY:
        h["x-api-key"] = S2_API_KEY
    return h


# ── Public API ────────────────────────────────────────────────────────────────

def parse_author_input(text: str) -> tuple[str | None, str]:
    """Parse user-provided text into an S2 author ID or ORCID.

    Returns (s2_author_id | None, input_type) where input_type is one of:
      "s2_url", "s2_id", "orcid", "unknown"
    """
    text = text.strip()
    if not text:
        return None, "unknown"

    # 1. Try S2 URL
    m = _S2_URL_RE.search(text)
    if m:
        return m.group(1), "s2_url"

    # 2. Try ORCID
    m = _ORCID_RE.search(text)
    if m:
        return m.group(0), "orcid"

    # 3. Try raw numeric ID
    if _RAW_ID_RE.match(text):
        return text, "s2_id"

    return None, "unknown"


async def resolve_orcid(orcid: str) -> str | None:
    """Resolve an ORCID to an S2 author ID via the author search endpoint.

    Returns the S2 authorId string or None if not found.
    """
    url = f"{_BASE}/author/search"
    params = {"query": orcid, "limit": 1}
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(url, params=params, headers=_headers())
        resp.raise_for_status()
        data = resp.json()
        authors = data.get("data", [])
        if authors:
            return str(authors[0]["authorId"])
    return None


async def fetch_author_arxiv_papers(
    author_id: str, limit: int = 50,
) -> list[str]:
    """Fetch an author's papers from S2 and return arXiv IDs.

    Filters to papers that have an ArXiv external ID.
    Returns at most `limit` arXiv IDs, ordered by citation count (desc).
    """
    url = f"{_BASE}/author/{author_id}/papers"
    params = {
        "fields": "externalIds,citationCount",
        "limit": min(limit * 2, 500),  # over-fetch since not all have arXiv IDs
    }
    arxiv_ids: list[tuple[int, str]] = []  # (citation_count, arxiv_id)

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(url, params=params, headers=_headers())
        resp.raise_for_status()
        data = resp.json()
        for paper in data.get("data", []):
            ext = paper.get("externalIds") or {}
            arxiv_id = ext.get("ArXiv")
            if arxiv_id:
                cites = paper.get("citationCount") or 0
                arxiv_ids.append((cites, arxiv_id))

    # Sort by citation count descending so we import the most impactful first
    arxiv_ids.sort(key=lambda x: x[0], reverse=True)
    return [aid for _, aid in arxiv_ids[:limit]]
