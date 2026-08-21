"""
arXiv API service.

Responsibilities
────────────────
1. search(query)     – keyword search via export.arxiv.org/api/query
2. fetch_metadata()  – fetch a single paper's metadata by arxiv_id
3. fetch_metadata_batch() – fetch multiple papers, using SQLite cache first

ArXiv IDs come in two formats:
  Old: YYMM.NNNN   e.g. 0704.0002
  New: YYMM.NNNNN  e.g. 1706.03762
The arXiv API returns full URLs like http://arxiv.org/abs/1706.03762v5.
We always normalise to bare id (no version suffix, no URL prefix).
"""
import asyncio
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime

import httpx

from app import config
from app import http_client
from app import db

# XML namespace used in the Atom feed returned by arXiv API
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

# Prefixes an id may arrive wrapped in, longest first so the URL forms win.
_ID_PREFIXES = (
    "https://arxiv.org/abs/", "http://arxiv.org/abs/",
    "https://www.arxiv.org/abs/", "http://www.www.arxiv.org/abs/",
    "arxiv:", "arXiv:",
)

_VERSION_RE = re.compile(r"v\d+$")


def _normalise_id(raw: str) -> str:
    r"""Strip URL prefix and version suffix from an arxiv ID string.

    Old-style ids carry a category prefix — `math/0309136v1`,
    `hep-ph/0512038v2`, `acc-phys/9502001` — and 115,604 papers in the corpus
    (6.4%) use that form.

    The previous implementation matched `[^\s/v]+`, which stops at the slash, so
    every one of those normalised to just its category. `math/0309136v1` and
    `math/0511124v2` both became `math`: metadata for one paper could be
    returned for another, and `meta.get("math/0309136")` missed entirely, so the
    arXiv fallback silently returned nothing for 6.4% of the corpus. The
    character class also treated a literal `v` as a terminator, so any id
    containing one would have been truncated at it.

    Handled procedurally rather than with one regex because the two operations
    are independent — strip a known prefix, strip a trailing version — and a
    single pattern doing both is what hid the bug.

    Per CLAUDE.md §3.9 the result is always a string, never coerced.
    """
    text = raw.strip()
    lowered = text.lower()
    for prefix in _ID_PREFIXES:
        if lowered.startswith(prefix.lower()):
            text = text[len(prefix):]
            break
    return _VERSION_RE.sub("", text.strip())


def _parse_entry(entry: ET.Element) -> dict:
    """Convert one <entry> element into a paper dict."""
    def text(tag: str) -> str:
        el = entry.find(tag, _NS)
        return el.text.strip() if el is not None and el.text else ""

    raw_id = text("atom:id")
    arxiv_id = _normalise_id(raw_id)

    authors = [
        a.findtext("atom:name", namespaces=_NS, default="").strip()
        for a in entry.findall("atom:author", _NS)
    ]

    # Primary category
    cat_el = entry.find("arxiv:primary_category", _NS)
    category = cat_el.attrib.get("term", "") if cat_el is not None else ""

    published = text("atom:published")[:10]  # keep YYYY-MM-DD only
    year = int(published[:4]) if published else 0

    return {
        "arxiv_id": arxiv_id,
        "title": text("atom:title").replace("\n", " "),
        "abstract": text("atom:summary").replace("\n", " "),
        "authors": json.dumps(authors[:5]),   # store as JSON string
        "category": category,
        "published": published,
        "year": year,
    }


async def search(query: str, max_results: int = config.ARXIV_MAX_RESULTS) -> list[dict]:
    """
    Search arXiv and return a list of paper dicts with metadata.
    Results are also written into the metadata cache.
    """
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    resp = await http_client.get_client().get(
        config.ARXIV_API_URL, params=params, timeout=20)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    papers = [_parse_entry(e) for e in root.findall("atom:entry", _NS)]

    # Cache metadata for every result we got
    for paper in papers:
        await db.cache_metadata(paper)

    return papers


async def fetch_metadata(arxiv_id: str) -> dict | None:
    """
    Return metadata for a single paper.
    Checks the SQLite cache first; falls back to arXiv API.
    """
    cached = await db.get_cached_metadata(arxiv_id)
    if cached:
        return cached

    params = {"id_list": arxiv_id, "max_results": 1}
    resp = await http_client.get_client().get(
        config.ARXIV_API_URL, params=params, timeout=15)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    entries = root.findall("atom:entry", _NS)
    if not entries:
        return None

    paper = _parse_entry(entries[0])
    await db.cache_metadata(paper)
    return paper


async def fetch_metadata_batch(arxiv_ids: list[str]) -> dict[str, dict]:
    """
    Return {arxiv_id: metadata} for all IDs.
    Loads from cache where possible; fetches missing ones from arXiv API.
    Rate-limits arXiv API to 3 req/s as per their policy.
    """
    if not arxiv_ids:
        return {}

    result = await db.get_cached_metadata_batch(arxiv_ids)
    missing = [aid for aid in arxiv_ids if aid not in result]

    if missing:
        # arXiv allows up to 20 IDs per request
        BATCH = 20
        for i in range(0, len(missing), BATCH):
            chunk = missing[i : i + BATCH]
            params = {"id_list": ",".join(chunk), "max_results": len(chunk)}
            resp = await http_client.get_client().get(
                config.ARXIV_API_URL, params=params, timeout=20)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            for entry in root.findall("atom:entry", _NS):
                paper = _parse_entry(entry)
                await db.cache_metadata(paper)
                result[paper["arxiv_id"]] = paper
            if i + BATCH < len(missing):
                await asyncio.sleep(0.35)  # ~3 req/s

    return result
