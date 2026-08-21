"""
Tests for arxiv_svc.py.
- ID normalisation (unit, no network)
- XML parsing (unit, no network)
- search() and fetch_metadata() (integration, hits real arXiv API)
"""
import json
import pytest

from app.arxiv_svc import _normalise_id, _parse_entry, _NS
import xml.etree.ElementTree as ET


# ── Pure unit tests (no I/O) ──────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("http://arxiv.org/abs/1706.03762v5", "1706.03762"),
    ("https://arxiv.org/abs/1706.03762",  "1706.03762"),
    ("arxiv:1706.03762v2",                "1706.03762"),
    ("1706.03762v3",                      "1706.03762"),
    ("1706.03762",                        "1706.03762"),
    ("0704.0002",                         "0704.0002"),
    ("http://arxiv.org/abs/0704.0002v1",  "0704.0002"),
    # Old-style ids carry a CATEGORY PREFIX, and 115,604 papers in the corpus
    # (6.4%) use that form. The previous regex matched [^\s/v]+, which stops at
    # the slash, so every one of these normalised to just its category:
    # math/0309136v1 and math/0511124v2 both became "math". Metadata for one
    # paper could be returned for another, and lookups by the real id missed
    # entirely, so the arXiv fallback silently returned nothing for 6.4% of the
    # corpus. None of these cases was covered, which is why it survived.
    ("http://arxiv.org/abs/math/0309136v1",   "math/0309136"),
    ("http://arxiv.org/abs/hep-ph/0512038v2", "hep-ph/0512038"),
    ("acc-phys/9502001",                      "acc-phys/9502001"),
    ("cond-mat/0211034v11",                   "cond-mat/0211034"),
    ("arxiv:math/0309136v1",                  "math/0309136"),
    # A literal 'v' inside the id must not act as a terminator either.
    ("nlin/0507021v1",                        "nlin/0507021"),
])
def test_normalise_id(raw, expected):
    assert _normalise_id(raw) == expected


_SAMPLE_ENTRY_XML = """
<entry xmlns="http://www.w3.org/2005/Atom"
       xmlns:arxiv="http://arxiv.org/schemas/atom">
  <id>http://arxiv.org/abs/1706.03762v5</id>
  <title>Attention Is All You Need</title>
  <summary>The dominant sequence transduction models are based on complex neural networks.</summary>
  <published>2017-06-12T00:00:00Z</published>
  <author><name>Ashish Vaswani</name></author>
  <author><name>Noam Shazeer</name></author>
  <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.CL"/>
</entry>
"""

def test_parse_entry():
    entry = ET.fromstring(_SAMPLE_ENTRY_XML)
    paper = _parse_entry(entry)
    assert paper["arxiv_id"] == "1706.03762"
    assert paper["title"] == "Attention Is All You Need"
    assert "dominant sequence" in paper["abstract"]
    assert paper["category"] == "cs.CL"
    assert paper["published"] == "2017-06-12"
    assert paper["year"] == 2017
    authors = json.loads(paper["authors"])
    assert "Ashish Vaswani" in authors
    assert "Noam Shazeer" in authors


# ── Integration tests (hit real arXiv API) ───────────────────────────────────
# These are skipped in CI if the API is unreachable.

@pytest.mark.asyncio
async def test_fetch_metadata_known_paper(tmp_path, monkeypatch):
    """Fetch metadata for 'Attention is All You Need'."""
    import app.config as cfg
    import app.db as db_mod
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    await db_mod.init_db()

    from app.arxiv_svc import fetch_metadata
    paper = await fetch_metadata("1706.03762")
    assert paper is not None
    assert "1706.03762" in paper["arxiv_id"]
    assert "Attention" in paper["title"]
    assert paper["category"] != ""


@pytest.mark.asyncio
async def test_search_returns_results(tmp_path, monkeypatch):
    """Search for 'transformer attention' and get at least 1 result."""
    import app.config as cfg
    import app.db as db_mod
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    await db_mod.init_db()

    from app.arxiv_svc import search
    papers = await search("transformer attention mechanism", max_results=3)
    assert len(papers) > 0
    for p in papers:
        assert p["arxiv_id"]
        assert p["title"]
        assert p["abstract"]


@pytest.mark.asyncio
async def test_fetch_metadata_uses_cache(tmp_path, monkeypatch):
    """Second call should hit SQLite cache, not arXiv API."""
    import app.config as cfg
    import app.db as db_mod
    import httpx
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(cfg, "DB_PATH", db_path)
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    await db_mod.init_db()

    # Prime the cache
    await db_mod.cache_metadata({
        "arxiv_id": "1706.03762",
        "title": "Cached Title",
        "abstract": "Cached abstract",
        "authors": "[]",
        "category": "cs.CL",
        "published": "2017-06-12",
    })

    # Patch httpx to raise if called (should not be needed)
    original_get = httpx.AsyncClient.get
    call_count = {"n": 0}
    async def patched_get(self, *a, **kw):
        call_count["n"] += 1
        return await original_get(self, *a, **kw)
    monkeypatch.setattr(httpx.AsyncClient, "get", patched_get)

    from app.arxiv_svc import fetch_metadata
    paper = await fetch_metadata("1706.03762")
    assert paper["title"] == "Cached Title"
    assert call_count["n"] == 0   # no HTTP call made


def test_distinct_old_style_ids_do_not_collide():
    """Three different papers must not collapse onto one key.

    This is the same class of defect the codebase already documents for Qdrant
    point ids: a lookup that silently returns another paper's data.
    """
    ids = ["math/0309136v1", "math/0511124v2", "hep-ph/0512038v1"]
    normalised = [_normalise_id(f"http://arxiv.org/abs/{i}") for i in ids]
    assert len(set(normalised)) == 3, f"ids collided: {normalised}"


def test_normalise_id_always_returns_a_string():
    """CLAUDE.md §3.9 — arXiv ids are strings, never coerced."""
    for raw in ("0704.0001", "math/0309136v1", "1706.03762v5"):
        assert isinstance(_normalise_id(raw), str)
    # Leading zeros survive; pandas-style coercion would eat them.
    assert _normalise_id("0704.0001") == "0704.0001"
