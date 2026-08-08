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
