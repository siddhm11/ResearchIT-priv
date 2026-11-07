"""The process-wide paper caches must hand every caller its own dicts.

Search and collections write per-reader fields (saved, query_id, position)
onto the papers they render. When the metadata cache handed out its stored
objects, those fields leaked to the next reader of the same paper: the 4 Nov
audit (F1) saw 52 of 150 concurrent readers shown another reader's save.
"""
import pytest

from app import turso_svc


@pytest.fixture(autouse=True)
def _empty_caches():
    turso_svc._METADATA_CACHE.clear()
    turso_svc._TRENDING_CACHE.clear()
    yield
    turso_svc._METADATA_CACHE.clear()
    turso_svc._TRENDING_CACHE.clear()


PAPER = {"arxiv_id": "2609.31507", "title": "SatNav", "abstract": "a"}


async def test_a_fetched_paper_changed_by_one_reader_is_clean_for_the_next():
    turso_svc._cache_put(PAPER["arxiv_id"], dict(PAPER))

    first = await turso_svc.fetch_metadata_batch([PAPER["arxiv_id"]])
    first[PAPER["arxiv_id"]]["saved"] = True
    first[PAPER["arxiv_id"]]["query_id"] = "reader-A-qid"

    second = await turso_svc.fetch_metadata_batch([PAPER["arxiv_id"]])
    assert "saved" not in second[PAPER["arxiv_id"]]
    assert "query_id" not in second[PAPER["arxiv_id"]]


async def test_a_paper_changed_after_it_was_cached_does_not_change_the_cache():
    paper = dict(PAPER)
    turso_svc._cache_put(paper["arxiv_id"], paper)
    paper["saved"] = True

    assert "saved" not in turso_svc._cache_get(paper["arxiv_id"])


async def test_cached_starter_lists_are_copies(monkeypatch):
    from app import local_meta
    monkeypatch.setattr(local_meta, "is_available", lambda: True)
    monkeypatch.setattr(local_meta, "fetch_trending",
                        lambda codes, limit, months: [dict(PAPER, categories="cs.CV")])

    first = await turso_svc.fetch_trending_by_categories({"cs.CV"}, limit=1)
    first[0]["saved"] = True
    second = await turso_svc.fetch_trending_by_categories({"cs.CV"}, limit=1)

    assert "saved" not in second[0]
