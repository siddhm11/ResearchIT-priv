"""The year shown on a paper is its publication year, not its last revision."""
from app.turso_svc import _to_paper_dict


def _row(arxiv_id, update_date):
    return {"arxiv_id": arxiv_id, "title": "T", "authors": "A", "categories": "cs.CL",
            "update_date": update_date, "abstract_preview": "x", "citation_count": 1}


def test_year_comes_from_the_arxiv_id():
    assert _to_paper_dict(_row("1706.03762", "2023-08-03"))["year"] == 2017
    assert _to_paper_dict(_row("math/0309136", "2010-01-01"))["year"] == 2003
    assert _to_paper_dict(_row("2512.10942", "2026-02-01"))["year"] == 2025


def test_unparseable_id_falls_back_to_update_date():
    assert _to_paper_dict(_row("weird-id", "2019-05-01"))["year"] == 2019
    # The revision date itself is still there for anything that needs it.
    assert _to_paper_dict(_row("1706.03762", "2023-08-03"))["published"] == "2023-08-03"
