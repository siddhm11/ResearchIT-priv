"""
Age features measure time since FIRST publication, not since the last revision.

The sidecar/Turso `published` field is `update_date`, the date of a paper's
latest arXiv version. Ageing from it made "Attention Is All You Need" (2017,
revised 2023-08-03) look like a 2023 paper and earn a fresh-paper boost. The
arXiv ID records the month of first submission, so that is the age source;
the metadata date is only a fallback for IDs that do not parse.
"""
from datetime import datetime, timezone

import numpy as np

from app.recommend.reranker import _publication_date, compute_features


def _age_days(date_str: str) -> int:
    d = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - d).days


def test_new_style_id_wins_over_a_later_revision_date():
    meta = {"arxiv_id": "1706.03762", "published": "2023-08-03"}
    assert _publication_date(meta) == "2017-06-15"


def test_five_digit_and_old_style_ids_parse():
    assert _publication_date({"arxiv_id": "2407.21783"}) == "2024-07-15"
    assert _publication_date({"arxiv_id": "math/0309136"}) == "2003-09-15"
    assert _publication_date({"arxiv_id": "hep-th/9901001"}) == "1999-01-15"


def test_unparseable_id_falls_back_to_the_metadata_date():
    assert _publication_date({"arxiv_id": "paper_3", "published": "2025-02-01"}) == "2025-02-01"
    assert _publication_date({"arxiv_id": "", "update_date": "2024-05-05"}) == "2024-05-05"
    assert _publication_date({}) == ""


def test_revised_classic_is_aged_from_2017_in_every_age_feature():
    embs = np.eye(2, 1024, dtype=np.float32)
    meta = [{"arxiv_id": "1706.03762", "published": "2023-08-03"},
            {"arxiv_id": "paper_x", "published": "2023-08-03"}]
    f = compute_features(embs, meta)
    age = _age_days("2017-06-15")
    assert f[0, 5] == age                                  # candidate_age_days
    assert np.isclose(f[0, 6], np.exp(-0.002 * age))       # recency
    assert f[0, 9] == datetime.now(timezone.utc).year - 2017   # year_diff
    # The unparseable id keeps using its metadata date, as before.
    assert f[1, 5] == _age_days("2023-08-03")
    assert f[0, 6] < f[1, 6]


def test_genuinely_new_paper_outranks_a_revised_classic_on_recency():
    embs = np.eye(2, 1024, dtype=np.float32)
    meta = [{"arxiv_id": "1706.03762", "published": "2026-09-30"},   # revised last week
            {"arxiv_id": "2609.01234", "published": "2026-09-02"}]   # first posted Sep 2026
    f = compute_features(embs, meta)
    assert f[1, 6] > f[0, 6]
