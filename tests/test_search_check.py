"""The nightly search check reads ranks the way a reader sees them (audit F24)."""
import json
from pathlib import Path

from scripts.search_check import QUERIES, ranked_ids, score


def test_ranks_come_from_the_page_in_order():
    html = ('<article id="paper-2401.00001"></article><div id="paper-hep-th/9901001"></div>'
            '<a href="#paper-2401.00001"></a><article id="paper-2401.00001"></article>')
    assert ranked_ids(html) == ["2401.00001", "hep-th/9901001"]


def test_score_is_hit_at_10_and_reciprocal_rank():
    ids = [f"p{i}" for i in range(12)]
    assert score(ids, ["p0"]) == (True, 1.0)
    assert score(ids, ["p3"]) == (True, 0.25)
    assert score(ids, ["p11"]) == (False, 0.0)          # rank 12 is past the first page


def test_the_fixed_queries_are_100_known_answers():
    qs = json.loads(Path(QUERIES).read_text())
    assert len(qs) == 100 and len({q["qid"] for q in qs}) == 100
    assert {q["kind"] for q in qs} == {"keyword", "question"}
    assert all(q["gold"] and all(isinstance(g, str) for g in q["gold"]) for q in qs)
