"""OAI-PMH harvesting in scripts/ingest_arxiv.py."""
import io
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ingest_arxiv as ia  # noqa: E402

PAGE = """<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>
<record><header><identifier>oai:arXiv.org:2609.02991</identifier><datestamp>2026-09-29</datestamp></header>
<metadata><arXiv xmlns="http://arxiv.org/OAI/arXiv/"><id>2609.02991</id><created>2026-09-02</created>
<authors><author><keyname>Robert</keyname><forenames>Christian P.</forenames></author>
<author><keyname>Smith</keyname><forenames>Ann</forenames><suffix>Jr</suffix></author></authors>
<title>Uncertainty sets
   for robust learning</title><categories>cs.LG cs.NA math.NA stat.ML</categories>
<abstract>  We study $ϕ$-divergence uncertainty sets and   function approximation in depth.  </abstract>
</arXiv></metadata></record>
<record><header status="deleted"><identifier>oai:arXiv.org:2609.00001</identifier><datestamp>2026-09-29</datestamp></header></record>
<resumptionToken completeListSize="2" cursor="0"></resumptionToken>
</ListRecords></OAI-PMH>"""


def test_record_is_normalised_like_the_search_api():
    root = ET.fromstring(PAGE)
    recs = list(root.iter("{http://www.openarchives.org/OAI/2.0/}record"))
    p = ia.parse_oai_record(recs[0])
    assert p["arxiv_id"] == "2609.02991"
    assert p["title"] == "Uncertainty sets for robust learning"
    assert "$\u03c6$-divergence" in p["abstract"] and "\u03d5" not in p["abstract"]   # NFKC
    assert p["abstract"].startswith("We study") and "  " not in p["abstract"]
    assert p["categories"] == "cs.LG math.NA stat.ML"       # cs.NA alias dropped, order kept
    assert p["primary_topic"] == "cs.LG"
    assert p["authors"] == "Christian P. Robert, Ann Smith Jr"
    assert p["update_date"] == "2026-09-02"
    assert ia.parse_oai_record(recs[1]) is None             # deleted


def test_alias_collapsing_keeps_the_canonical_position():
    assert ia.canonical_categories("eess.AS cs.AI cs.SD cs.SY eess.SP eess.SY") == "eess.AS cs.AI cs.SD eess.SP eess.SY"
    assert ia.canonical_categories("cs.IT cs.DC math.IT") == "cs.IT cs.DC"
    assert ia.canonical_categories("math.IT cs.LG") == "cs.IT cs.LG"      # alias alone maps in place


def test_sets_cover_every_configured_category():
    sets = {ia.oai_set(c) for c in ia.DEFAULT_CATEGORIES}
    assert sets == {"cs", "stat", "math", "q-bio", "econ", "eess", "physics"}


def _paper(aid, cats):
    return {"arxiv_id": aid, "title": "t" * 6, "authors": "a", "abstract": "x" * 30,
            "categories": cats, "primary_topic": cats.split()[0], "update_date": "2026-09-29"}


def test_harvest_filters_pages_through_and_marks_complete_sets():
    pages = {
        ("cs", None): ([_paper("2609.00010", "cs.CL"), _paper("2609.00011", "cs.OS"),     # cs.OS not configured
                        _paper("1810.04449", "cs.LG"), _paper("2512.00001", "cs.LG")],     # old ids
                       "tok1", 5, 4),
        ("cs", "tok1"): ([_paper("2609.00012", "cs.CV")], None, 5, 1),
        ("stat", None): ([_paper("2609.00020", "stat.ML math.ST")], None, 1, 1),
    }
    fake = lambda s, since, until, tok: pages[(s, tok)]
    handled, state = [], {}
    seen, new = ia.harvest(["cs.CL", "cs.CV", "stat.ML"], "2026-09-28", "2026-09-30", "2608", state,
                           lambda ps: handled.extend(p["arxiv_id"] for p in ps) or len(ps),
                           lambda: None, fetch=fake, delay=0, log=lambda m: None)
    assert handled == ["2609.00010", "2609.00012", "2609.00020"] and seen == new == 3
    assert state == {"oai:cs": "done", "oai:stat": "done"}
    assert ia.report_oai_coverage(["cs.CL", "stat.ML"], state) == 0


def test_short_or_failed_sets_fail_coverage():
    pages = {("cs", None): ([], None, 9, 4),          # arXiv promised 9 records, 4 arrived
             ("stat", None): ([], None, None, None)}  # request failed
    state = {}
    ia.harvest(["cs.CL", "stat.ML"], "2026-09-28", "2026-09-30", "2608", state, lambda ps: 0,
               lambda: None, fetch=lambda s, a, b, t: pages[(s, t)], delay=0, log=lambda m: None)
    assert state == {"oai:cs": "short 4/9", "oai:stat": "failed"}
    assert ia.report_oai_coverage(["cs.CL", "stat.ML"], state) == 3


def test_a_day_with_no_records_is_complete(monkeypatch):
    body = (b'<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
            b'<error code="noRecordsMatch">no records</error></OAI-PMH>')
    monkeypatch.setattr(ia.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(body))
    assert ia.fetch_oai_page("econ", "2026-09-28", "2026-09-30", None) == ([], None, 0, 0)
    state = {}
    ia.harvest(["econ.TH"], "2026-09-28", "2026-09-30", "2608", state, lambda ps: 0, lambda: None,
               delay=0, log=lambda m: None)
    assert state == {"oai:econ": "done"}


def test_parsing_the_sample_page_via_fetch(monkeypatch):
    monkeypatch.setattr(ia.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(PAGE.encode()))
    papers, token, size, records = ia.fetch_oai_page("cs", "2026-09-28", "2026-09-30", None)
    assert [p["arxiv_id"] for p in papers] == ["2609.02991"] and token is None and size == 2 and records == 2


def test_requests_identify_themselves_and_bad_requests_are_not_retried(monkeypatch):
    import urllib.error
    seen, waits = [], []

    def urlopen(req, timeout):
        seen.append(req.get_header("User-agent"))
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, None)

    monkeypatch.setattr(ia.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(ia.time, "sleep", waits.append)
    assert ia.fetch_oai_page("cs", "2026-09-28", "2026-09-30", None) == ([], None, None, None)
    assert seen == [ia.USER_AGENT] and waits == []          # one attempt, no back-off
    assert "Python-urllib" not in ia.USER_AGENT
    assert ia._permanent(urllib.error.HTTPError("u", 404, "x", {}, None))
    assert not ia._permanent(urllib.error.HTTPError("u", 429, "x", {}, None))
    assert not ia._permanent(urllib.error.HTTPError("u", 406, "x", {}, None))   # was a temporary block
    assert ia.retry_wait(urllib.error.HTTPError("u", 406, "x", {}, None), 0) == 30
    assert not ia._permanent(TimeoutError("x"))
