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


def _record_with_licence(licence: str | None) -> ET.Element:
    tag = f"<license>{licence}</license>" if licence is not None else ""
    return ET.fromstring(
        '<record xmlns="http://www.openarchives.org/OAI/2.0/"><header><identifier>oai:arXiv.org:2610.00042'
        '</identifier></header><metadata><arXiv xmlns="http://arxiv.org/OAI/arXiv/"><id>2610.00042</id>'
        '<created>2026-10-19</created><authors><author><keyname>Lee</keyname></author></authors>'
        '<title>Figures with licences</title><categories>cs.CV</categories>'
        f'<abstract>An abstract long enough to keep the record.</abstract>{tag}</arXiv></metadata></record>')


def test_the_licence_is_read_from_the_record():
    cc = ia.parse_oai_record(_record_with_licence("http://creativecommons.org/licenses/by/4.0/"))
    assert cc["licence"] == "CC-BY-4.0" and cc["licence_url"] == "http://creativecommons.org/licenses/by/4.0/"
    default = ia.parse_oai_record(_record_with_licence(" http://arxiv.org/licenses/nonexclusive-distrib/1.0/ "))
    assert default["licence"] == "ARXIV-NONEXCLUSIVE-1.0"
    odd = ia.parse_oai_record(_record_with_licence("https://example.org/my-licence"))
    assert odd["licence"] == "" and odd["licence_url"] == "https://example.org/my-licence"   # kept to remap later
    none = ia.parse_oai_record(list(ET.fromstring(PAGE).iter("{http://www.openarchives.org/OAI/2.0/}record"))[0])
    assert none["licence"] == "" and none["licence_url"] == ""


def test_progress_lines_carry_their_utc_time(capsys):
    # The daily job prints this script's output only at the end; each line's
    # own time is what shows where a timed-out run spent its minutes.
    import re
    ia.stamp("  oai cs       records=1,300")
    assert re.fullmatch(r"\d\d:\d\d:\d\d   oai cs       records=1,300\n", capsys.readouterr().out)


# ── arXiv not answering (5 and 6 Nov 2026: most requests timed out for hours) ──

class _Clock:
    """time.monotonic and time.sleep for a harvest: sleeping moves the clock, and
    a fetch that fails costs its own time, as a 180 s timeout does."""

    def __init__(self, monkeypatch):
        self.now, self.sleeps = 0.0, []
        monkeypatch.setattr(ia.time, "monotonic", lambda: self.now)
        monkeypatch.setattr(ia.time, "sleep", self._sleep)

    def _sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def _run(cats, fetch, state, stop_at=None):
    logs = []
    out = ia.harvest(cats, "2026-11-03", "2026-11-06", "2610", state,
                     lambda ps: len(ps), lambda: None, fetch=fetch, delay=0, log=logs.append,
                     stop_at=stop_at)
    return out, logs


def test_a_set_that_gets_no_answer_does_not_hold_up_the_others(monkeypatch):
    """6 Nov: cs spent about 18 minutes failing six tries before econ, which answered, got a turn."""
    clock = _Clock(monkeypatch)
    calls = []

    def fetch(s, since, until, tok):
        calls.append(s)
        if s == "cs" and calls.count("cs") < 3:
            clock.now += 180
            raise TimeoutError("The read operation timed out")
        return [_paper("2611.00001", "cs.CL" if s == "cs" else "econ.TH")], None, 1, 1

    state = {}
    (seen, new), _ = _run(["cs.CL", "econ.TH"], fetch, state, stop_at=10_000)
    assert calls == ["cs", "econ", "cs", "cs"]           # econ went straight after the first failure
    assert state == {"oai:cs": "done", "oai:econ": "done", "oai_failed_tries": 2}
    assert seen == new == 2


def test_a_list_carries_on_from_its_page_after_a_failure(monkeypatch):
    clock = _Clock(monkeypatch)
    requested = []
    answers = {None: ([_paper("2611.00001", "cs.CL")], "tok1", 2, 1),
               "tok1": ([_paper("2611.00002", "cs.CL")], None, 2, 1)}
    failed = []

    def fetch(s, since, until, tok):
        requested.append(tok)
        if tok == "tok1" and not failed:
            failed.append(1)
            raise ConnectionResetError("reset")
        return answers[tok]

    state = {}
    _run(["cs.CL"], fetch, state, stop_at=10_000)
    assert requested == [None, "tok1", "tok1"]           # page 1 is not fetched again
    assert state["oai:cs"] == "done"


def test_waits_grow_while_nothing_answers_and_reset_when_something_does(monkeypatch):
    clock = _Clock(monkeypatch)
    results = iter([TimeoutError("t"), TimeoutError("t"), TimeoutError("t"),
                    ([], None, 0, 0), TimeoutError("t")])

    def fetch(s, since, until, tok):
        r = next(results, ([], None, 0, 0))
        if isinstance(r, Exception):
            raise r
        return r

    _run(["cs.CL", "econ.TH"], fetch, {}, stop_at=100_000)
    assert clock.sleeps == [30, 60, 120, 30]             # not 5, 10, 15 s: that hammers a struggling server


def test_a_retry_after_is_honoured_but_capped():
    def http(code, after=None):
        return ia.urllib.error.HTTPError("u", code, "x", {"Retry-After": after} if after else {}, None)

    assert ia.oai_wait(http(503, "45"), 3) == 45
    assert ia.oai_wait(http(503, "9000"), 0) == 300
    assert ia.oai_wait(http(503), 1) == 60
    assert ia.oai_wait(TimeoutError("t"), 0) == 30
    assert ia.oai_wait(TimeoutError("t"), 20) == 300


def test_the_time_limit_stops_cleanly_and_reports_what_is_unfinished(monkeypatch):
    clock = _Clock(monkeypatch)

    def fetch(s, since, until, tok):
        if s == "econ":
            return [_paper("2611.00001", "econ.TH")], None, 1, 1
        if tok is None:
            return [_paper("2611.00002", "cs.CL")], "tok1", 9, 4
        clock.now += 180
        raise TimeoutError("The read operation timed out")

    state, saves = {}, []
    logs = []
    seen, new = ia.harvest(["cs.CL", "econ.TH"], "2026-11-03", "2026-11-06", "2610", state,
                           lambda ps: len(ps), lambda: saves.append(dict(state)),
                           fetch=fetch, delay=0, log=logs.append, stop_at=1_000)
    assert clock.now <= 1_000 + 180                      # a try in flight may finish; none starts after
    assert state["oai:cs"] == "unfinished 4/9" and state["oai:econ"] == "done"
    assert seen == new == 2                              # what was stored still counts
    assert saves[-1]["oai:cs"] == "unfinished 4/9"       # saved, so the job can report it
    assert ia.report_oai_coverage(["cs.CL", "econ.TH"], state) == 3


def test_a_wait_never_runs_past_the_time_limit(monkeypatch):
    clock = _Clock(monkeypatch)

    def fetch(s, since, until, tok):
        raise TimeoutError("t")

    _run(["cs.CL"], fetch, {}, stop_at=40)
    assert clock.now == 40 and clock.sleeps == [30, 10]


def test_every_failed_try_is_logged_with_how_long_it_took(monkeypatch):
    """The old log kept only each set's last error, which is why nobody could tell
    afterwards how long arXiv took to fail."""
    clock = _Clock(monkeypatch)

    def fetch(s, since, until, tok):
        clock.now += 180
        raise TimeoutError("The read operation timed out")

    _, logs = _run(["cs.CL"], fetch, {}, stop_at=200)
    assert any("cs" in m and "180 s" in m and "timed out" in m for m in logs)


def test_a_permanent_failure_ends_that_set_without_waiting(monkeypatch):
    clock = _Clock(monkeypatch)
    state = {}
    _run(["cs.CL", "econ.TH"],
         lambda s, a, b, t: ([], None, None, None) if s == "cs" else ([], None, 0, 0), state, stop_at=1_000)
    assert state == {"oai:cs": "failed", "oai:econ": "done"} and clock.sleeps == []


def test_an_expired_token_restarts_that_set(monkeypatch):
    """arXiv's tokens expire at the next UTC midnight, so a long wait can outlive one."""
    _Clock(monkeypatch)
    requested = []

    def fetch(s, since, until, tok):
        requested.append(tok)
        if tok is None:
            return [_paper("2611.00001", "cs.CL")], ("old" if requested.count(None) == 1 else None), 1, 1
        raise ia.ResumptionExpired("badResumptionToken")

    state = {}
    _run(["cs.CL"], fetch, state, stop_at=1_000)
    assert requested == [None, "old", None] and state["oai:cs"] == "done"


def _urlopen_returning(body: bytes):
    return lambda req, timeout: io.BytesIO(body)


def test_fetch_treats_a_garbled_reply_as_a_failed_try(monkeypatch):
    """Parsing sat outside the retry block, so an HTML error page crashed the whole ingest."""
    import pytest
    monkeypatch.setattr(ia.urllib.request, "urlopen", _urlopen_returning(b"<html>Service Unavailable"))
    with pytest.raises(ET.ParseError):
        ia.fetch_oai_page("cs", "2026-11-03", "2026-11-06", None)


def test_fetch_raises_what_is_worth_trying_again(monkeypatch):
    import urllib.error
    import pytest

    def busy(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 503, "first byte timeout", {}, None)

    monkeypatch.setattr(ia.urllib.request, "urlopen", busy)
    with pytest.raises(urllib.error.HTTPError):
        ia.fetch_oai_page("cs", "2026-11-03", "2026-11-06", None)


def test_fetch_reports_an_expired_token(monkeypatch):
    import pytest
    body = (b'<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
            b'<error code="badResumptionToken">expired</error></OAI-PMH>')
    monkeypatch.setattr(ia.urllib.request, "urlopen", _urlopen_returning(body))
    with pytest.raises(ia.ResumptionExpired):
        ia.fetch_oai_page("cs", "2026-11-03", "2026-11-06", "old")


def test_a_real_outage_end_to_end_through_the_real_fetch(monkeypatch):
    """Real sockets, real HTTP errors, real parsing: 503s, a garbled page and a dropped
    connection, then arXiv recovers and every set finishes."""
    import http.server
    import threading

    answers = iter(["503", "html", "drop", "ok", "ok"])

    class Fake(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            kind = next(answers, "ok")
            if kind == "drop":
                self.close_connection = True
                return
            if kind == "503":
                self.send_response(503, "first byte timeout")
                self.send_header("Retry-After", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"<html>Service Unavailable" if kind == "html" else PAGE.encode())

        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(ia, "OAI_API", f"http://127.0.0.1:{server.server_port}/oai")
        monkeypatch.setattr(ia, "oai_wait", lambda e, n: 0)
        state, stored = {}, []
        ia.harvest(["cs.LG", "stat.ML"], "2026-09-28", "2026-09-30", "2608", state,
                   lambda ps: stored.extend(ps) or len(ps), lambda: None, delay=0, log=lambda m: None,
                   stop_at=ia.time.monotonic() + 30)
    finally:
        server.shutdown()
    assert state == {"oai:cs": "done", "oai:stat": "done", "oai_failed_tries": 3}
    assert [p["arxiv_id"] for p in stored] == ["2609.02991", "2609.02991"]
