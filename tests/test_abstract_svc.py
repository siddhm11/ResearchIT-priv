"""
app/abstract_svc.py: full abstracts for papers the sidecar cut at 500 characters.

The arXiv API is never called here. `_fetch` is replaced where the test is
about the queue, and the shared HTTP client gets a MockTransport where the test
is about the request itself. A fake clock makes the 3.1 s spacing checkable
without waiting for it.
"""
import asyncio

import httpx
import pytest

from app import abstract_svc, config, http_client

STUMP = ("We study the problem of learning models " * 20)[:499] + "z"   # 500 chars, ends mid-word
FULL = STUMP + " and finish the sentence properly, with results."


@pytest.fixture(autouse=True)
async def _clean_queue(monkeypatch):
    monkeypatch.setenv("FULL_ABSTRACTS", "on")
    yield
    await abstract_svc.stop()


@pytest.fixture
def clock(monkeypatch):
    t = {"now": 1000.0, "sleeps": []}
    monkeypatch.setattr(abstract_svc, "_now", lambda: t["now"])

    async def fake_sleep(seconds):
        t["sleeps"].append(seconds)
        t["now"] += seconds
        await asyncio.sleep(0)

    monkeypatch.setattr(abstract_svc, "_sleep", fake_sleep)
    return t


def test_the_stump_is_seen_as_cut():
    assert abstract_svc.is_truncated(STUMP)
    assert not abstract_svc.is_truncated(FULL[-60:])


async def test_overlay_swaps_only_cut_abstracts_and_leaves_its_input_alone():
    await abstract_svc.save("2609.00001", FULL)
    await abstract_svc.save("2609.00002", "short")
    cut = {"arxiv_id": "2609.00001", "abstract": STUMP}
    whole = {"arxiv_id": "2609.00002", "abstract": "A complete abstract."}
    out = await abstract_svc.overlay([cut, whole])
    assert out[0]["abstract"] == FULL and out[0]["abstract_source"] == "full"
    assert out[1] is whole                              # not cut, so untouched
    assert cut["abstract"] == STUMP                     # the cached dict is not edited


async def test_45_ids_take_three_requests_at_least_3_1_seconds_apart(monkeypatch, clock):
    calls = []

    async def fake_fetch(ids):
        calls.append((clock["now"], list(ids)))
        return {i: FULL for i in ids}

    monkeypatch.setattr(abstract_svc, "_fetch", fake_fetch)
    futs = [abstract_svc.request(f"2609.{i:05d}") for i in range(45)]
    await asyncio.gather(*futs)
    assert [len(ids) for _, ids in calls] == [20, 20, 5]
    gaps = [b[0] - a[0] for a, b in zip(calls, calls[1:])]
    assert all(g >= abstract_svc.MIN_INTERVAL_S for g in gaps)
    assert len(await abstract_svc.stored([f"2609.{i:05d}" for i in range(45)])) == 45


async def test_a_429_pauses_for_its_retry_after_then_retries(monkeypatch, clock):
    attempts = []

    async def fake_fetch(ids):
        attempts.append(list(ids))
        if len(attempts) == 1:
            raise abstract_svc._SoftBlock(30.0)
        return {i: FULL for i in ids}

    monkeypatch.setattr(abstract_svc, "_fetch", fake_fetch)
    assert await abstract_svc.request("2609.00001") == FULL
    assert 30.0 in clock["sleeps"]
    assert attempts == [["2609.00001"], ["2609.00001"]]


async def test_the_daily_cap_stops_fetching(monkeypatch, clock):
    monkeypatch.setattr(config, "FULL_ABSTRACTS_DAILY_CAP", 20)
    fetched = []

    async def fake_fetch(ids):
        fetched.extend(ids)
        return {i: FULL for i in ids}

    monkeypatch.setattr(abstract_svc, "_fetch", fake_fetch)
    results = await asyncio.gather(*[abstract_svc.request(f"2609.{i:05d}") for i in range(25)])
    assert len(fetched) == 20
    assert results.count(None) == 5


@pytest.mark.parametrize("bad", ["", "../etc/passwd", "2609.315071", "not-an-id", "2609.31507v2"])
def test_ids_that_are_not_arxiv_ids_are_never_queued(bad):
    assert not abstract_svc.valid_id(bad)


async def test_old_style_ids_are_accepted():
    assert abstract_svc.valid_id("hep-th/9901001")
    assert abstract_svc.valid_id("math.GT/0309136")


async def test_with_the_switch_off_nothing_is_fetched(monkeypatch):
    monkeypatch.setenv("FULL_ABSTRACTS", "off")
    assert abstract_svc.request("2609.00001") is None
    assert await abstract_svc.ensure("2609.00001") is None


async def test_concurrent_requests_for_one_paper_make_one_fetch(monkeypatch, clock):
    calls = []

    async def fake_fetch(ids):
        calls.append(list(ids))
        return {i: FULL for i in ids}

    monkeypatch.setattr(abstract_svc, "_fetch", fake_fetch)
    results = await asyncio.gather(*[abstract_svc.ensure("2609.00001") for _ in range(5)])
    assert results == [FULL] * 5
    assert calls == [["2609.00001"]]


async def test_a_timed_out_ensure_still_lets_the_row_land(monkeypatch):
    release = asyncio.Event()

    async def slow_fetch(ids):
        await release.wait()
        return {i: FULL for i in ids}

    monkeypatch.setattr(abstract_svc, "_fetch", slow_fetch)
    assert await abstract_svc.ensure("2609.00001", timeout=0.05) is None
    release.set()
    for _ in range(50):
        if await abstract_svc.stored(["2609.00001"]):
            break
        await asyncio.sleep(0.01)
    assert (await abstract_svc.stored(["2609.00001"]))["2609.00001"] == FULL


async def test_a_paper_arxiv_did_not_return_is_skipped_for_a_day(monkeypatch, clock):
    calls = []

    async def empty_fetch(ids):
        calls.append(list(ids))
        return {}

    monkeypatch.setattr(abstract_svc, "_fetch", empty_fetch)
    assert await abstract_svc.request("2609.00001") is None
    assert abstract_svc.request("2609.00001") is None
    assert len(calls) == 1


ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry><id>http://arxiv.org/abs/2609.00001v2</id><title>One</title>
    <summary>  A full abstract
      over two lines, ending at https://example.org/code</summary></entry>
  <entry><id>http://arxiv.org/abs/2609.00002v1</id><title>Two</title><summary>Another.</summary></entry>
</feed>"""


def _mock_client(monkeypatch, handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(http_client, "_client", client)
    monkeypatch.setattr(http_client, "_loop", asyncio.get_running_loop())
    return client


async def test_fetch_identifies_itself_and_parses_the_feed(monkeypatch):
    seen = {}

    def handler(request):
        seen["ua"] = request.headers.get("user-agent")
        seen["ids"] = request.url.params.get("id_list")
        return httpx.Response(200, text=ATOM)

    _mock_client(monkeypatch, handler)
    out = await abstract_svc._fetch(["2609.00001", "2609.00002"])
    assert seen["ua"] == config.OUTBOUND_USER_AGENT
    assert seen["ids"] == "2609.00001,2609.00002"
    assert out["2609.00001"] == "A full abstract over two lines, ending at https://example.org/code"
    assert out["2609.00002"] == "Another."


async def test_fetch_turns_a_429_into_a_soft_block_with_its_retry_after(monkeypatch):
    _mock_client(monkeypatch, lambda r: httpx.Response(429, headers={"Retry-After": "42"}))
    with pytest.raises(abstract_svc._SoftBlock) as exc:
        await abstract_svc._fetch(["2609.00001"])
    assert exc.value.retry_after == 42.0
