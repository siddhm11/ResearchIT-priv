from datetime import datetime, timezone
import httpx
import pytest
from app.hf_papers_svc import normalize_snapshot, fetch_snapshot

NOW=datetime(2026,9,25,tzinfo=timezone.utc)
def row(aid='2609.00001', **kw):
    return {'paper':{'id':aid,'title':'A research paper','publishedAt':'2026-09-20T00:00:00Z',**kw},
            'publishedAt':'2026-09-24T00:00:00Z'}

def test_ids_versions_dates_and_zero_votes_are_preserved():
    s=normalize_snapshot([row('0704.0001v2',upvotes=0),row('0704.0001v1'),row('hep-ph/0512038')],NOW)
    assert [p['arxiv_id'] for p in s['papers']]==['0704.0001','hep-ph/0512038']
    assert s['duplicates']==1 and s['papers'][0]['upvotes']==0
    assert s['papers'][0]['published_at'].startswith('2026-09-20')
    assert s['papers'][1]['upvotes'] is None and s['trend_velocity'] is None

@pytest.mark.parametrize('record',[{},None,row('../bad'),row(title=''),row(publishedAt='invalid'),row(publishedAt='2027-01-01T00:00:00Z')])
def test_malformed_and_future_records_rejected(record):
    s=normalize_snapshot([record],NOW)
    assert s['rejected']==1 and not s['papers']

@pytest.mark.parametrize('payload',[{},None,[{}]*1001])
def test_invalid_envelope_rejected(payload):
    with pytest.raises(ValueError):normalize_snapshot(payload,NOW)

async def test_fetch_reads_normalizes_without_auth_or_writes():
    def handle(req):
        assert req.method=='GET' and req.url.host=='huggingface.co'
        assert 'authorization' not in req.headers
        return httpx.Response(200,json=[row()])
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as c:
        assert len((await fetch_snapshot(c,now=NOW))['papers'])==1

@pytest.mark.parametrize('status',[429,500,503])
async def test_outage_is_not_a_successful_empty_feed(status):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _:httpx.Response(status))) as c:
        with pytest.raises(httpx.HTTPStatusError): await fetch_snapshot(c,now=NOW)

async def test_invalid_json_not_silently_accepted():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _:httpx.Response(200,text='bad'))) as c:
        with pytest.raises(ValueError): await fetch_snapshot(c,now=NOW)

async def test_oversize_response_bounded():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _:httpx.Response(200,content=b' '*4_000_001))) as c:
        with pytest.raises(ValueError): await fetch_snapshot(c,now=NOW)


def test_observed_web_sample_matches_adapter_schema():
    import json
    from pathlib import Path
    sample=json.loads((Path(__file__).resolve().parents[1]/'reports/recommendations/hf-observed-sample.json').read_text())
    result=normalize_snapshot(sample['records'],NOW)
    assert result['input_count']==2 and len(result['papers'])==2
    assert result['rejected']==0
