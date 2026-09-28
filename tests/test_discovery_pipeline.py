"""Real SQLite + real ranking components; remote sources/encoding are controlled."""
from collections import Counter
from datetime import datetime, timezone
import json
import sqlite3

import httpx
import numpy as np
import pytest

from app.discovery_store import DiscoveryStore, EMBEDDING_CONTRACT, validated_vector
from app.discovery_worker import collect_once, prepare_once
from app.discovery_shadow import compare_feed
from app.hf_papers_svc import normalize_snapshot

NOW=datetime(2026,9,25,tzinfo=timezone.utc).timestamp()

def vec(topic=0):
    v=np.zeros(1024);v[topic]=1
    return v.tolist()


def raw(aid='2609.00001', **kw):
    return {'paper':{'id':aid,'title':'Model evaluation','summary':'A full research abstract.',
                     'publishedAt':'2026-09-20T00:00:00Z','upvotes':0,**kw}}


def snapshot(rows=None,now=NOW):
    return normalize_snapshot(rows or [raw()],datetime.fromtimestamp(now,timezone.utc))


@pytest.fixture
def store(tmp_path):return DiscoveryStore(tmp_path/'source.db')


def test_persist_restart_idempotence_and_zero_votes(store):
    s=snapshot();store.save_snapshot(s,NOW);store.save_snapshot(s,NOW)
    reopened=DiscoveryStore(store.path)
    assert reopened.status(NOW)['candidates']==1
    with reopened.connection() as c:assert c.execute('select count(*) from snapshots').fetchone()[0]==1
    assert json.loads(reopened.pending(NOW)[0]['paper'])['upvotes']==0


def test_readiness_invalidation_on_changed_content(store):
    store.save_snapshot(snapshot(),NOW)
    r=store.pending(NOW)[0]
    assert store.finish(r,json.loads(r['paper']),vec(),NOW,EMBEDDING_CONTRACT)
    assert len(store.ready(NOW))==1
    store.save_snapshot(snapshot([raw(summary='Changed research abstract.')],NOW+60),NOW+60)
    assert not store.ready(NOW+60)
    assert len(store.pending(NOW+60))==1


def test_new_observation_with_same_content_keeps_vector(store):
    store.save_snapshot(snapshot(),NOW);r=store.pending(NOW)[0]
    store.finish(r,json.loads(r['paper']),vec(),NOW,EMBEDDING_CONTRACT)
    store.save_snapshot(snapshot([raw(upvotes=10)],NOW+60),NOW+60)
    assert len(store.ready(NOW+60))==1


def test_late_encoder_cannot_clobber_new_observation(store):
    store.save_snapshot(snapshot(),NOW);r=store.pending(NOW)[0]
    store.save_snapshot(snapshot([raw(title='Updated')],NOW+60),NOW+60)
    assert not store.finish(r,json.loads(r['paper']),vec(),NOW+60,EMBEDDING_CONTRACT)
    assert not store.ready(NOW+60)


def test_delayed_snapshot_does_not_roll_back_latest_metadata(store):
    store.save_snapshot(snapshot([raw(title='Latest')],NOW+60),NOW+60)
    store.save_snapshot(snapshot(),NOW+60)
    assert json.loads(store.pending(NOW+60)[0]['paper'])['title']=='Latest'


def test_stale_and_future_sources_ineligible(store):
    store.save_snapshot(snapshot(),NOW);r=store.pending(NOW)[0]
    store.finish(r,json.loads(r['paper']),vec(),NOW,EMBEDDING_CONTRACT)
    assert not store.ready(NOW+49*3600)
    assert store.status(NOW+49*3600)['source_stale']
    with pytest.raises(ValueError):store.save_snapshot(snapshot(now=NOW+60),NOW)
    assert not store.ready(NOW-60)

@pytest.mark.parametrize('bad',[[],[1]*1024,[0]*1024,[float('nan')]*1024])
def test_incompatible_vector_cannot_become_ready(store,bad):
    store.save_snapshot(snapshot(),NOW);r=store.pending(NOW)[0]
    with pytest.raises(ValueError):store.finish(r,json.loads(r['paper']),bad,NOW,EMBEDDING_CONTRACT)
    assert not store.ready(NOW)


def test_contract_rejected(store):
    store.save_snapshot(snapshot(),NOW);r=store.pending(NOW)[0]
    with pytest.raises(ValueError):store.finish(r,json.loads(r['paper']),vec(),NOW,'other-model')


async def test_collect_failure_preserves_previous_good_snapshot(store):
    store.save_snapshot(snapshot(),NOW)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _:httpx.Response(429))) as c:
        result=await collect_once(store,c,NOW+60)
    assert result['status']=='failed'
    assert store.status(NOW+60)['candidates']==1
    assert store.status(NOW+60)['last_success_age_hours']==pytest.approx(1/60)
    assert store.status(NOW+60)['last_run']['status']=='failed'


async def test_schema_drift_is_failure_not_empty_success(store):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _:httpx.Response(200,json=[{'unknown':1}]))) as c:
        assert (await collect_once(store,c,NOW))['status']=='failed'
    assert store.status(NOW)['source_stale']


async def test_zero_paper_response_is_valid_but_not_ready(store):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _:httpx.Response(200,json=[]))) as c:
        assert (await collect_once(store,c,NOW))['status']=='ok'
    assert store.status(NOW)['ready']==0


async def test_three_attempt_cap_and_backoff(store):
    store.save_snapshot(snapshot(),NOW)
    def broken(_):raise RuntimeError('private details never saved')
    async with httpx.AsyncClient() as c:
        for t in [NOW,NOW+300,NOW+900]:
            assert (await prepare_once(store,c,now=t,encoder=broken))['failed']==1
            assert not store.pending(t+1)
        assert not store.pending(NOW+10000)
    assert store.status(NOW)['retry_exhausted']==1
    with store.connection() as c:
        assert c.execute('select error from candidates').fetchone()[0]=='RuntimeError'


async def test_missing_abstract_enriched_then_encoded(store):
    store.save_snapshot(snapshot([raw(summary='')]),NOW)
    async def fetcher(client,paper):return {**paper,'abstract':'Recovered from arXiv'}
    seen=[]
    def encode(paper):seen.append(paper['abstract']);return vec()
    async with httpx.AsyncClient() as c:
        result=await prepare_once(store,c,now=NOW,encoder=encode,metadata_fetcher=fetcher)
    assert result['ready']==1 and seen==['Recovered from arXiv']
    assert store.ready(NOW)[0]['abstract']=='Recovered from arXiv'


async def test_one_bad_paper_does_not_block_other_preparation(store):
    store.save_snapshot(snapshot([raw(),raw('2609.00002')]),NOW)
    def encode(p):
        if p['arxiv_id'].endswith('1'):raise ValueError()
        return vec()
    async with httpx.AsyncClient() as c:r=await prepare_once(store,c,now=NOW,encoder=encode)
    assert r=={'ready':1,'failed':1,'superseded':0}


def baseline():
    return {'name':'Two-interest test reader','embedding_contract':EMBEDDING_CONTRACT,
            'seed_ids':[f'seed{i}' for i in range(10)],
            'seed_vectors':[vec(i%2) for i in range(10)],
            'papers':[{'arxiv_id':f'2501.{i:05d}','title':f'Baseline paper {i}',
                       'vector':vec(i%2),'published_at':'2025-01-01T00:00:00Z'} for i in range(10)]}


def fresh(topic=0,aid='2609.00001'):
    return {'arxiv_id':aid,'title':'Fresh paper','vector':vec(topic),
            'embedding_contract':EMBEDDING_CONTRACT,'last_seen':NOW,
            'published_at':'2026-09-20T00:00:00Z','upvotes':0}


def test_source_cap_and_minor_interest_survive():
    b=baseline();hf=[fresh(0,f'2609.{i:05d}') for i in range(20)]
    r=compare_feed(b,hf,NOW)
    assert len(r['combined'])==10 and len(r['changes'])==3
    assert Counter(p['interest'] for p in r['combined'])==Counter(p['interest'] for p in r['baseline'])
    assert len({p['arxiv_id'] for p in r['combined']})==10
    assert all('vector' not in p for p in r['combined'])


def test_popularity_cannot_promote_orthogonal_topic():
    r=compare_feed(baseline(),[{**fresh(2),'upvotes':1000000}],NOW)
    assert not r['changes'] and len(r['rejected'])==1


@pytest.mark.parametrize('change',[{'last_seen':NOW-49*3600},{'last_seen':NOW+60},
                                 {'embedding_contract':'other-model'},{'vector':vec(0)[:10]},
                                 {'published_at':'2027-01-01T00:00:00Z'}])
def test_invalid_candidates_leave_baseline_unchanged(change):
    r=compare_feed(baseline(),[{**fresh(),**change}],NOW)
    assert r['combined']==r['baseline'] and not r['hf_only']


def test_exclusions_and_duplicates_applied_before_selection():
    b=baseline();b['excluded_ids']=['2609.00001','2501.00000']
    r=compare_feed(b,[fresh(),fresh(aid='2501.00001')],NOW)
    assert not r['changes'] and len(r['baseline'])==9
    assert all(p['arxiv_id'] not in b['excluded_ids'] for p in r['combined'])


def test_no_source_is_identity_and_cap_zero_disables_experiment():
    b=baseline()
    assert compare_feed(b,[],NOW)['combined']==compare_feed(b,[],NOW)['baseline']
    assert not compare_feed(b,[fresh()],NOW,hf_cap=0)['changes']


def test_shadow_rejects_unknown_baseline_encoding():
    b=baseline();b['embedding_contract']='wrong'
    with pytest.raises(ValueError):compare_feed(b,[fresh()],NOW)


async def test_zero_vote_end_to_end_collection_to_comparison(store):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _:httpx.Response(200,json=[raw()]))) as c:
        assert (await collect_once(store,c,NOW))['status']=='ok'
        assert (await prepare_once(store,c,now=NOW,encoder=lambda _:vec()))['ready']==1
    reopened=DiscoveryStore(store.path)
    r=compare_feed(baseline(),reopened.ready(NOW),NOW)
    assert len(r['changes'])==1 and r['combined']!=r['baseline']
    # This store never creates or writes user-interaction tables.
    with store.connection() as c:
        assert not c.execute("select name from sqlite_master where name='interactions'").fetchone()


def test_recent_source_mention_does_not_make_old_paper_new():
    r=compare_feed(baseline(),[{**fresh(),'published_at':'2020-01-01T00:00:00Z'}],NOW)
    assert not r['changes']


def test_store_refuses_the_user_database(tmp_path):
    path=tmp_path/'user.db'
    with sqlite3.connect(path) as c:c.execute('create table interactions(id integer)')
    with pytest.raises(ValueError):DiscoveryStore(path)
    with sqlite3.connect(path) as c:
        assert not c.execute("select name from sqlite_master where name='candidates'").fetchone()


def test_shadow_recency_uses_as_of_time():
    b=baseline()
    first=compare_feed(b,[],NOW)
    later=compare_feed(b,[],NOW+30*86400)
    # Same fixed geometry and order, older publication ages must lower scores.
    assert all(a['score']>z['score'] for a,z in zip(first['baseline'],later['baseline']))
