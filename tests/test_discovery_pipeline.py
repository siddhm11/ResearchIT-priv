"""Real SQLite + real ranking components; remote sources/encoding are controlled."""
from collections import Counter
from datetime import datetime, timezone
import json
import sqlite3

import httpx
import numpy as np
import pytest

from app.discovery_store import DiscoveryStore, EMBEDDING_CONTRACT, validated_vector
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


def test_store_refuses_the_user_database(tmp_path):
    path=tmp_path/'user.db'
    with sqlite3.connect(path) as c:c.execute('create table interactions(id integer)')
    with pytest.raises(ValueError):DiscoveryStore(path)
    with sqlite3.connect(path) as c:
        assert not c.execute("select name from sqlite_master where name='candidates'").fetchone()
