"""Lifecycle and failure isolation for the deployed shadow worker."""
import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from app import discovery_runtime as runtime
from app.discovery_worker import prepare_once
from tests.test_discovery_pipeline import NOW, snapshot
from app.discovery_store import DiscoveryStore


@pytest.mark.asyncio
async def test_disabled_has_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.setenv('HF_DISCOVERY_MODE', 'off')
    monkeypatch.setenv('HF_DISCOVERY_DB_PATH', str(tmp_path/'source.db'))
    worker = runtime.DiscoveryRuntime()
    await worker.start()
    assert worker.task is None
    assert worker.status() == {'scheduler':'disabled','serving_enabled':False}
    assert not (tmp_path/'source.db').exists()
    assert not (tmp_path/'source.worker.lock').exists()


@pytest.mark.asyncio
@pytest.mark.parametrize('key,value', [('HF_DISCOVERY_MODE','pilot'),
    ('HF_DISCOVERY_INTERVAL_SECONDS','2'), ('HF_DISCOVERY_BATCH_SIZE','101'),
    ('HF_DISCOVERY_INTERVAL_SECONDS','invalid')])
async def test_bad_settings_fail_closed(monkeypatch, key, value):
    monkeypatch.setenv('HF_DISCOVERY_MODE','shadow')
    monkeypatch.setenv(key,value)
    worker = runtime.DiscoveryRuntime()
    await worker.start()
    assert worker.status()['scheduler'] == 'blocked'
    assert worker.task is None


@pytest.mark.asyncio
async def test_cycle_shutdown_and_exclusive_cli_lock(tmp_path, monkeypatch):
    path=tmp_path/'source.db'
    monkeypatch.setenv('HF_DISCOVERY_MODE','shadow')
    monkeypatch.setenv('HF_DISCOVERY_DB_PATH',str(path))
    finished=asyncio.Event()
    async def collect(*args):
        return {'status':'failed','reason':'ConnectError'}
    async def prepare(*args, **kwargs):
        finished.set()
        return {'blocked':True,'reason':'ModuleNotFoundError'}
    monkeypatch.setattr(runtime,'collect_once',collect)
    monkeypatch.setattr(runtime,'prepare_once',prepare)
    worker=runtime.DiscoveryRuntime()
    await worker.start()
    await asyncio.wait_for(finished.wait(),2)
    with pytest.raises(RuntimeError):
        with runtime.worker_lock(path.with_suffix('.worker.lock')):
            pass
    await worker.stop()
    status=worker.status()
    assert status['scheduler']=='stopped'
    assert status['last_cycle']['collection']['reason']=='ConnectError'
    assert status['last_cycle']['preparation']['blocked']
    assert status['serving_enabled'] is False
    with runtime.worker_lock(path.with_suffix('.worker.lock')):
        pass


@pytest.mark.asyncio
async def test_second_owner_does_not_collect(tmp_path, monkeypatch):
    path=tmp_path/'source.db'
    monkeypatch.setenv('HF_DISCOVERY_MODE','shadow')
    monkeypatch.setenv('HF_DISCOVERY_DB_PATH',str(path))
    collect=AsyncMock()
    monkeypatch.setattr(runtime,'collect_once',collect)
    with runtime.worker_lock(path.with_suffix('.worker.lock')):
        worker=runtime.DiscoveryRuntime()
        await worker.start()
        await worker.task
    assert worker.status()['scheduler']=='blocked'
    collect.assert_not_called()


@pytest.mark.asyncio
async def test_missing_model_does_not_burn_candidate_retries(tmp_path, monkeypatch):
    from app import embed_svc
    store=DiscoveryStore(tmp_path/'source.db')
    store.save_snapshot(snapshot(),NOW)
    def missing():
        raise ModuleNotFoundError('private path should never appear')
    monkeypatch.setattr(embed_svc,'get_model',missing)
    async with httpx.AsyncClient() as client:
        for _ in range(4):
            report=await prepare_once(store,client,now=NOW)
            assert report['blocked'] is True
            assert report['reason']=='ModuleNotFoundError'
    assert store.pending(NOW)[0]['attempts']==0


@pytest.mark.asyncio
async def test_health_endpoint_only_reads_cached_status(monkeypatch):
    from app.main import app
    worker=runtime.DiscoveryRuntime()
    worker.state.update(scheduler='running',last_cycle={'collection':{'status':'ok'}})
    monkeypatch.setattr(app.state,'discovery',worker,raising=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
        response=await client.get('/healthz/discovery')
    assert response.status_code==200
    assert response.json()['scheduler']=='running'
    assert response.json()['serving_enabled'] is False


@pytest.mark.asyncio
async def test_recovers_on_next_cycle_after_unexpected_failure(tmp_path, monkeypatch):
    monkeypatch.setenv('HF_DISCOVERY_MODE','shadow')
    monkeypatch.setenv('HF_DISCOVERY_DB_PATH',str(tmp_path/'source.db'))
    calls=0
    async def collect(*args):
        nonlocal calls
        calls+=1
        if calls==1:
            raise OSError('sensitive details')
        worker.stop_event.set()
        return {'status':'ok'}
    monkeypatch.setattr(runtime,'collect_once',collect)
    monkeypatch.setattr(runtime,'prepare_once',AsyncMock(return_value={'ready':0}))
    worker=runtime.DiscoveryRuntime()
    await worker.start()
    worker.interval=.001
    await asyncio.wait_for(worker.task,2)
    assert calls==2
    assert worker.status()['last_cycle']['collection']['status']=='ok'
    assert 'reason' not in worker.status()
