"""Opt-in, single-host shadow scheduler owned by the FastAPI lifespan."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import time

import httpx

from app.discovery_store import DiscoveryStore
from app.discovery_worker import collect_once, prepare_once


@contextmanager
def worker_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another discovery worker holds this store lock') from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class DiscoveryRuntime:
    def __init__(self):
        self.task = None
        self.stop_event = asyncio.Event()
        self.state = {'scheduler': 'disabled', 'serving_enabled': False}

    async def start(self):
        mode = os.getenv('HF_DISCOVERY_MODE', 'off').strip().lower()
        if mode == 'off':
            return
        try:
            if mode != 'shadow':
                raise ValueError('Only off and shadow are supported')
            self.interval = int(os.getenv('HF_DISCOVERY_INTERVAL_SECONDS', '21600'))
            self.limit = int(os.getenv('HF_DISCOVERY_BATCH_SIZE', '10'))
            if self.interval < 300 or not 1 <= self.limit <= 100:
                raise ValueError('Invalid interval or batch size')
            self.path = Path(os.getenv('HF_DISCOVERY_DB_PATH', 'data/discovery.sqlite')).resolve()
        except (ValueError, OSError) as exc:
            self.state.update(scheduler='blocked', reason=type(exc).__name__)
            return
        if self.task is not None:
            return
        self.state['scheduler'] = 'starting'
        self.task = asyncio.create_task(self._run(), name='hf-discovery-shadow')

    async def _run(self):
        try:
            # Same lock as the CLI: a second owner never runs a collection cycle.
            with worker_lock(self.path.with_suffix('.worker.lock')):
                store = await asyncio.to_thread(DiscoveryStore, self.path)
                self.state['scheduler'] = 'running'
                async with httpx.AsyncClient() as client:
                    while not self.stop_event.is_set():
                        try:
                            result = {'collection': await collect_once(store, client)}
                            result['preparation'] = await prepare_once(store, client, limit=self.limit)
                            result['store'] = await asyncio.to_thread(store.status, time.time())
                            self.state.update(last_cycle=result, checked_at=time.time())
                            self.state.pop('reason', None)
                        except Exception as exc:
                            self.state.update(reason=type(exc).__name__, checked_at=time.time())
                        try:
                            await asyncio.wait_for(self.stop_event.wait(), timeout=self.interval)
                        except asyncio.TimeoutError:
                            pass
        except Exception as exc:
            self.state.update(scheduler='blocked', reason=type(exc).__name__)
        finally:
            if self.state['scheduler'] == 'running':
                self.state['scheduler'] = 'stopped'

    async def stop(self):
        self.stop_event.set()
        if self.task is not None:
            # Drain the bounded active batch rather than cancel to_thread while
            # it still owns SQLite/model work and prematurely release the lock.
            await self.task

    def status(self):
        result = dict(self.state)
        checked = result.get('checked_at')
        if checked is not None:
            result['last_cycle_age_seconds'] = max(0, time.time() - checked)
        return result
