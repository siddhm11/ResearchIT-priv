"""Persistent source observations and locally encoded candidates for shadow work.

Separate SQLite file: never writes user history, Turso, or vector collections.
Synchronous batch API; call via asyncio.to_thread from asynchronous workers.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

EMBEDDING_CONTRACT = 'BAAI/bge-m3:1024:title256-space-abstract1024:max_length512:v1'


def content_hash(paper: dict) -> str:
    text = f"{paper['title'][:256]} {paper.get('abstract', '')[:1024]}"
    return hashlib.sha256(text.encode()).hexdigest()


def validated_vector(vector) -> list[float]:
    a = np.asarray(vector, dtype=np.float32)
    if a.shape != (1024,) or not np.isfinite(a).all() or np.linalg.norm(a) < 1e-9:
        raise ValueError('Expected a finite nonzero 1024-dimensional vector')
    # Cosine consumers expect unit vectors. Reject grossly wrong encodings (e.g. uint8).
    if not .9 <= np.linalg.norm(a) <= 1.1:
        raise ValueError('Expected a normalized BGE-M3 vector')
    return (a / np.linalg.norm(a)).tolist()


class DiscoveryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as c:
            if c.execute("SELECT name FROM sqlite_master WHERE name='interactions'").fetchone():
                raise ValueError('Use separate discovery storage, not the user database')
            c.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS runs (
                  id INTEGER PRIMARY KEY, observed REAL NOT NULL,
                  status TEXT NOT NULL, detail TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS snapshots (
                  digest TEXT PRIMARY KEY, observed REAL NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS candidates (
                  id TEXT PRIMARY KEY, paper TEXT NOT NULL, content_hash TEXT NOT NULL,
                  first_seen REAL NOT NULL, last_seen REAL NOT NULL,
                  vector TEXT, contract TEXT, attempts INTEGER NOT NULL DEFAULT 0,
                  next_attempt REAL NOT NULL DEFAULT 0, error TEXT);
                CREATE INDEX IF NOT EXISTS candidates_seen ON candidates(last_seen);
            ''')

    @contextmanager
    def connection(self):
        c = sqlite3.connect(self.path, timeout=15)
        c.row_factory = sqlite3.Row
        try:
            with c:
                yield c
        finally:
            c.close()

    def save_snapshot(self, snapshot: dict, now: float) -> None:
        """Atomically append observation + merge candidates; same snapshot is idempotent."""
        from datetime import datetime
        observed = datetime.fromisoformat(snapshot['observed_at']).timestamp()
        if observed > now + 1:
            raise ValueError('Future source observation')
        payload = json.dumps(snapshot, sort_keys=True, allow_nan=False)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        with self.connection() as c:
            c.execute('BEGIN IMMEDIATE')
            if c.execute('SELECT 1 FROM snapshots WHERE digest=?', (digest,)).fetchone():
                return
            c.execute('INSERT INTO snapshots VALUES (?,?,?)', (digest, observed, payload))
            for paper in snapshot['papers']:
                old = c.execute('SELECT * FROM candidates WHERE id=?', (paper['arxiv_id'],)).fetchone()
                if old and old['last_seen'] > observed:
                    continue  # delayed replay cannot roll current metadata backwards
                # Do not discard an abstract obtained from arXiv when HF omits it.
                if old and not paper.get('abstract'):
                    paper = {**paper, 'abstract': json.loads(old['paper']).get('abstract', '')}
                hashed = content_hash(paper)
                if old:
                    changed = old['content_hash'] != hashed
                    c.execute('''UPDATE candidates SET paper=?, content_hash=?, last_seen=?,
                        vector=CASE WHEN ? THEN NULL ELSE vector END,
                        contract=CASE WHEN ? THEN NULL ELSE contract END,
                        attempts=CASE WHEN ? THEN 0 ELSE attempts END,
                        next_attempt=CASE WHEN ? THEN 0 ELSE next_attempt END,
                        error=CASE WHEN ? THEN NULL ELSE error END WHERE id=?''',
                        (json.dumps(paper), hashed, observed, *([changed]*5), paper['arxiv_id']))
                else:
                    c.execute('INSERT INTO candidates(id,paper,content_hash,first_seen,last_seen) VALUES(?,?,?,?,?)',
                              (paper['arxiv_id'], json.dumps(paper), hashed, observed, observed))
            c.execute('INSERT INTO runs(observed,status,detail) VALUES(?,?,?)',
                      (now, 'ok', json.dumps({'accepted':len(snapshot['papers']), 'rejected':snapshot['rejected']})))

    def record_failure(self, now: float, error: str) -> None:
        with self.connection() as c:
            c.execute('INSERT INTO runs(observed,status,detail) VALUES(?,?,?)', (now,'failed',error))

    def pending(self, now: float, limit: int = 10, ttl: float = 48*3600) -> list[dict]:
        with self.connection() as c:
            rows = c.execute('''SELECT * FROM candidates WHERE
                (vector IS NULL OR contract IS NULL OR contract != ?) AND attempts < 3
                AND next_attempt <= ? AND last_seen >= ? AND last_seen <= ?
                ORDER BY last_seen DESC,id LIMIT ?''',
                (EMBEDDING_CONTRACT,now,now-ttl,now,max(0,min(limit,100)))).fetchall()
            return [dict(r) for r in rows]

    def finish(self, row: dict, paper: dict, vector, now: float, contract: str) -> bool:
        if contract != EMBEDDING_CONTRACT:
            raise ValueError('Incompatible embedding contract')
        vec = validated_vector(vector)
        if paper['arxiv_id'] != row['id'] or not paper.get('abstract','').strip():
            raise ValueError('Missing or mismatched paper metadata')
        with self.connection() as c:
            # Optimistic compare prevents a slow encoder overwriting newer source content.
            result = c.execute('''UPDATE candidates SET paper=?,content_hash=?,vector=?,contract=?,
                attempts=0,next_attempt=0,error=NULL WHERE id=? AND content_hash=? AND last_seen=?''',
                (json.dumps(paper),content_hash(paper),json.dumps(vec),contract,row['id'],row['content_hash'],row['last_seen']))
            return result.rowcount == 1

    def fail_candidate(self, row: dict, now: float, error: str) -> None:
        with self.connection() as c:
            c.execute('''UPDATE candidates SET attempts=attempts+1,next_attempt=?,error=?
                WHERE id=? AND content_hash=? AND last_seen=?''',
                (now+min(3600,300*2**row['attempts']),error,row['id'],row['content_hash'],row['last_seen']))

    def ready(self, now: float, ttl: float = 48*3600) -> list[dict]:
        with self.connection() as c:
            rows = c.execute('''SELECT * FROM candidates WHERE vector IS NOT NULL AND contract=?
                AND last_seen >= ? AND last_seen <= ? ORDER BY last_seen DESC,id''',
                (EMBEDDING_CONTRACT,now-ttl,now)).fetchall()
            return [{**json.loads(r['paper']), 'vector':json.loads(r['vector']),
                     'embedding_contract':r['contract'], 'last_seen':r['last_seen']} for r in rows]

    def status(self, now: float) -> dict:
        with self.connection() as c:
            latest=c.execute('SELECT MAX(observed) FROM snapshots').fetchone()[0]
            last=c.execute('SELECT status,detail FROM runs ORDER BY id DESC LIMIT 1').fetchone()
            count=c.execute('SELECT COUNT(*) FROM candidates').fetchone()[0]
            exhausted=c.execute('SELECT COUNT(*) FROM candidates WHERE attempts >= 3').fetchone()[0]
        return {'candidates':count,'ready':len(self.ready(now)), 'retry_exhausted':exhausted,
                'last_success_age_hours':(now-latest)/3600 if latest is not None else None,
                'source_stale':latest is None or now-latest>48*3600,
                'last_run':dict(last) if last else None}
