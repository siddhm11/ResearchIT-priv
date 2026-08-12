"""
Durable user data: local SQLite stays the working store, Turso is the backup.

Why not just move db.py onto Turso
-----------------------------------
Turso is ~968 ms per round trip from the Space (measured; the database is in
aws-ap-south-1 and the Space is not). A single page load makes 5-10 database
calls, so a direct port would turn every request into 5-10 seconds. Reads have
to stay local.

Why this is needed at all
--------------------------
DB_PATH is /tmp/interactions.db on an ephemeral filesystem. Every rebuild
destroys every save, EWMA profile, cluster and onboarding record. Users are
re-onboarded from scratch, and no interaction history can ever accumulate --
which is also why there is nothing to retrain the reranker on.

Design
------
    boot   restore local SQLite from Turso, but only when local is empty
    read   untouched -- app/db.py keeps its aiosqlite calls
    write  untouched -- picked up by the next sync
    sync   every TURSO_SYNC_INTERVAL seconds, push rows newer than a watermark

Incremental rather than write-through, deliberately: it needs no changes to the
six existing write paths and cannot coupling-break when their SQL changes. The
cost is a bounded loss window equal to the sync interval.

Only user data is replicated. paper_metadata and paper_qdrant_map are caches
rebuildable from Turso and Qdrant, so copying them would waste quota.
"""
from __future__ import annotations

import asyncio
import base64
import json
import time

import aiosqlite
import httpx

from app import config

SYNC_INTERVAL = int(__import__("os").getenv("TURSO_SYNC_INTERVAL", "60"))

# local table -> (remote table, watermark column, primary key columns)
#
# interactions is append-only with an AUTOINCREMENT id, so an integer watermark
# is exact. The other three are upserted in place, so they use a timestamp.
TABLES: dict[str, dict] = {
    "interactions": {
        "remote": "user_interactions",
        "watermark": "id",
        "wm_kind": "int",
        "pk": ["id"],
        "cols": ["id", "user_id", "paper_id", "event_type", "source", "position",
                 "query_id", "ranker_version", "candidate_source", "cluster_id",
                 "propensity", "policy_id", "timestamp"],
        "blobs": [],
    },
    "user_profiles": {
        "remote": "user_profiles",
        "watermark": "updated_at",
        "wm_kind": "text",
        "pk": ["user_id", "profile_type"],
        "cols": ["user_id", "profile_type", "vector", "interaction_count", "updated_at"],
        "blobs": ["vector"],
    },
    "user_clusters": {
        "remote": "user_clusters",
        "watermark": "computed_at",
        "wm_kind": "text",
        "pk": ["user_id", "cluster_idx"],
        "cols": ["user_id", "cluster_idx", "medoid_paper_id", "medoid_embedding_blob",
                 "importance", "paper_ids", "computed_at"],
        "blobs": ["medoid_embedding_blob"],
    },
    "user_onboarding": {
        "remote": "user_onboarding",
        "watermark": "updated_at",
        "wm_kind": "text",
        "pk": ["user_id"],
        "cols": ["user_id", "selected_categories", "onboarding_completed",
                 "created_at", "updated_at"],
        "blobs": [],
    },
}

REMOTE_DDL = [
    """CREATE TABLE IF NOT EXISTS user_interactions (
        id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, paper_id TEXT NOT NULL,
        event_type TEXT NOT NULL, source TEXT, position INTEGER, query_id TEXT,
        ranker_version TEXT, candidate_source TEXT, cluster_id INTEGER,
        propensity REAL, policy_id TEXT, timestamp TEXT)""",
    "CREATE INDEX IF NOT EXISTS idx_ui_user ON user_interactions(user_id)",
    """CREATE TABLE IF NOT EXISTS user_profiles (
        user_id TEXT NOT NULL, profile_type TEXT NOT NULL, vector BLOB NOT NULL,
        interaction_count INTEGER DEFAULT 0, updated_at TEXT,
        PRIMARY KEY (user_id, profile_type))""",
    """CREATE TABLE IF NOT EXISTS user_clusters (
        user_id TEXT NOT NULL, cluster_idx INTEGER NOT NULL,
        medoid_paper_id TEXT NOT NULL, medoid_embedding_blob BLOB,
        importance REAL, paper_ids TEXT, computed_at TEXT,
        PRIMARY KEY (user_id, cluster_idx))""",
    """CREATE TABLE IF NOT EXISTS user_onboarding (
        user_id TEXT PRIMARY KEY, selected_categories TEXT,
        onboarding_completed INTEGER DEFAULT 0, created_at TEXT, updated_at TEXT)""",
]

_LOCAL_STATE_DDL = """CREATE TABLE IF NOT EXISTS sync_state (
    key TEXT PRIMARY KEY, value TEXT)"""

_task: asyncio.Task | None = None
_last: dict = {"ok": None, "at": None, "pushed": 0, "error": None}


def enabled() -> bool:
    """Whether user-data replication runs.

    TURSO_SYNC_DISABLED exists because reads and replication share one pair of
    credentials. A developer running a local server against real credentials --
    to exercise search or the feed, which need Turso metadata -- also silently
    gets replication, and the shutdown flush pushes whatever that session
    created into the production database. That has happened; nine synthetic
    saves reached production from a UI test.

    Setting this to 1 keeps turso_svc metadata reads working while making the
    session read-only with respect to user data.
    """
    if __import__("os").getenv("TURSO_SYNC_DISABLED", "").strip().lower() in (
            "1", "true", "yes"):
        return False
    return bool(config.TURSO_URL and config.TURSO_DB_TOKEN)


# ── Turso HTTP ───────────────────────────────────────────────────────────────

def _url() -> str:
    u = config.TURSO_URL.rstrip("/")
    if u.startswith("libsql://"):
        u = "https://" + u[len("libsql://"):]
    elif not u.startswith("https://"):
        u = "https://" + u.lstrip("http://")
    return u


def _cell(v):
    """Python value -> libSQL argument."""
    if v is None:
        return {"type": "null", "value": None}
    if isinstance(v, bool):
        return {"type": "integer", "value": str(int(v))}
    if isinstance(v, int):
        return {"type": "integer", "value": str(v)}
    if isinstance(v, float):
        return {"type": "float", "value": v}
    if isinstance(v, (bytes, bytearray, memoryview)):
        # Hrana keys blob payloads as "base64", not "value" — using "value"
        # here returns a bare 400 with no explanation.
        return {"type": "blob", "base64": base64.b64encode(bytes(v)).decode()}
    return {"type": "text", "value": str(v)}


def _b64decode(s: str) -> bytes:
    """Decode a Hrana blob payload.

    Turso returns base64 with the padding stripped, and occasionally the
    URL-safe alphabet, so a plain b64decode raises "Incorrect padding".
    """
    s = s or ""
    s += "=" * (-len(s) % 4)
    try:
        return base64.b64decode(s)
    except Exception:
        return base64.urlsafe_b64decode(s)


def _uncell(c):
    t = c.get("type")
    if t == "null":
        return None
    if t == "blob":
        return _b64decode(c.get("base64") or c.get("value") or "")
    v = c.get("value")
    if t == "integer":
        return int(v)
    if t == "float":
        return float(v)
    return v


async def _execute(stmts: list[dict], timeout: int = 60) -> list[list[list]]:
    """Run statements in one round trip. Returns rows per statement."""
    payload = {"requests": [{"type": "execute", "stmt": s} for s in stmts]
                           + [{"type": "close"}]}
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            f"{_url()}/v2/pipeline", json=payload,
            headers={"Authorization": f"Bearer {config.TURSO_DB_TOKEN}",
                     "Content-Type": "application/json"})
        if r.status_code >= 400:
            # Surface the body. Turso returns the actual reason there, and
            # raise_for_status() alone gives a bare status with no diagnosis.
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
    out = []
    for res in data.get("results", []):
        if res.get("type") == "error":
            raise RuntimeError(str(res.get("error"))[:250])
        resp = res.get("response", {})
        if resp.get("type") != "execute":
            continue
        rows = resp.get("result", {}).get("rows", [])
        out.append([[_uncell(c) for c in row] for row in rows])
    return out


# ── Watermarks ───────────────────────────────────────────────────────────────

async def _get_wm(conn, table: str) -> str | None:
    cur = await conn.execute(
        "SELECT value FROM sync_state WHERE key = ?", (f"wm:{table}",))
    row = await cur.fetchone()
    return row[0] if row else None


async def _set_wm(conn, table: str, value) -> None:
    await conn.execute(
        "INSERT OR REPLACE INTO sync_state (key, value) VALUES (?, ?)",
        (f"wm:{table}", str(value)))


# ── Public API ───────────────────────────────────────────────────────────────

async def ensure_remote_schema() -> None:
    await _execute([{"sql": d} for d in REMOTE_DDL])


async def restore() -> dict:
    """Pull user data from Turso into the local DB.

    Only fills tables that are locally EMPTY. A container that already has data
    is never clobbered, so this is safe to run on every boot.
    """
    restored = {}
    async with aiosqlite.connect(config.DB_PATH) as conn:
        await conn.execute(_LOCAL_STATE_DDL)
        await conn.commit()
        for local, spec in TABLES.items():
            cur = await conn.execute(f"SELECT COUNT(*) FROM {local}")
            n = (await cur.fetchone())[0]
            if n:
                continue  # local already has data; leave it alone
            cols = ", ".join(spec["cols"])
            try:
                rows = (await _execute(
                    [{"sql": f"SELECT {cols} FROM {spec['remote']}"}]))[0]
            except Exception as e:
                print(f"[turso_sync] restore {local} failed: {str(e)[:120]}")
                continue
            if not rows:
                continue
            ph = ", ".join("?" * len(spec["cols"]))
            await conn.executemany(
                f"INSERT OR REPLACE INTO {local} ({cols}) VALUES ({ph})", rows)
            # Restored rows came FROM Turso, so advance the watermark past them
            # or the next sync would push them straight back.
            wm = spec["watermark"]
            cur = await conn.execute(f"SELECT MAX({wm}) FROM {local}")
            hi = (await cur.fetchone())[0]
            if hi is not None:
                await _set_wm(conn, local, hi)
            restored[local] = len(rows)
        await conn.commit()
    if restored:
        print(f"[turso_sync] restored from Turso: {restored}")
    return restored


async def sync_once() -> dict:
    """Push rows newer than the watermark. Returns {table: count}."""
    pushed = {}
    async with aiosqlite.connect(config.DB_PATH) as conn:
        await conn.execute(_LOCAL_STATE_DDL)
        await conn.commit()
        for local, spec in TABLES.items():
            wm_col, kind = spec["watermark"], spec["wm_kind"]
            wm = await _get_wm(conn, local)
            if wm is None:
                where, args = "", ()
            else:
                where = f"WHERE {wm_col} > ?"
                args = (int(wm),) if kind == "int" else (wm,)
            cols = ", ".join(spec["cols"])
            cur = await conn.execute(
                f"SELECT {cols} FROM {local} {where} ORDER BY {wm_col} LIMIT 500", args)
            rows = await cur.fetchall()
            if not rows:
                continue

            ph = ", ".join("?" * len(spec["cols"]))
            stmts = [{
                "sql": f"INSERT OR REPLACE INTO {spec['remote']} ({cols}) VALUES ({ph})",
                "args": [_cell(v) for v in row],
            } for row in rows]
            await _execute(stmts, timeout=120)

            wm_idx = spec["cols"].index(wm_col)
            await _set_wm(conn, local, rows[-1][wm_idx])
            pushed[local] = len(rows)
        await conn.commit()
    return pushed


async def _loop() -> None:
    while True:
        try:
            await asyncio.sleep(SYNC_INTERVAL)
            pushed = await sync_once()
            _last.update(ok=True, at=time.time(),
                         pushed=sum(pushed.values()), error=None)
            if pushed:
                print(f"[turso_sync] pushed {pushed}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _last.update(ok=False, at=time.time(), error=str(e)[:200])
            print(f"[turso_sync] sync failed (will retry): {str(e)[:150]}")


async def start() -> None:
    """Called from the app lifespan. Never raises — sync is best-effort."""
    global _task
    if not enabled():
        print("[turso_sync] TURSO not configured — user data stays ephemeral")
        return
    try:
        await ensure_remote_schema()
        await restore()
    except Exception as e:
        print(f"[turso_sync] startup failed ({str(e)[:150]}) — continuing without sync")
        return
    _task = asyncio.create_task(_loop())
    print(f"[turso_sync] active, every {SYNC_INTERVAL}s")


async def stop() -> None:
    """Final flush on shutdown so the last interval is not lost."""
    global _task
    if _task:
        _task.cancel()
        try:
            await _task
        except (asyncio.CancelledError, Exception):
            pass
        _task = None
    if enabled():
        try:
            await sync_once()
        except Exception as e:
            print(f"[turso_sync] final flush failed: {str(e)[:120]}")


def status() -> dict:
    return {
        "enabled": enabled(),
        "running": _task is not None and not _task.done(),
        "interval_s": SYNC_INTERVAL,
        "last_ok": _last["ok"],
        "last_at": _last["at"],
        "last_pushed": _last["pushed"],
        "last_error": _last["error"],
    }
