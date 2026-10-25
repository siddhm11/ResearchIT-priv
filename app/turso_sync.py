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

With a mirror (TURSO_MIRROR_URL, see app/turso_remotes.py)
-----------------------------------------------------------
    boot   ask every remote how far each table goes (one MAX per table, one
           request per remote) and restore each empty table from the remote
           that is furthest ahead, falling back to the other if that fails
    sync   push to every remote, each with its own watermark. A remote that is
           down keeps its watermark and catches up when it answers again; it
           never holds up the other.
    guard  a remote that could not be asked at boot is asked at its first push.
           If it holds rows newer than anything this container saw, pushing
           there could overwrite them (the same id collision restore() guards
           against), so that table is held back from that remote and listed in
           status()["remotes"][name]["held"].
Without a mirror none of this runs: one remote, the code path as before.
"""
from __future__ import annotations

import asyncio
import base64
import json
import time

import aiosqlite
import httpx

from app import config
from app import http_client
from app import turso_remotes
from app.turso_remotes import Remote

SYNC_INTERVAL = int(__import__("os").getenv("TURSO_SYNC_INTERVAL", "60"))

# Tables this container must not push, because its restore did not complete
# and pushing would overwrite remote rows it never saw. See restore().
_no_push: set[str] = set()

# Mirror mode only (more than one remote). Per table: the newest watermark value
# this container knew at boot, restored or reserved. A remote holding anything
# newer holds rows this container never saw.
_seen: dict[str, object] = {}
# remote name -> tables whose watermark on that remote is settled for this boot.
_ready: dict[str, set[str]] = {}
# remote name -> tables not pushed there, because it holds rows we never saw.
_held: dict[str, set[str]] = {}
# table -> remote it was restored from.
_restored_from: dict[str, str] = {}
# remotes whose schema has been ensured this boot.
_schema_ok: set[str] = set()
# remote name -> outcome of the last push to it.
_remote_last: dict[str, dict] = {}

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
                 "propensity", "policy_id", "timestamp", "layout"],
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
    # Paper-level caches that cost money or outside requests to rebuild. Each
    # has an AUTOINCREMENT `id` as its watermark (a one-second timestamp skips
    # rows in a burst) and a UNIQUE natural key, so INSERT OR REPLACE here and
    # on the remote both resolve an update to one row. `paged` restores them in
    # chunks: one unpaged SELECT of a large table times out at boot.
    "paper_explanations": {
        "remote": "paper_explanations",
        "watermark": "id",
        "wm_kind": "int",
        "pk": ["id"],
        "cols": ["id", "cache_key", "arxiv_id", "explanation", "model", "created_at"],
        "blobs": [],
        "paged": True,
    },
    "paper_abstracts": {
        "remote": "paper_abstracts",
        "watermark": "id",
        "wm_kind": "int",
        "pk": ["id"],
        "cols": ["id", "arxiv_id", "abstract", "source", "fetched_at"],
        "blobs": [],
        "paged": True,
    },
    "paper_sources": {
        "remote": "paper_sources",
        "watermark": "id",
        "wm_kind": "int",
        "pk": ["id"],
        "cols": ["id", "arxiv_id", "status", "version", "licence", "licence_text", "figures_json",
                 "n_figures", "n_sections", "n_paragraphs", "code_repo", "code_stars",
                 "code_checked_at", "parser_version", "fetched_at", "error"],
        "blobs": [],
        "paged": True,
    },
    "llm_calls": {
        "remote": "llm_calls",
        "watermark": "id",
        "wm_kind": "int",
        "pk": ["id"],
        "cols": ["id", "feature", "provider", "model", "chars_in", "chars_out", "usd_est",
                 "latency_ms", "status", "created_at", "tokens_in", "tokens_out", "model_rank",
                 "truncated", "requests_left", "tokens_left", "subject"],
        "blobs": [],
        "paged": True,
    },
    "paper_briefings": {
        "remote": "paper_briefings",
        "watermark": "id",
        "wm_kind": "int",
        "pk": ["id"],
        "cols": ["id", "cache_key", "arxiv_id", "status", "briefing_json", "provider", "model",
                 "prompt_version", "created_at"],
        "blobs": [],
        "paged": True,
    },
    "paper_answers": {
        "remote": "paper_answers",
        "watermark": "id",
        "wm_kind": "int",
        "pk": ["id"],
        "cols": ["id", "cache_key", "arxiv_id", "question_key", "answer_json", "model",
                 "prompt_version", "created_at"],
        "blobs": [],
        "paged": True,
    },
    "content_reports": {
        "remote": "content_reports",
        "watermark": "id",
        "wm_kind": "int",
        "pk": ["id"],
        "cols": ["id", "arxiv_id", "feature", "reason", "user_id", "created_at"],
        "blobs": [],
        "paged": True,
    },    "search_log": {
        "remote": "search_log",
        "watermark": "id",
        "wm_kind": "int",
        "pk": ["id"],
        "cols": ["id", "created_at", "query", "query_id", "n_results", "mode", "rewritten", "latency_ms", "owner"],
        "blobs": [],
        "paged": True,
    },
    "search_overviews": {
        "remote": "search_overviews",
        "watermark": "id",
        "wm_kind": "int",
        "pk": ["id"],
        "cols": ["id", "cache_key", "query", "paper_ids", "overview", "created_at"],
        "blobs": [],
        "paged": True,
    },
    "reader_days": {
        "remote": "reader_days",
        "watermark": "id",
        "wm_kind": "int",
        "pk": ["id"],
        "cols": ["id", "day", "user_id", "owner", "pages", "searches", "feed_pages", "opens", "saves",
                 "passes", "asks"],
        "blobs": [],
        "paged": True,
    },
    "ai_daily": {
        "remote": "ai_daily",
        "watermark": "id",
        "wm_kind": "int",
        "pk": ["id"],
        "cols": ["id", "day", "feature", "metric", "n"],
        "blobs": [],
        "paged": True,
    },
}

RESTORE_PAGE = 2000

REMOTE_DDL = [
    """CREATE TABLE IF NOT EXISTS user_interactions (
        id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, paper_id TEXT NOT NULL,
        event_type TEXT NOT NULL, source TEXT, position INTEGER, query_id TEXT,
        ranker_version TEXT, candidate_source TEXT, cluster_id INTEGER,
        propensity REAL, policy_id TEXT, timestamp TEXT, layout TEXT)""",
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
    """CREATE TABLE IF NOT EXISTS paper_explanations (
        id INTEGER PRIMARY KEY, cache_key TEXT NOT NULL UNIQUE, arxiv_id TEXT NOT NULL,
        explanation TEXT NOT NULL, model TEXT, created_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS paper_sources (
        id INTEGER PRIMARY KEY, arxiv_id TEXT NOT NULL UNIQUE, status TEXT NOT NULL,
        version TEXT, licence TEXT, licence_text TEXT, figures_json TEXT, n_figures INTEGER,
        n_sections INTEGER, n_paragraphs INTEGER, code_repo TEXT, code_stars INTEGER,
        code_checked_at TEXT, parser_version INTEGER, fetched_at TEXT, error TEXT)""",
    """CREATE TABLE IF NOT EXISTS llm_calls (
        id INTEGER PRIMARY KEY, feature TEXT NOT NULL, provider TEXT, model TEXT,
        chars_in INTEGER, chars_out INTEGER, usd_est REAL, latency_ms INTEGER,
        status TEXT NOT NULL, created_at TEXT, tokens_in INTEGER, tokens_out INTEGER,
        model_rank INTEGER, truncated INTEGER, requests_left INTEGER, tokens_left INTEGER, subject TEXT)""",
    """CREATE TABLE IF NOT EXISTS paper_briefings (
        id INTEGER PRIMARY KEY, cache_key TEXT NOT NULL UNIQUE, arxiv_id TEXT NOT NULL,
        status TEXT NOT NULL, briefing_json TEXT, provider TEXT, model TEXT,
        prompt_version TEXT, created_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS paper_answers (
        id INTEGER PRIMARY KEY, cache_key TEXT NOT NULL UNIQUE, arxiv_id TEXT NOT NULL,
        question_key TEXT NOT NULL, answer_json TEXT NOT NULL, model TEXT,
        prompt_version TEXT, created_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS content_reports (
        id INTEGER PRIMARY KEY, arxiv_id TEXT NOT NULL, feature TEXT NOT NULL,
        reason TEXT, user_id TEXT, created_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS search_log (
        id INTEGER PRIMARY KEY, created_at TEXT, query TEXT, query_id TEXT, n_results INTEGER,
        mode TEXT, rewritten TEXT, latency_ms INTEGER, owner INTEGER)""",
    """CREATE TABLE IF NOT EXISTS search_overviews (
        id INTEGER PRIMARY KEY, cache_key TEXT NOT NULL UNIQUE, query TEXT, paper_ids TEXT,
        overview TEXT, created_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS reader_days (
        id INTEGER PRIMARY KEY, day TEXT NOT NULL, user_id TEXT NOT NULL, owner INTEGER, pages INTEGER,
        searches INTEGER, feed_pages INTEGER, opens INTEGER, saves INTEGER, passes INTEGER, asks INTEGER,
        UNIQUE (day, user_id))""",
    """CREATE TABLE IF NOT EXISTS ai_daily (
        id INTEGER PRIMARY KEY, day TEXT NOT NULL, feature TEXT NOT NULL, metric TEXT NOT NULL,
        n INTEGER, UNIQUE (day, feature, metric))""",
    """CREATE TABLE IF NOT EXISTS paper_abstracts (
        id INTEGER PRIMARY KEY, arxiv_id TEXT NOT NULL UNIQUE, abstract TEXT NOT NULL,
        source TEXT, fetched_at TEXT)""",
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
    """Run statements on the primary in one round trip. Returns rows per statement."""
    return await _post(_url(), config.TURSO_DB_TOKEN, stmts, timeout)


async def _run(remote: Remote, stmts: list[dict], timeout: int = 60) -> list[list[list]]:
    """Run statements on one remote. The primary goes through _execute, so code
    and tests that know only _execute keep seeing every primary call."""
    if remote.name == "primary":
        return await _execute(stmts, timeout=timeout)
    return await _post(remote.url, remote.token, stmts, timeout)


async def read(stmts: list[dict], timeout: int = 60) -> list[list[list]]:
    """A read that moves to the mirror while the primary is blocked or down.

    Only outages move it (turso_remotes.is_outage_exception); an error in the
    SQL itself is raised at once, since the mirror would fail the same way.
    """
    last: Exception | None = None
    for r in turso_remotes.read_order():
        try:
            out = await _run(r, stmts, timeout)
        except Exception as e:
            if not turso_remotes.is_outage_exception(e):
                raise
            turso_remotes.bench(r, str(e))
            last = e
            continue
        turso_remotes.recovered(r)
        return out
    if last is not None:
        raise last
    raise RuntimeError("TURSO_URL or TURSO_DB_TOKEN not configured")


async def write_all(stmts: list[dict], timeout: int = 60) -> None:
    """Run a write on every remote, for in-place changes the watermark push can't
    carry (analytics.housekeep clearing old search text). A mirror failure is
    logged, not raised: the primary's outcome is the caller's."""
    await _execute(stmts, timeout=timeout)
    for r in turso_remotes.remotes()[1:]:
        try:
            await _run(r, stmts, timeout)
        except Exception as e:
            print(f"[turso_sync] write to {r.name} failed: {str(e)[:150]}")


async def _post(base: str, token: str, stmts: list[dict], timeout: int) -> list[list[list]]:
    payload = {"requests": [{"type": "execute", "stmt": s} for s in stmts]
                           + [{"type": "close"}]}
    # Shared pooled client — see app/http_client.py. The replication loop runs
    # every 60s, so it was re-handshaking TLS on a timer for its whole lifetime.
    r = await http_client.get_client().post(
        f"{base}/v2/pipeline", json=payload, timeout=timeout,
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"})
    if r.status_code >= 400:
        # Surface the body. Turso returns the actual reason there, and
        # raise_for_status() alone gives a bare status with no diagnosis.
        raise turso_remotes.TursoError(f"HTTP {r.status_code}: {r.text[:300]}",
                                       turso_remotes.is_outage_status(r.status_code))
    data = r.json()
    out = []
    for res in data.get("results", []):
        if res.get("type") == "error":
            raise turso_remotes.TursoError(str(res.get("error"))[:250],
                                           turso_remotes.is_outage_error(res.get("error")))
        resp = res.get("response", {})
        if resp.get("type") != "execute":
            continue
        rows = resp.get("result", {}).get("rows", [])
        out.append([[_uncell(c) for c in row] for row in rows])
    return out


# ── Watermarks ───────────────────────────────────────────────────────────────

def _wm_key(table: str, remote: str) -> str:
    # The primary keeps the key it always had.
    return f"wm:{table}" if remote == "primary" else f"wm@{remote}:{table}"


async def _get_wm(conn, table: str, remote: str = "primary") -> str | None:
    cur = await conn.execute(
        "SELECT value FROM sync_state WHERE key = ?", (_wm_key(table, remote),))
    row = await cur.fetchone()
    return row[0] if row else None


async def _set_wm(conn, table: str, value, remote: str = "primary") -> None:
    await conn.execute(
        "INSERT OR REPLACE INTO sync_state (key, value) VALUES (?, ?)",
        (_wm_key(table, remote), str(value)))


async def _clear_wm(conn, table: str, remote: str) -> None:
    await conn.execute("DELETE FROM sync_state WHERE key = ?", (_wm_key(table, remote),))


def _wm_value(spec: dict, v):
    """A watermark as stored remotely or locally, in a form that compares right."""
    if v is None:
        return None
    return int(v) if spec["wm_kind"] == "int" else str(v)


async def _local_max(conn, local: str, spec: dict):
    cur = await conn.execute(f"SELECT MAX({spec['watermark']}) FROM {local}")
    return _wm_value(spec, (await cur.fetchone())[0])


# ── Public API ───────────────────────────────────────────────────────────────

# Columns added to remote tables after they were first created. CREATE TABLE IF
# NOT EXISTS never adds a column to a table that already exists, so each is
# added here, once; a "duplicate column" error means it is already there, and
# any other error stops start-up of replication rather than pushing rows the
# remote can't take (trap T8 in docs/redesign/03-start-here-agent-handoff.md).
REMOTE_COLUMNS = [
    ("user_interactions", "layout", "TEXT"),
    ("llm_calls", "tokens_in", "INTEGER"),
    ("llm_calls", "tokens_out", "INTEGER"),
    ("llm_calls", "model_rank", "INTEGER"),
    ("llm_calls", "truncated", "INTEGER"),
    ("llm_calls", "requests_left", "INTEGER"),
    ("llm_calls", "tokens_left", "INTEGER"),
    ("llm_calls", "subject", "TEXT"),
]


async def ensure_remote_schema(remote: Remote | None = None) -> None:
    remote = remote or Remote("primary", "", "")
    await _run(remote, [{"sql": d} for d in REMOTE_DDL])
    for table, column, kind in REMOTE_COLUMNS:
        try:
            await _run(remote, [{"sql": f"ALTER TABLE {table} ADD COLUMN {column} {kind}"}])
        except Exception as e:
            if "duplicate column" not in str(e).lower():
                raise
    _schema_ok.add(remote.name)


async def _reserve_id_space(conn, local: str, spec: dict) -> bool:
    """Push the local AUTOINCREMENT counter above the remote MAX(id).

    Guards the one path that can destroy data that is already safe.

    DB_PATH is /tmp on Spaces, so a fresh container starts with an empty local
    DB and its AUTOINCREMENT restarts at 1. If `restore()` cannot pull the rows
    back — a Turso 5xx, a timeout on a large table — the table stays empty and
    its watermark stays None. The first save then gets id 1, `sync_once()` sees
    no watermark so it selects everything, and it pushes with INSERT OR REPLACE
    keyed on `id`. Remote row 1 is a real user's first save; it is overwritten
    by the new container's first save, and the sync reports success.

    Reproduced end to end: five rows of one user's history, restore fails, two
    new saves arrive, one sync tick, and the first two rows of the original
    history are gone.

    Reserving the id range makes the collision impossible rather than merely
    unlikely, and it costs one cheap aggregate — which can succeed even when
    the full-table SELECT that failed above would not. Returns False when even
    that is unavailable, and the caller then refuses to push the table at all.
    """
    try:
        res = await _execute([{"sql": f"SELECT MAX(id) FROM {spec['remote']}"}])
        remote_max = res[0][0][0] if res and res[0] and res[0][0] else None
    except Exception as e:
        print(f"[turso_sync] {local}: cannot read remote MAX(id) ({str(e)[:80]})")
        return False

    if remote_max is None:
        return True                     # remote is empty; nothing to collide with
    return await _reserve_above(conn, local, int(remote_max))


async def _reserve_above(conn, local: str, top: int) -> bool:
    """Make every new local id in `local` greater than `top`."""
    try:
        # sqlite_sequence only exists once an AUTOINCREMENT table has taken a
        # row, so both statements are guarded rather than assumed.
        await conn.execute(
            "INSERT INTO sqlite_sequence (name, seq) SELECT ?, ? "
            "WHERE NOT EXISTS (SELECT 1 FROM sqlite_sequence WHERE name = ?)",
            (local, top, local))
        await conn.execute(
            "UPDATE sqlite_sequence SET seq = ? WHERE name = ? AND seq < ?",
            (top, local, top))
        await conn.commit()
    except Exception as e:
        print(f"[turso_sync] {local}: could not reserve id space ({str(e)[:80]})")
        return False

    print(f"[turso_sync] {local}: local ids now start above {top}")
    return True


async def _fetch_remote(spec: dict, cols: str, remote: Remote | None = None) -> list[list]:
    """Every remote row of one table: one query, or pages of RESTORE_PAGE by id.

    Pages are gathered before anything is written, so a failure partway leaves
    the local table empty and the caller's failure handling applies unchanged.
    """
    remote = remote or Remote("primary", "", "")
    if not spec.get("paged"):
        return (await _run(remote, [{"sql": f"SELECT {cols} FROM {spec['remote']}"}]))[0]
    rows: list[list] = []
    last = 0
    while True:
        page = (await _run(remote, [{
            "sql": (f"SELECT {cols} FROM {spec['remote']} WHERE id > ? "
                    f"ORDER BY id LIMIT {RESTORE_PAGE}"),
            "args": [_cell(last)],
        }]))[0]
        rows.extend(page)
        if len(page) < RESTORE_PAGE:
            return rows
        last = int(page[-1][spec["cols"].index("id")])


async def _remote_maxima(rs: list[Remote], tables: list[str]) -> dict[str, dict[str, object]]:
    """{remote: {table: MAX(watermark) or None}}, one request per remote, both at
    once. A remote that cannot answer is left out."""
    if not tables:
        return {}
    stmts = [{"sql": f"SELECT MAX({TABLES[t]['watermark']}) FROM {TABLES[t]['remote']}"}
             for t in tables]
    answers = await asyncio.gather(*(_run(r, stmts) for r in rs), return_exceptions=True)
    out: dict[str, dict[str, object]] = {}
    for r, ans in zip(rs, answers):
        if isinstance(ans, BaseException):
            print(f"[turso_sync] {r.name}: cannot read how far each table goes ({str(ans)[:120]})")
            continue
        out[r.name] = {t: _wm_value(TABLES[t], res[0][0] if res and res[0] else None)
                       for t, res in zip(tables, ans)}
    return out


def _restore_order(rs: list[Remote], maxima: dict, table: str) -> list[Remote]:
    """Furthest-ahead first; on a tie the primary. Remotes that could not say
    how far they go come last, tried only if every other fetch fails."""
    known = [r for r in rs if r.name in maxima]
    unknown = [r for r in rs if r.name not in maxima]
    kind = TABLES[table]["wm_kind"]
    floor = 0 if kind == "int" else ""
    known.sort(key=lambda r: (maxima[r.name][table] is not None,
                              maxima[r.name][table] if maxima[r.name][table] is not None else floor),
               reverse=True)       # sort is stable, so ties keep the primary first
    return known + unknown


def _newer(spec: dict, a, b) -> bool:
    """a is newer than b (None is older than anything)."""
    if a is None:
        return False
    return b is None or a > b


async def restore() -> dict:
    """Pull user data from Turso into the local DB.

    Only fills tables that are locally EMPTY. A container that already has data
    is never clobbered, so this is safe to run on every boot.
    """
    if turso_remotes.has_mirror():
        return await _restore_mirrored()
    restored = {}
    _no_push.clear()
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
                rows = await _fetch_remote(spec, cols)
            except Exception as e:
                print(f"[turso_sync] restore {local} failed: {str(e)[:120]}")
                # The rows are not here, so this container cannot know which
                # remote ids are taken. Reserve the id range if we can; refuse
                # to push if we cannot. Replicating this boot's writes is worth
                # less than not destroying every earlier boot's.
                if spec["wm_kind"] == "int":
                    if not await _reserve_id_space(conn, local, spec):
                        _no_push.add(local)
                else:
                    # Upserted tables key on (user_id, ...), so a fresh local
                    # row for a returning user would overwrite their restored
                    # profile with a cold one.
                    _no_push.add(local)
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


async def _restore_mirrored() -> dict:
    """restore() with a mirror. Each empty table comes from the remote that is
    furthest ahead, and every remote that answered has its watermark settled
    here: one that is behind gets the rows it lacks from the next pushes.

    Ids: local ids for an id table are reserved above the highest id ANY
    remote reported, not just the source's, so a row this container creates can
    never land on a row another remote already holds under that id.
    """
    restored = {}
    rs = turso_remotes.remotes()
    for s in (_no_push, _seen, _ready, _held, _restored_from):
        s.clear()
    async with aiosqlite.connect(config.DB_PATH) as conn:
        await conn.execute(_LOCAL_STATE_DDL)
        await conn.commit()
        empty = []
        for local, spec in TABLES.items():
            cur = await conn.execute(f"SELECT COUNT(*) FROM {local}")
            if (await cur.fetchone())[0]:
                # A warm container keeps its watermarks; a remote it has none
                # for is settled at its first push (_settle).
                _seen[local] = await _local_max(conn, local, spec)
            else:
                empty.append(local)
        maxima = await _remote_maxima(rs, empty)

        for local in empty:
            spec = TABLES[local]
            is_int = spec["wm_kind"] == "int"
            cols = ", ".join(spec["cols"])
            rows, source = None, None
            for r in _restore_order(rs, maxima, local):
                try:
                    rows = await _fetch_remote(spec, cols, r)
                    source = r
                    break
                except Exception as e:
                    print(f"[turso_sync] restore {local} from {r.name} failed: {str(e)[:120]}")
            known = [m[local] for m in maxima.values() if m.get(local) is not None]

            if source is None:
                # Nothing restored. An id table can still replicate new rows if
                # some remote said how high its ids go; anything else cannot
                # tell our rows from remote rows (see restore()).
                if not (is_int and maxima):
                    _no_push.add(local)
                    continue
                top = max(known, default=None)
                if top is not None and not await _reserve_above(conn, local, top):
                    _no_push.add(local)
                    continue
                _seen[local] = top
                for name, m in maxima.items():
                    await _settle_known(conn, name, local, spec, m[local], top)
                continue

            if rows:
                ph = ", ".join("?" * len(spec["cols"]))
                await conn.executemany(
                    f"INSERT OR REPLACE INTO {local} ({cols}) VALUES ({ph})", rows)
                restored[local] = len(rows)
                _restored_from[local] = source.name
            hi = await _local_max(conn, local, spec)
            seen = hi
            if is_int:
                top = max(known + ([hi] if hi is not None else []), default=None)
                if top is not None and (hi is None or top > hi):
                    if not await _reserve_above(conn, local, top):
                        _no_push.add(local)
                        continue
                seen = top
            _seen[local] = seen

            # The source now matches local: restored rows must not go back.
            if hi is None:
                await _clear_wm(conn, local, source.name)
            else:
                await _set_wm(conn, local, hi, source.name)
            _ready.setdefault(source.name, set()).add(local)
            for name, m in maxima.items():
                if name != source.name:
                    await _settle_known(conn, name, local, spec, m[local], seen)
        await conn.commit()
    if restored:
        print(f"[turso_sync] restored from Turso: "
              f"{ {t: f'{n} from {_restored_from[t]}' for t, n in restored.items()} }")
    return restored


async def _settle_known(conn, remote: str, local: str, spec: dict, remote_max, seen) -> None:
    """Set `remote`'s watermark for `local` from how far it said it goes.

    Behind or level: push from where it stops, so it gets what it lacks.
    Ahead, on an id table: its extra ids are below our reserved range, so new
    rows can't collide; push from where it stops (it gets only new rows).
    Ahead, on a timestamp table: it holds versions newer than ours, and pushing
    could overwrite them with older ones, so hold the table back from it.
    """
    if spec["wm_kind"] != "int" and _newer(spec, remote_max, seen):
        _held.setdefault(remote, set()).add(local)
        print(f"[turso_sync] {local}: {remote} holds newer rows than this container restored; "
              f"not pushing it there")
        return
    if remote_max is None:
        await _clear_wm(conn, local, remote)
    else:
        await _set_wm(conn, local, remote_max, remote)
    _ready.setdefault(remote, set()).add(local)


async def _settle(conn, r: Remote, local: str, spec: dict) -> bool:
    """First push of `local` to a remote that was not asked at boot. Returns
    whether to push. Raises if the remote can't answer (retried next tick)."""
    if await _get_wm(conn, local, r.name) is not None:
        _ready.setdefault(r.name, set()).add(local)
        return True
    res = await _run(r, [{"sql": f"SELECT MAX({spec['watermark']}) FROM {spec['remote']}"}])
    remote_max = _wm_value(spec, res[0][0][0] if res and res[0] else None)
    seen = _seen.get(local)
    if _newer(spec, remote_max, seen):
        # On an id table too: ids above what we saw may be ones we hand out.
        _held.setdefault(r.name, set()).add(local)
        print(f"[turso_sync] {local}: {r.name} holds rows this container never saw "
              f"({remote_max} > {seen}); not pushing it there")
        return False
    if remote_max is not None:
        await _set_wm(conn, local, remote_max, r.name)
    _ready.setdefault(r.name, set()).add(local)
    return True


async def _push_table(conn, r: Remote, local: str, spec: dict) -> int:
    """Push one batch of rows newer than `r`'s watermark. Returns rows pushed."""
    wm_col, kind = spec["watermark"], spec["wm_kind"]
    wm = await _get_wm(conn, local, r.name)
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
        return 0

    ph = ", ".join("?" * len(spec["cols"]))
    stmts = [{
        "sql": f"INSERT OR REPLACE INTO {spec['remote']} ({cols}) VALUES ({ph})",
        "args": [_cell(v) for v in row],
    } for row in rows]
    await _run(r, stmts, timeout=120)

    wm_idx = spec["cols"].index(wm_col)
    await _set_wm(conn, local, rows[-1][wm_idx], r.name)
    return len(rows)


async def sync_once() -> dict:
    """Push rows newer than the watermark. Returns {table: count}."""
    if turso_remotes.has_mirror():
        return await _sync_mirrored()
    pushed = {}
    primary = Remote("primary", "", "")
    async with aiosqlite.connect(config.DB_PATH) as conn:
        await conn.execute(_LOCAL_STATE_DDL)
        await conn.commit()
        for local, spec in TABLES.items():
            if local in _no_push:
                # Restore failed and the id range could not be reserved, so we
                # cannot tell our rows from rows already on the server.
                continue
            n = await _push_table(conn, primary, local, spec)
            if n:
                pushed[local] = n
        await conn.commit()
    return pushed


async def _sync_mirrored() -> dict:
    """sync_once() with a mirror: each remote in turn, each from its own
    watermark. One remote failing never holds up the other; a primary failure
    is still raised, after the mirror's turn, so the loop reports it."""
    pushed: dict[str, int] = {}
    primary_error: Exception | None = None
    async with aiosqlite.connect(config.DB_PATH) as conn:
        await conn.execute(_LOCAL_STATE_DDL)
        await conn.commit()
        for r in turso_remotes.remotes():
            n_pushed = 0
            try:
                if r.name not in _schema_ok:
                    await ensure_remote_schema(r)
                for local, spec in TABLES.items():
                    if local in _no_push or local in _held.get(r.name, ()):
                        continue
                    if local not in _ready.get(r.name, ()):
                        cur = await conn.execute(f"SELECT 1 FROM {local} LIMIT 1")
                        if await cur.fetchone() is None:
                            continue        # nothing to push yet; settle later
                        if not await _settle(conn, r, local, spec):
                            continue
                    n = await _push_table(conn, r, local, spec)
                    if n:
                        n_pushed += n
                        pushed[local if r.name == "primary" else f"{local}@{r.name}"] = n
            except Exception as e:
                _remote_last[r.name] = {"ok": False, "at": time.time(), "pushed": n_pushed,
                                        "error": str(e)[:200]}
                if r.name == "primary":
                    primary_error = e
                else:
                    print(f"[turso_sync] push to {r.name} failed (will retry): {str(e)[:150]}")
            else:
                _remote_last[r.name] = {"ok": True, "at": time.time(), "pushed": n_pushed,
                                        "error": None}
            # Each watermark was set only after its batch landed, so keeping
            # them is right even when a later batch failed.
            await conn.commit()
    if primary_error is not None:
        raise primary_error
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
        if turso_remotes.has_mirror():
            # Either remote is enough to start: restore takes the one that
            # answers, and the other is caught up when it does.
            reachable = []
            for r in turso_remotes.remotes():
                try:
                    await ensure_remote_schema(r)
                    reachable.append(r.name)
                except Exception as e:
                    print(f"[turso_sync] {r.name} schema check failed: {str(e)[:120]}")
            if not reachable:
                raise RuntimeError("no Turso remote answered")
        else:
            await ensure_remote_schema()
        await restore()
    except Exception as e:
        print(f"[turso_sync] startup failed ({str(e)[:150]}) — continuing without sync")
        return
    _task = asyncio.create_task(_loop())
    names = "+".join(r.name for r in turso_remotes.remotes())
    print(f"[turso_sync] active ({names}), every {SYNC_INTERVAL}s")


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
        # Non-empty means this container is deliberately NOT replicating those
        # tables, because its restore did not complete and pushing would
        # overwrite remote rows it never saw. Writes are still safe locally,
        # but they are not backed up — worth alerting on.
        "not_replicating": sorted(_no_push),
        # Mirror mode only: each remote's last push, and tables held back from
        # it because it holds rows this container never saw. A held table needs
        # a person: scripts/turso_catchup.py compares the two.
        "remotes": {
            r.name: {**_remote_last.get(r.name, {}), "held": sorted(_held.get(r.name, ()))}
            for r in turso_remotes.remotes()
        } if turso_remotes.has_mirror() else {},
        "restored_from": dict(_restored_from),
        "reads": turso_remotes.status(),
    }
