"""
Two Turso databases holding the same data: app/turso_remotes.py, and the mirror
half of app/turso_sync.py.

Each remote here is a real SQLite database built from turso_sync.REMOTE_DDL, and
the two HTTP boundaries (_execute for the primary, _post for the mirror) run the
statements against it. Everything above them is the real code: restore's choice
of source, the per-remote watermarks, the id reservation, the hold, failover.

What must hold:
  * after boot and a sync, both remotes have every row, whichever was ahead;
  * a remote that is down catches up when it is back, and never blocks the other;
  * no push can land on a row a remote holds that this container never saw;
  * without a mirror nothing changes (no extra queries at all).
"""
import asyncio
import sqlite3
import time

import aiosqlite
import pytest

from app import config, turso_remotes, turso_sync
from app.turso_remotes import TursoError


class FakeRemote:
    def __init__(self, name: str):
        self.name = name
        self.db = sqlite3.connect(":memory:")
        for ddl in turso_sync.REMOTE_DDL:
            self.db.execute(ddl)
        self.down = False           # every request fails, like a network outage
        self.reads_blocked = False  # SELECTs fail, writes work: the 2026-10-24 block
        self.statements: list[str] = []

    def run(self, stmts):
        if self.down:
            raise TursoError("HTTP 503: unavailable", outage=True)
        out = []
        for s in stmts:
            sql = s["sql"]
            self.statements.append(sql)
            if self.reads_blocked and sql.lstrip().upper().startswith("SELECT"):
                raise TursoError("{'message': 'Operation was blocked: SQL read operations are "
                                 "forbidden', 'code': 'BLOCKED'}", outage=True)
            args = [turso_sync._uncell(a) for a in s.get("args", [])]
            try:
                cur = self.db.execute(sql, args)
            except sqlite3.Error as e:
                raise TursoError(f"SQLITE_ERROR: {e}", outage=False) from None
            out.append([list(r) for r in cur.fetchall()])
        self.db.commit()
        return out

    def rows(self, table: str, order: str = "1"):
        return self.db.execute(f"SELECT * FROM {table} ORDER BY {order}").fetchall()


@pytest.fixture
def remotes(monkeypatch):
    primary, mirror = FakeRemote("primary"), FakeRemote("mirror")
    monkeypatch.setattr(config, "TURSO_URL", "https://primary.example.turso.io")
    monkeypatch.setattr(config, "TURSO_DB_TOKEN", "p")
    monkeypatch.setattr(config, "TURSO_MIRROR_URL", "libsql://mirror.example.turso.io")
    monkeypatch.setattr(config, "TURSO_MIRROR_DB_TOKEN", "m")
    monkeypatch.delenv("TURSO_SYNC_DISABLED", raising=False)

    async def execute(stmts, timeout=60):
        return primary.run(stmts)

    async def post(base, token, stmts, timeout):
        assert base == "https://mirror.example.turso.io" and token == "m"
        return mirror.run(stmts)

    monkeypatch.setattr(turso_sync, "_execute", execute)
    monkeypatch.setattr(turso_sync, "_post", post)
    turso_sync._schema_ok.clear()
    turso_sync._remote_last.clear()
    turso_remotes._benched_until.clear()
    yield primary, mirror
    turso_remotes._benched_until.clear()
    for s in (turso_sync._no_push, turso_sync._seen, turso_sync._ready, turso_sync._held,
              turso_sync._restored_from, turso_sync._schema_ok, turso_sync._remote_last):
        s.clear()


@pytest.fixture
def local_db(tmp_path, monkeypatch):
    import app.db as db_mod
    path = str(tmp_path / "local.db")
    monkeypatch.setattr(config, "DB_PATH", path)
    monkeypatch.setattr(db_mod, "DB_PATH", path)
    asyncio.run(db_mod.init_db())
    return path


def _interaction(i: int, user: str = "u1", paper: str | None = None):
    return (i, user, paper or f"2610.{i:05d}", "save", "feed", 0, f"q{i}", "v", "dense",
            None, 1.0, "p", f"2026-10-2{i % 10}T00:00:00", None)


def _seed_interactions(remote: FakeRemote, ids):
    remote.db.executemany(
        "INSERT INTO user_interactions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [_interaction(i) for i in ids])
    remote.db.commit()


async def _save(local_db, paper: str, user: str = "u2") -> int:
    async with aiosqlite.connect(local_db) as conn:
        cur = await conn.execute(
            "INSERT INTO interactions (user_id, paper_id, event_type, timestamp) "
            "VALUES (?, ?, 'save', '2026-11-01T10:00:00')", (user, paper))
        await conn.commit()
        return cur.lastrowid


async def _start(monkeypatch):
    """start() without the background loop: schema on both, then restore."""
    created = []
    monkeypatch.setattr(asyncio, "create_task", lambda coro: created.append(coro.close()))
    await turso_sync.start()


def _ids(remote: FakeRemote, table: str = "user_interactions"):
    return [r[0] for r in remote.db.execute(f"SELECT id FROM {table} ORDER BY id")]


# ── boot ─────────────────────────────────────────────────────────────────────

async def test_boot_restores_from_the_remote_that_is_ahead_and_catches_up_the_other(
        remotes, local_db, monkeypatch):
    primary, mirror = remotes
    _seed_interactions(primary, [1, 2, 3])
    _seed_interactions(mirror, [1, 2, 3, 4, 5])

    await _start(monkeypatch)
    assert turso_sync.status()["restored_from"]["interactions"] == "mirror"

    await turso_sync.sync_once()
    assert _ids(primary) == _ids(mirror) == [1, 2, 3, 4, 5]
    assert primary.rows("user_interactions", "id") == mirror.rows("user_interactions", "id")


async def test_a_tie_restores_from_the_primary(remotes, local_db, monkeypatch):
    primary, mirror = remotes
    _seed_interactions(primary, [1, 2])
    _seed_interactions(mirror, [1, 2])
    await _start(monkeypatch)
    assert turso_sync.status()["restored_from"]["interactions"] == "primary"


async def test_restored_rows_are_not_pushed_back_to_a_remote_that_has_them(
        remotes, local_db, monkeypatch):
    primary, mirror = remotes
    _seed_interactions(primary, [1, 2, 3])
    _seed_interactions(mirror, [1, 2, 3])
    await _start(monkeypatch)
    primary.statements.clear(), mirror.statements.clear()
    assert await turso_sync.sync_once() == {}
    assert not any("INSERT" in s for s in primary.statements + mirror.statements)


async def test_a_new_save_reaches_both(remotes, local_db, monkeypatch):
    primary, mirror = remotes
    _seed_interactions(primary, [1, 2])
    _seed_interactions(mirror, [1, 2])
    await _start(monkeypatch)
    new_id = await _save(local_db, "1706.03762")
    pushed = await turso_sync.sync_once()
    assert pushed == {"interactions": 1, "interactions@mirror": 1}
    assert _ids(primary) == _ids(mirror) == [1, 2, new_id]


async def test_profiles_come_from_the_remote_with_the_newer_version(remotes, local_db, monkeypatch):
    primary, mirror = remotes
    primary.db.execute("INSERT INTO user_profiles VALUES ('u1','longterm',x'00',3,'2026-10-20T00:00:00')")
    mirror.db.execute("INSERT INTO user_profiles VALUES ('u1','longterm',x'01',4,'2026-10-24T00:00:00')")
    await _start(monkeypatch)
    await turso_sync.sync_once()
    async with aiosqlite.connect(local_db) as conn:
        cur = await conn.execute("SELECT interaction_count FROM user_profiles")
        assert (await cur.fetchone())[0] == 4
    # The primary was behind; it now has the newer version too.
    assert primary.rows("user_profiles") == mirror.rows("user_profiles")


# ── one remote down ──────────────────────────────────────────────────────────

async def test_primary_down_at_boot_restores_from_the_mirror_then_catches_up(
        remotes, local_db, monkeypatch):
    primary, mirror = remotes
    _seed_interactions(primary, [1, 2, 3])
    _seed_interactions(mirror, [1, 2, 3])
    primary.down = True

    await _start(monkeypatch)
    assert turso_sync.status()["restored_from"]["interactions"] == "mirror"
    a = await _save(local_db, "2610.11111")

    with pytest.raises(TursoError):        # the primary's failure is still reported
        await turso_sync.sync_once()
    assert _ids(mirror) == [1, 2, 3, a]     # the mirror was not held up
    assert turso_sync.status()["remotes"]["primary"]["ok"] is False

    primary.down = False
    b = await _save(local_db, "2610.22222")
    await turso_sync.sync_once()
    assert _ids(primary) == _ids(mirror) == [1, 2, 3, a, b]


async def test_mirror_down_mid_session_catches_up_when_back(remotes, local_db, monkeypatch):
    primary, mirror = remotes
    await _start(monkeypatch)
    mirror.down = True
    ids = [await _save(local_db, f"2610.0000{i}") for i in range(3)]
    assert await turso_sync.sync_once() == {"interactions": 3}   # no raise for a mirror failure
    assert _ids(mirror) == []
    mirror.down = False
    await turso_sync.sync_once()
    assert _ids(primary) == _ids(mirror) == ids


async def test_reads_blocked_on_the_primary_still_boots_from_the_mirror(remotes, local_db, monkeypatch):
    """The 2026-10-24 case: reads refused, writes accepted."""
    primary, mirror = remotes
    _seed_interactions(primary, [1, 2])
    _seed_interactions(mirror, [1, 2])
    primary.reads_blocked = True
    await _start(monkeypatch)
    async with aiosqlite.connect(local_db) as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM interactions")
        assert (await cur.fetchone())[0] == 2


# ── never overwrite what this container never saw ────────────────────────────

async def test_new_ids_start_above_every_remote_even_one_that_could_not_restore(
        remotes, local_db, monkeypatch):
    primary, mirror = remotes
    _seed_interactions(primary, [1, 2, 3])
    _seed_interactions(mirror, [1, 2, 3, 4, 5])
    real_fetch = turso_sync._fetch_remote

    async def fetch(spec, cols, remote=None):
        if remote is not None and remote.name == "mirror" and spec["remote"] == "user_interactions":
            raise TursoError("timeout", outage=True)
        return await real_fetch(spec, cols, remote)

    monkeypatch.setattr(turso_sync, "_fetch_remote", fetch)
    await _start(monkeypatch)
    assert turso_sync.status()["restored_from"]["interactions"] == "primary"
    new_id = await _save(local_db, "2610.33333")
    assert new_id > 5, "a new id at 4 or 5 would overwrite the mirror's rows"

    before = mirror.rows("user_interactions", "id")
    await turso_sync.sync_once()
    assert mirror.rows("user_interactions", "id")[:5] == before
    assert _ids(primary) == [1, 2, 3, new_id]


async def test_a_remote_unreachable_at_boot_that_holds_unseen_rows_is_held(
        remotes, local_db, monkeypatch):
    primary, mirror = remotes
    _seed_interactions(primary, [1, 2])
    _seed_interactions(mirror, [1, 2, 3, 4])     # rows the primary never got
    mirror.down = True
    await _start(monkeypatch)                     # restores 1, 2 from the primary
    new_id = await _save(local_db, "2610.44444")
    assert new_id == 3, "only the primary's ids were known at boot"

    mirror.down = False
    await turso_sync.sync_once()
    assert turso_sync.status()["remotes"]["mirror"]["held"] == ["interactions"]
    assert mirror.rows("user_interactions", "id")[2][2] == "2610.00003", "mirror row 3 overwritten"
    assert _ids(primary) == [1, 2, 3]


async def test_a_held_table_does_not_stop_other_tables(remotes, local_db, monkeypatch):
    primary, mirror = remotes
    _seed_interactions(mirror, [1])
    mirror.down = True
    await _start(monkeypatch)
    await _save(local_db, "2610.55555")
    async with aiosqlite.connect(local_db) as conn:
        await conn.execute("INSERT INTO user_onboarding (user_id, selected_categories, onboarding_completed, "
                           "created_at, updated_at) VALUES ('u9', '[]', 1, 't', 't')")
        await conn.commit()
    mirror.down = False
    await turso_sync.sync_once()
    assert turso_sync.status()["remotes"]["mirror"]["held"] == ["interactions"]
    assert [r[0] for r in mirror.rows("user_onboarding")] == ["u9"]


# ── without a mirror ─────────────────────────────────────────────────────────

async def test_without_a_mirror_restore_and_sync_send_nothing_new(local_db, monkeypatch):
    monkeypatch.setattr(config, "TURSO_URL", "https://primary.example.turso.io")
    monkeypatch.setattr(config, "TURSO_DB_TOKEN", "p")
    monkeypatch.setattr(config, "TURSO_MIRROR_URL", "")
    monkeypatch.setattr(config, "TURSO_MIRROR_DB_TOKEN", "")
    primary = FakeRemote("primary")
    _seed_interactions(primary, [1, 2])

    async def execute(stmts, timeout=60):
        return primary.run(stmts)

    async def post(*a, **k):
        raise AssertionError("no second database without a mirror")

    monkeypatch.setattr(turso_sync, "_execute", execute)
    monkeypatch.setattr(turso_sync, "_post", post)
    await turso_sync.restore()
    await _save(local_db, "2610.66666")
    await turso_sync.sync_once()
    assert not any("MAX(" in s for s in primary.statements), "single-database boot changed"
    assert _ids(primary) == [1, 2, 3]
    assert turso_sync.status()["remotes"] == {}


# ── reads ────────────────────────────────────────────────────────────────────

async def test_a_blocked_primary_moves_reads_to_the_mirror_and_they_stay_there(remotes):
    primary, mirror = remotes
    primary.db.execute("CREATE TABLE papers (arxiv_id TEXT)")
    mirror.db.execute("CREATE TABLE papers (arxiv_id TEXT)")
    mirror.db.execute("INSERT INTO papers VALUES ('2610.00001')")
    primary.reads_blocked = True
    stmt = [{"sql": "SELECT arxiv_id FROM papers"}]

    assert await turso_sync.read(stmt) == [[["2610.00001"]]]
    assert turso_remotes.read_order()[0].name == "mirror"
    primary.statements.clear()
    await turso_sync.read(stmt)
    assert primary.statements == [], "a benched primary must not be asked first"


async def test_reads_return_to_the_primary_after_the_bench(remotes, monkeypatch):
    primary, mirror = remotes
    turso_remotes.bench(turso_remotes.remotes()[0], "BLOCKED")
    assert turso_remotes.read_order()[0].name == "mirror"
    later = time.monotonic() + config.TURSO_FAILOVER_S + 5
    monkeypatch.setattr(time, "monotonic", lambda: later)
    assert turso_remotes.read_order()[0].name == "primary"


async def test_a_bad_query_is_not_an_outage(remotes):
    primary, mirror = remotes
    with pytest.raises(TursoError):
        await turso_sync.read([{"sql": "SELECT nope FROM no_such_table"}])
    assert not any("no_such_table" in r for r in mirror.statements), "a SQL error moved to the mirror"
    assert turso_remotes.read_order()[0].name == "primary"


async def test_without_a_mirror_bench_does_nothing(monkeypatch):
    monkeypatch.setattr(config, "TURSO_URL", "https://primary.example.turso.io")
    monkeypatch.setattr(config, "TURSO_DB_TOKEN", "p")
    monkeypatch.setattr(config, "TURSO_MIRROR_URL", "")
    turso_remotes._benched_until.clear()
    turso_remotes.bench(turso_remotes.remotes()[0], "BLOCKED")
    assert turso_remotes._benched_until == {}
    assert [r.name for r in turso_remotes.read_order()] == ["primary"]


async def test_write_all_reaches_both_and_tolerates_a_down_mirror(remotes):
    primary, mirror = remotes
    sql = [{"sql": "UPDATE search_log SET query = NULL WHERE created_at < '2026-01-01'"}]
    await turso_sync.write_all(sql)
    assert sql[0]["sql"] in primary.statements and sql[0]["sql"] in mirror.statements
    mirror.down = True
    await turso_sync.write_all(sql)                # no raise
    primary.down = True
    with pytest.raises(TursoError):
        await turso_sync.write_all(sql)


@pytest.mark.parametrize("error,outage", [
    ({"message": "Operation was blocked: SQL read operations are forbidden", "code": "BLOCKED"}, True),
    ({"message": "SQL write operations are forbidden", "code": ""}, True),
    ({"message": "no such table: papers", "code": "SQLITE_ERROR"}, False),
    (None, False),
])
def test_outage_errors_are_told_apart_from_sql_errors(error, outage):
    assert turso_remotes.is_outage_error(error) is outage


@pytest.mark.parametrize("status,outage", [(500, True), (503, True), (401, True), (429, True),
                                           (400, False), (404, False)])
def test_outage_statuses(status, outage):
    assert turso_remotes.is_outage_status(status) is outage


def test_mirror_url_needs_the_primary_and_is_normalised(monkeypatch):
    monkeypatch.setattr(config, "TURSO_URL", "")
    monkeypatch.setattr(config, "TURSO_DB_TOKEN", "")
    monkeypatch.setattr(config, "TURSO_MIRROR_URL", "libsql://m.turso.io/")
    monkeypatch.setattr(config, "TURSO_MIRROR_DB_TOKEN", "m")
    assert turso_remotes.remotes() == []
    monkeypatch.setattr(config, "TURSO_URL", "libsql://p.turso.io")
    monkeypatch.setattr(config, "TURSO_DB_TOKEN", "p")
    assert [(r.name, r.url) for r in turso_remotes.remotes()] == [
        ("primary", "https://p.turso.io"), ("mirror", "https://m.turso.io")]


# ── the app's own reads ──────────────────────────────────────────────────────

def _turso_http(monkeypatch, answers: dict):
    """Patch the shared HTTP client: answers[host] is a callable(json) -> Response."""
    import httpx
    from app import http_client
    seen = []

    class Client:
        async def post(self, url, json=None, **kw):
            host = httpx.URL(url).host
            seen.append(host)
            return answers[host](url, json)

    monkeypatch.setattr(http_client, "get_client", lambda: Client())
    return seen


def _ok(rows, cols):
    import httpx

    def answer(url, body):
        return httpx.Response(200, request=httpx.Request("POST", url), json={"results": [
            {"type": "ok", "response": {"type": "execute", "result": {
                "cols": [{"name": c} for c in cols],
                "rows": [[{"type": "text", "value": v} if v is not None else {"type": "null"} for v in r]
                         for r in rows]}}},
            {"type": "ok"}]})
    return answer


def _blocked(url, body):
    import httpx
    return httpx.Response(200, request=httpx.Request("POST", url), json={"results": [
        {"type": "error", "error": {"message": "Operation was blocked: SQL read operations are forbidden",
                                    "code": "BLOCKED"}}, {"type": "ok"}]})


META_COLS = ["arxiv_id", "title", "authors", "categories", "primary_topic", "update_date",
             "abstract_preview", "citation_count", "influential_citations"]


async def test_paper_details_come_from_the_mirror_while_the_primary_is_blocked(remotes, monkeypatch):
    from app import local_meta, turso_svc
    monkeypatch.setattr(local_meta, "is_available", lambda: False)
    turso_svc._METADATA_CACHE.clear()
    row = ["2607.24653", "Kimi K3", "A; B", "cs.CL", "AI/ML", "2026-07-30", "abstract", "5", "1"]
    seen = _turso_http(monkeypatch, {"primary.example.turso.io": _blocked,
                                     "mirror.example.turso.io": _ok([row], META_COLS)})
    got = await turso_svc.fetch_metadata_batch(["2607.24653"])
    assert got["2607.24653"]["title"] == "Kimi K3"
    assert seen == ["primary.example.turso.io", "mirror.example.turso.io"]

    turso_svc._METADATA_CACHE.clear()
    seen.clear()
    await turso_svc.fetch_metadata_batch(["2607.24653"])
    assert seen == ["mirror.example.turso.io"], "the blocked primary was asked again"
    turso_svc._METADATA_CACHE.clear()


async def test_paper_details_without_a_mirror_are_one_request_as_before(monkeypatch):
    from app import local_meta, turso_svc
    monkeypatch.setattr(config, "TURSO_URL", "https://primary.example.turso.io")
    monkeypatch.setattr(config, "TURSO_DB_TOKEN", "p")
    monkeypatch.setattr(config, "TURSO_MIRROR_URL", "")
    monkeypatch.setattr(local_meta, "is_available", lambda: False)
    turso_svc._METADATA_CACHE.clear()
    seen = _turso_http(monkeypatch, {"primary.example.turso.io": _blocked})
    assert await turso_svc.fetch_metadata_batch(["2607.24653"]) == {}
    assert seen == ["primary.example.turso.io"]


async def test_daily_papers_history_fails_over_too(remotes, monkeypatch):
    from app import turso_svc
    cols = ["arxiv_id", "listed_on", "title", "keywords", "summary", "github_repo", "upvotes", "github_stars"]
    row = ["2606.18208", "2026-06-20", "Looped World Models", "[]", "", None, "466", "0"]
    _turso_http(monkeypatch, {"primary.example.turso.io": _blocked,
                              "mirror.example.turso.io": _ok([row], cols)})
    got = await turso_svc.fetch_hf_daily("2026-06-01")
    assert [(p["arxiv_id"], p["upvotes"]) for p in got] == [("2606.18208", 466)]


async def test_deep_health_pings_both_and_says_where_reads_go(remotes, monkeypatch):
    import httpx
    from fastapi.testclient import TestClient
    from app.main import app
    real_post = httpx.AsyncClient.post

    async def post(self, url, *a, **kw):
        host = httpx.URL(str(url)).host
        if host == "primary.example.turso.io":
            return _blocked(url, None)
        if host == "mirror.example.turso.io":
            return _ok([["1"]], ["1"])(url, None)
        return await real_post(self, url, *a, **kw)

    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    turso_remotes.bench(turso_remotes.remotes()[0], "BLOCKED")
    with TestClient(app) as client:
        services = client.get("/healthz/deep").json()["services"]
    assert services["turso"]["status"] == "error" and "BLOCKED" in services["turso"]["error"]
    assert services["turso_mirror"]["status"] == "ok"
    assert services["turso"]["reads"]["read_order"] == ["mirror", "primary"]
