"""
Copying rows between two Turso databases: the daily job's mirror stage
(scripts/daily_refresh.py) and the one-off catch-up (scripts/turso_catchup.py).

The app keeps its user data in both through turso_sync (app/turso_remotes.py).
The daily job writes only to the primary, so this copies what it wrote to the
mirror afterwards. What it copies, and how it finds it without scanning:

  papers             rows above the mirror's highest rowid, rowids kept. The
                     two databases hold the same rows under the same rowids
                     (the second one was seeded with rowids kept), and the
                     daily job only appends, so the tail is exactly what is new.
  citation counts    citation_history rows past a watermark of the primary's
                     rowid, kept in the mirror's `mirror_state`, re-reading a
                     short overlap for rows a same-day re-run updated in place.
                     Each copied row also sets the paper's count.
  hf lists           hf_upvote_history the same way; hf_daily_papers for the
                     papers those rows name.
  licences           the licence rows of the last RECENT_ROWS papers: the ingest
                     writes a paper's licence after its row, and a failed
                     licence write is redone by a later run.

Every write is all-or-nothing per batch (the same conditional Hrana batch as
daily_refresh.tx_batch), and rows the mirror already holds unchanged are not
written: the free plan allows 10M writes a month but 500M reads, so comparing
first is the cheap side.
"""
from __future__ import annotations

import base64

import httpx

BATCH = 200            # rows per all-or-nothing write
KEYS_PER_READ = 100    # keys per IN (...) lookup
RECENT_ROWS = 5000     # papers whose licences are re-checked each run (~5 days)
OVERLAP = 10_000       # history rows re-read below the watermark

STATE_DDL = "CREATE TABLE IF NOT EXISTS mirror_state (key TEXT PRIMARY KEY, value TEXT)"


# ── a small Turso client ─────────────────────────────────────────────────────

def cell(v) -> dict:
    if v is None:
        return {"type": "null"}
    if isinstance(v, bool):
        return {"type": "integer", "value": str(int(v))}
    if isinstance(v, int):
        return {"type": "integer", "value": str(v)}
    if isinstance(v, float):
        return {"type": "float", "value": v}
    if isinstance(v, (bytes, bytearray, memoryview)):
        return {"type": "blob", "base64": base64.b64encode(bytes(v)).decode()}
    return {"type": "text", "value": str(v)}


def uncell(c: dict):
    t = c.get("type")
    if t == "null":
        return None
    if t == "integer":
        return int(c["value"])
    if t == "float":
        return float(c["value"])
    if t == "blob":
        s = c.get("base64") or ""
        return base64.b64decode(s + "=" * (-len(s) % 4))
    return c.get("value")


def tx_batch(stmts: list[dict]) -> dict:
    """Commits only if every statement succeeded (see daily_refresh.tx_batch,
    which this mirrors: a plain BEGIN/COMMIT pipeline keeps partial work)."""
    steps = [{"stmt": {"sql": "BEGIN"}}]
    for s in stmts + [{"sql": "COMMIT"}]:
        steps.append({"stmt": s, "condition": {"type": "ok", "step": len(steps) - 1}})
    steps.append({"stmt": {"sql": "ROLLBACK"},
                  "condition": {"type": "not", "cond": {"type": "ok", "step": len(steps) - 1}}})
    return {"steps": steps}


class Db:
    """One Turso database over HTTP. Values come back as Python types."""

    def __init__(self, name: str, url: str, token: str, http: httpx.Client | None = None):
        self.name = name
        self.url = url.replace("libsql://", "https://").rstrip("/")
        self.token = token
        self.http = http or httpx.Client(timeout=180)

    def _post(self, request: dict) -> dict:
        r = self.http.post(f"{self.url}/v2/pipeline", headers={"Authorization": f"Bearer {self.token}"},
                           json={"requests": [request, {"type": "close"}]})
        r.raise_for_status()
        res = r.json()["results"][0]
        if res.get("type") != "ok":
            raise RuntimeError(f"{self.name}: {str(res.get('error') or res)[:300]}")
        return res["response"]

    def rows(self, sql: str, args: tuple | list = ()) -> list[list]:
        resp = self._post({"type": "execute", "stmt": {"sql": sql, "args": [cell(a) for a in args]}})
        return [[uncell(c) for c in row] for row in resp["result"]["rows"]]

    def tx(self, stmts: list[dict]) -> None:
        if not stmts:
            return
        resp = self._post({"type": "batch", "batch": tx_batch(stmts)})
        errs = [e for e in resp["result"]["step_errors"] if e]
        if errs:
            raise RuntimeError(f"{self.name}: {str(errs[0])[:300]}")


# ── helpers over any Db-like object (rows, tx) ───────────────────────────────

def stmt(sql: str, args: list | tuple = ()) -> dict:
    return {"sql": sql, "args": [cell(a) for a in args]}


def max_rowid(db, table: str) -> int:
    return int(db.rows(f"SELECT MAX(rowid) FROM {table}")[0][0] or 0)


def columns(db, table: str) -> list[str]:
    return [r[1] for r in db.rows(f"PRAGMA table_info({table})")]


def tables(db) -> list[str]:
    return [r[0] for r in db.rows("SELECT name FROM sqlite_master WHERE type = 'table' "
                                  "AND name NOT LIKE 'sqlite_%' ORDER BY name")]


def get_state(db, key: str) -> str | None:
    got = db.rows("SELECT value FROM mirror_state WHERE key = ?", (key,))
    return got[0][0] if got else None


def set_state_stmt(key: str, value) -> dict:
    return stmt("INSERT OR REPLACE INTO mirror_state (key, value) VALUES (?, ?)", (key, str(value)))


def upsert_stmt(table: str, cols: list[str], row: list | tuple) -> dict:
    marks = ", ".join("?" * len(cols))
    return stmt(f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({marks})", row)


def fetch_by_keys(db, table: str, cols: list[str], key_cols: list[str], keys: list[tuple]) -> dict:
    """{key: row} for the rows of `table` whose key is in `keys`."""
    out: dict[tuple, list] = {}
    idx = [cols.index(k) for k in key_cols]
    for i in range(0, len(keys), KEYS_PER_READ):
        part = keys[i:i + KEYS_PER_READ]
        if len(key_cols) == 1:
            where = f"{key_cols[0]} IN ({', '.join('?' * len(part))})"
            args = [k[0] for k in part]
        else:
            one = "(" + ", ".join("?" * len(key_cols)) + ")"
            where = f"({', '.join(key_cols)}) IN (VALUES {', '.join([one] * len(part))})"
            args = [v for k in part for v in k]
        for row in db.rows(f"SELECT {', '.join(cols)} FROM {table} WHERE {where}", args):
            out[tuple(row[j] for j in idx)] = row
    return out


def differing(dst, table: str, cols: list[str], key_cols: list[str], rows: list[list]) -> list[list]:
    """The rows of `rows` that `dst` lacks or holds differently."""
    idx = [cols.index(k) for k in key_cols]
    have = fetch_by_keys(dst, table, cols, key_cols, [tuple(r[j] for j in idx) for r in rows])
    return [r for r in rows if have.get(tuple(r[j] for j in idx)) != list(r)]


def write(dst, stmts: list[dict]) -> None:
    for i in range(0, len(stmts), BATCH):
        dst.tx(stmts[i:i + BATCH])


# ── the daily mirror stage ───────────────────────────────────────────────────

def copy_papers(src, dst) -> dict:
    """Rows above the mirror's highest rowid, rowids kept. Refuses when the two
    no longer line up, since appending would then put rows under wrong rowids."""
    s_max, d_max = max_rowid(src, "papers"), max_rowid(dst, "papers")
    out = {"primary_max_rowid": s_max, "mirror_max_rowid_before": d_max, "copied": 0}
    if d_max > s_max:
        out["problem"] = f"mirror has papers above the primary's last rowid ({d_max} > {s_max})"
        return out
    if d_max:
        a = src.rows("SELECT arxiv_id FROM papers WHERE rowid = ?", (d_max,))
        b = dst.rows("SELECT arxiv_id FROM papers WHERE rowid = ?", (d_max,))
        if a != b:
            out["problem"] = f"papers differ at rowid {d_max}: {a} vs {b}"
            return out
    cols = columns(src, "papers")
    last = d_max
    while True:
        page = src.rows(f"SELECT rowid, {', '.join(cols)} FROM papers WHERE rowid > ? "
                        f"ORDER BY rowid LIMIT {BATCH}", (last,))
        if not page:
            break
        dst.tx([upsert_stmt("papers", ["rowid"] + cols, r) for r in page])
        out["copied"] += len(page)
        last = page[-1][0]
    return out


def copy_licences(src, dst, since_rowid: int) -> dict:
    cols = ["arxiv_id", "licence", "licence_url", "source", "updated_at"]
    rows = src.rows(f"SELECT {', '.join('l.' + c for c in cols)} FROM papers p "
                    "JOIN paper_licences l ON l.arxiv_id = p.arxiv_id WHERE p.rowid > ?", (since_rowid,))
    todo = differing(dst, "paper_licences", cols, ["arxiv_id"], rows)
    write(dst, [upsert_stmt("paper_licences", cols, r) for r in todo])
    return {"checked": len(rows), "copied": len(todo)}


def write_groups(dst, groups: list[list[dict]], tail: dict | None = None) -> None:
    """Write statement groups, never splitting a group across transactions, so a
    row and the statements that go with it land together or not at all. `tail`
    goes in the last transaction."""
    batch: list[dict] = []
    for g in groups:
        if batch and len(batch) + len(g) > BATCH:
            dst.tx(batch)
            batch = []
        batch.extend(g)
    if tail is not None:
        batch.append(tail)
    dst.tx(batch)


def copy_by_watermark(src, dst, table: str, cols: list[str], key_cols: list[str], state_key: str,
                      extra=None, overlap: int = OVERLAP) -> dict:
    """Rows of `table` past the watermark (the primary's rowid), plus `overlap`
    rows below it for in-place updates. `extra(row)` gives statements that must
    land with that row (its paper's citation count)."""
    wm = get_state(dst, state_key)
    if wm is None:
        return {"problem": f"mirror_state has no {state_key}: run scripts/turso_catchup.py first"}
    wm = int(wm)
    start = max(0, wm - overlap)
    out = {"from_rowid": start, "checked": 0, "copied": 0}
    last = start
    page_size = BATCH * 5
    while True:
        page = src.rows(f"SELECT rowid, {', '.join(cols)} FROM {table} WHERE rowid > ? "
                        f"ORDER BY rowid LIMIT {page_size}", (last,))
        if not page:
            break
        last = page[-1][0]
        rows = [r[1:] for r in page]
        todo = differing(dst, table, cols, key_cols, rows)
        groups = [[upsert_stmt(table, cols, r)] + (extra(r) if extra else []) for r in todo]
        # The watermark moves with the last rows it covers.
        write_groups(dst, groups, tail=set_state_stmt(state_key, max(last, wm)))
        out["checked"] += len(rows)
        out["copied"] += len(todo)
        if len(page) < page_size:
            break
    out["to_rowid"] = max(last, wm)
    return out


HISTORY_COLS = ["arxiv_id", "checked_on", "citations_before", "citations",
                "influential_before", "influential"]
UPVOTE_COLS = ["arxiv_id", "checked_on", "upvotes", "comments", "github_stars", "day_rank", "day_size"]
HF_PAPER_COLS = ["arxiv_id", "listed_on", "title", "keywords", "summary", "github_repo",
                 "project_page", "first_seen"]


def _count_update(row: list) -> list[dict]:
    # Rows arrive in rowid order, so a paper's latest day is applied last.
    return [stmt("UPDATE papers SET citation_count = ?, influential_citations = ? WHERE arxiv_id = ?",
                 (row[3], row[5], row[0]))]


def copy_hf_papers(src, dst, arxiv_ids: list[str]) -> int:
    ids = sorted(set(arxiv_ids))
    rows = list(fetch_by_keys(src, "hf_daily_papers", HF_PAPER_COLS, ["arxiv_id"], [(a,) for a in ids]).values())
    todo = differing(dst, "hf_daily_papers", HF_PAPER_COLS, ["arxiv_id"], rows)
    write(dst, [upsert_stmt("hf_daily_papers", HF_PAPER_COLS, r) for r in todo])
    return len(todo)


def mirror_daily(src, dst, ddl: list[str], recent_rows: int = RECENT_ROWS) -> dict:
    """Copy what the daily job wrote to the primary onto the mirror. Returns a
    summary; anything in its "problems" means the mirror is not equal."""
    out: dict = {"problems": []}
    dst.tx([{"sql": d} for d in ddl + [STATE_DDL]])
    out["papers"] = copy_papers(src, dst)
    if "problem" in out["papers"]:
        out["problems"].append(out["papers"]["problem"])
    else:
        since = max(0, out["papers"]["mirror_max_rowid_before"] - recent_rows)
        out["licences"] = copy_licences(src, dst, since)

    out["citations"] = copy_by_watermark(src, dst, "citation_history", HISTORY_COLS,
                                         ["arxiv_id", "checked_on"], "citation_history_rowid",
                                         extra=_count_update)
    copied_ids: list[str] = []

    def note(row):
        copied_ids.append(row[0])
        return []

    out["hf_upvotes"] = copy_by_watermark(src, dst, "hf_upvote_history", UPVOTE_COLS,
                                          ["arxiv_id", "checked_on"], "hf_upvote_history_rowid", extra=note)
    out["hf_papers"] = copy_hf_papers(src, dst, copied_ids) if copied_ids else 0
    for k in ("citations", "hf_upvotes"):
        if "problem" in out[k]:
            out["problems"].append(out[k]["problem"])
    after = max_rowid(dst, "papers")
    if "problem" not in out["papers"] and after != out["papers"]["primary_max_rowid"]:
        out["problems"].append(f"mirror papers end at rowid {after}, primary at "
                               f"{out['papers']['primary_max_rowid']}")
    return out
