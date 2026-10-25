#!/usr/bin/env python3
"""
Make two Turso databases hold the same data, once, before the app mirrors to
both (app/turso_remotes.py, doc 06 2026-10-25).

The situation it was written for
--------------------------------
On 2026-10-24 the first database ran out of reads and the Space moved to a
second one, seeded from the sidecar with rowids kept. So each holds what the
other lacks:
  * the first ("mirror" here, the old one): every user's data and the paper
    caches up to 24 Oct, licences the daily harvest wrote, Hugging Face lists;
  * the second ("primary", the new one): the papers, citation changes,
    licences and lists the daily job wrote since, and the few user and cache
    rows the Space pushed after the switch, numbered from 1 again.

What it does
------------
  daily-job tables  papers by rowid (each side's tail onto the other, after
                    checking the rowids line up), and the union of
                    paper_licences, citation_history, hf_daily_papers and
                    hf_upvote_history by key; then citation counts from each
                    paper's latest history row, and the mirror's watermarks
                    for the daily copy (scripts/turso_copy.py).
  user data/caches  every table turso_sync replicates, merged into one set
                    written to both. Rows match by natural key (cache_key,
                    arxiv_id, (day, user_id), ...) or, without one, by every
                    column but the id. The side with the higher ids keeps
                    them; the other side's unmatched rows are renumbered above.
                    A key on both sides takes the newer row (its timestamp
                    column), except the daily counters in reader_days and
                    ai_daily, which add up: each side counted part of the day.
                    Profiles and onboarding take the newer updated_at; a
                    user's clusters come whole from the side computed later.
                    Re-running is safe once a run has finished: rows already
                    equal on both sides are left as they are. A run stopped
                    between the two sides of a table can add a split day's
                    counters twice on the next run (reader_days, ai_daily only).

Runbook (1 Nov, or whenever the old database answers reads again)
-----------------------------------------------------------------
  1. python scripts/turso_catchup.py check            both readable? inventory
  2. Stop the Space pushing: set TURSO_SYNC_DISABLED=1 in its variables and
     restart (the shutdown flush pushes its last rows first). Do not run it
     during the daily job (02:30 UTC).
  3. python scripts/turso_catchup.py plan             what would change, both ways
  4. python scripts/turso_catchup.py apply --space-sync-stopped
  5. python scripts/turso_catchup.py verify --full    every table equal
  6. Set TURSO_MIRROR_URL / TURSO_MIRROR_DB_TOKEN (the old database) on the
     Space and in GitHub secrets, remove TURSO_SYNC_DISABLED, restart once.

Credentials: the primary from TURSO_URL / TURSO_DB_TOKEN, the mirror from
<prefix>_URL / <prefix>_DB_TOKEN with --mirror-prefix (default TURSO_MIRROR;
.env.local calls the old one TURSO_OLD). Read cost: every row of the small
tables and of paper_licences on each side, about 1M reads; --full adds every
paper's counts, about 2M a side. Both well inside 500M.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tc = _load("turso_copy")

# Natural keys of the replicated id tables (their UNIQUE constraints in
# turso_sync.REMOTE_DDL; a test keeps the two in step).
NATURAL = {
    "paper_explanations": ["cache_key"], "paper_abstracts": ["arxiv_id"], "paper_sources": ["arxiv_id"],
    "paper_briefings": ["cache_key"], "paper_answers": ["cache_key"], "search_overviews": ["cache_key"],
    "reader_days": ["day", "user_id"], "ai_daily": ["day", "feature", "metric"],
}
# Which of two rows with one key is newer.
NEWER_BY = {
    "paper_explanations": "created_at", "paper_abstracts": "fetched_at", "paper_sources": "fetched_at",
    "paper_briefings": "created_at", "paper_answers": "created_at", "search_overviews": "created_at",
}
# Counters a split day adds up.
SUMMED = {"reader_days": ["pages", "searches", "feed_pages", "opens", "saves", "passes", "asks"],
          "ai_daily": ["n"]}

# Daily-job tables merged by key: (table, key columns).
DAILY_KEYED = [("paper_licences", ["arxiv_id"]), ("citation_history", ["arxiv_id", "checked_on"]),
               ("hf_daily_papers", ["arxiv_id"]), ("hf_upvote_history", ["arxiv_id", "checked_on"])]
PAGE = 5000


def sync_tables():
    from app import turso_sync
    return turso_sync.TABLES, turso_sync.REMOTE_DDL


# ── reading ──────────────────────────────────────────────────────────────────

def all_rows(db, table: str, cols: list[str]) -> list[list]:
    """Every row, in pages by rowid (one unpaged SELECT of a big table times out)."""
    out, last = [], 0
    while True:
        page = db.rows(f"SELECT rowid, {', '.join(cols)} FROM {table} WHERE rowid > ? "
                       f"ORDER BY rowid LIMIT {PAGE}", (last,))
        out.extend(r[1:] for r in page)
        if len(page) < PAGE:
            return out
        last = page[-1][0]


def rows_if_present(db, table: str, cols: list[str], present: set[str]) -> list[list]:
    return all_rows(db, table, cols) if table in present else []


def inventory(db) -> dict[str, int]:
    return {t: tc.max_rowid(db, t) for t in tc.tables(db)}


# ── merging ──────────────────────────────────────────────────────────────────

def _newer(row_a, row_b, idx) -> bool:
    """row_a is newer than row_b by column idx (None is oldest)."""
    a, b = row_a[idx], row_b[idx]
    return a is not None and (b is None or str(a) > str(b))


def merge_id_table(table: str, cols: list[str], a: list[list], b: list[list]) -> list[list]:
    """One set of rows for both sides. `a` is the side whose ids are kept (the
    caller passes the side with the higher ids); `b`'s rows that match none of
    `a`'s are renumbered above. Matching is by natural key when the table has
    one, else by every column but the id, counting duplicates."""
    iid = cols.index("id")
    nat = NATURAL.get(table)
    key_idx = [cols.index(k) for k in nat] if nat else [i for i in range(len(cols)) if i != iid]

    def key(r):
        return tuple(r[i] for i in key_idx)

    merged: dict[int, list] = {r[iid]: list(r) for r in a}
    by_key: dict[tuple, list[int]] = {}
    for r in a:
        by_key.setdefault(key(r), []).append(r[iid])
    next_id = max([r[iid] for r in a] + [r[iid] for r in b] + [0]) + 1
    for r in sorted(b, key=lambda r: r[iid]):
        ids = by_key.get(key(r))
        if ids:
            if not nat:
                ids.pop(0)              # the same row on both sides
                if not ids:
                    by_key.pop(key(r))
                continue
            target = merged[ids[0]]
            if list(r) == target:
                continue                # already the same row on both sides (a re-run)
            if table in SUMMED:
                for c in SUMMED[table]:
                    j = cols.index(c)
                    target[j] = (target[j] or 0) + (r[j] or 0)
                if "owner" in cols:
                    j = cols.index("owner")
                    target[j] = max(target[j] or 0, r[j] or 0)
            elif table in NEWER_BY and _newer(r, target, cols.index(NEWER_BY[table])):
                merged[ids[0]] = [ids[0] if i == iid else v for i, v in enumerate(r)]
            continue
        new = list(r)
        new[iid] = next_id
        next_id += 1
        merged[new[iid]] = new
        if nat:
            by_key[key(new)] = [new[iid]]
    return [merged[i] for i in sorted(merged)]


def merge_keyed(cols: list[str], key_cols: list[str], a: list[list], b: list[list],
                newer_col: str | None, prefer_b: bool = True) -> list[list]:
    """Union by key; on a conflict the newer row by `newer_col`, else `b`'s
    (or `a`'s when prefer_b is False)."""
    idx = [cols.index(k) for k in key_cols]
    out = {tuple(r[i] for i in idx): list(r) for r in a}
    j = cols.index(newer_col) if newer_col else None
    for r in b:
        k = tuple(r[i] for i in idx)
        if k not in out:
            out[k] = list(r)
        elif j is not None:
            if _newer(r, out[k], j) or (r[j] == out[k][j] and prefer_b):
                out[k] = list(r)
        elif prefer_b:
            out[k] = list(r)
    return [out[k] for k in sorted(out, key=lambda k: tuple("" if v is None else str(v) for v in k))]


def merge_clusters(cols: list[str], a: list[list], b: list[list]) -> list[list]:
    """A user's clusters are one recomputation: take them all from the side that
    computed later, never a mix."""
    u, t = cols.index("user_id"), cols.index("computed_at")

    def by_user(rows):
        out: dict[str, list] = {}
        for r in rows:
            out.setdefault(r[u], []).append(list(r))
        return out

    ua, ub = by_user(a), by_user(b)
    out = []
    for user in sorted(set(ua) | set(ub)):
        ra, rb = ua.get(user, []), ub.get(user, [])
        newest = lambda rows: max((str(r[t] or "") for r in rows), default="")  # noqa: E731
        out.extend(rb if newest(rb) >= newest(ra) and rb else ra)
    return out


# ── writing ──────────────────────────────────────────────────────────────────

def plan_side(db_rows: list[list], target: list[list], cols: list[str], key_cols: list[str]):
    """(rows to upsert, keys to delete) to turn db_rows into target."""
    idx = [cols.index(k) for k in key_cols]
    have = {tuple(r[i] for i in idx): list(r) for r in db_rows}
    want = {tuple(r[i] for i in idx): list(r) for r in target}
    upserts = [r for k, r in want.items() if have.get(k) != r]
    deletes = [k for k in have if k not in want]
    return upserts, deletes


def apply_side(db, table: str, cols: list[str], key_cols: list[str], upserts, deletes) -> None:
    """Upserts first, then deletes: a REPLACE can only remove rows that are
    either rewritten later in the same pass or listed for deletion."""
    tc.write(db, [tc.upsert_stmt(table, cols, r) for r in upserts])
    for i in range(0, len(deletes), tc.KEYS_PER_READ):
        part = deletes[i:i + tc.KEYS_PER_READ]
        if len(key_cols) == 1:
            where = f"{key_cols[0]} IN ({', '.join('?' * len(part))})"
            args = [k[0] for k in part]
        else:
            one = "(" + ", ".join("?" * len(key_cols)) + ")"
            where = f"({', '.join(key_cols)}) IN (VALUES {', '.join([one] * len(part))})"
            args = [v for k in part for v in k]
        db.tx([tc.stmt(f"DELETE FROM {table} WHERE {where}", args)])


# ── the passes ───────────────────────────────────────────────────────────────

def user_tables_pass(primary, mirror, write: bool) -> dict:
    tables, _ = sync_tables()
    report = {}
    tp, tm = set(tc.tables(primary)), set(tc.tables(mirror))
    for spec in tables.values():
        table, cols = spec["remote"], spec["cols"]
        a_rows, b_rows = rows_if_present(mirror, table, cols, tm), rows_if_present(primary, table, cols, tp)
        if "id" in cols:
            iid = cols.index("id")
            top_m = max((r[iid] for r in a_rows), default=0)
            top_p = max((r[iid] for r in b_rows), default=0)
            # The side with the higher ids keeps them; ties keep the mirror's,
            # which holds the history in the 2026-10-24 case.
            keep, other = (a_rows, b_rows) if top_m >= top_p else (b_rows, a_rows)
            target, key_cols = merge_id_table(table, cols, keep, other), ["id"]
        elif table == "user_clusters":
            target, key_cols = merge_clusters(cols, a_rows, b_rows), spec["pk"]
        else:
            target, key_cols = merge_keyed(cols, spec["pk"], a_rows, b_rows, spec["watermark"]), spec["pk"]
        sides = {}
        for name, db, rows in (("primary", primary, b_rows), ("mirror", mirror, a_rows)):
            up, dele = plan_side(rows, target, cols, key_cols)
            sides[name] = {"rows": len(rows), "upsert": len(up), "delete": len(dele)}
            if write and (up or dele):
                apply_side(db, table, cols, key_cols, up, dele)
        report[table] = {"merged": len(target), **sides}
    return report


def daily_tables_pass(primary, mirror, write: bool) -> dict:
    report: dict = {}
    # papers: each side's tail onto the other, rowids kept
    p_max, m_max = tc.max_rowid(primary, "papers"), tc.max_rowid(mirror, "papers")
    low = min(p_max, m_max)
    probe = sorted({1, low, max(1, low // 2), max(1, low // 3), max(1, low - 1000)} - {0})
    pa = {r: primary.rows("SELECT arxiv_id FROM papers WHERE rowid = ?", (r,)) for r in probe}
    ma = {r: mirror.rows("SELECT arxiv_id FROM papers WHERE rowid = ?", (r,)) for r in probe}
    mismatch = [r for r in probe if pa[r] != ma[r]]
    report["papers"] = {"primary_max_rowid": p_max, "mirror_max_rowid": m_max, "probed": probe,
                        "mismatched_rowids": mismatch}
    if mismatch:
        report["papers"]["problem"] = "rowids do not line up; papers not copied"
    elif write:
        if p_max > m_max:
            report["papers"]["copied_to_mirror"] = tc.copy_papers(primary, mirror)["copied"]
        elif m_max > p_max:
            report["papers"]["copied_to_primary"] = tc.copy_papers(mirror, primary)["copied"]

    tp, tm = set(tc.tables(primary)), set(tc.tables(mirror))
    for table, key_cols in DAILY_KEYED:
        cols = tc.columns(primary, table) or tc.columns(mirror, table)
        if not cols:
            continue
        a_rows, b_rows = rows_if_present(mirror, table, cols, tm), rows_if_present(primary, table, cols, tp)
        newer = "updated_at" if table == "paper_licences" else None
        target = merge_keyed(cols, key_cols, a_rows, b_rows, newer)
        sides = {}
        for name, db, rows in (("primary", primary, b_rows), ("mirror", mirror, a_rows)):
            up, _ = plan_side(rows, target, cols, key_cols)
            sides[name] = {"rows": len(rows), "upsert": len(up)}
            if write and up:
                tc.write(db, [tc.upsert_stmt(table, cols, r) for r in up])
                if table == "citation_history":
                    _counts_from_history(db, up, target, cols)
        report[table] = {"merged": len(target), **sides}

    if write:
        # Everything up to the primary's current rows is now on the mirror.
        mirror.tx([{"sql": tc.STATE_DDL},
                   tc.set_state_stmt("citation_history_rowid", tc.max_rowid(primary, "citation_history")),
                   tc.set_state_stmt("hf_upvote_history_rowid", tc.max_rowid(primary, "hf_upvote_history"))])
    return report


def _counts_from_history(db, written: list[list], history: list[list], cols: list[str]) -> None:
    """For each paper whose history this side just received, its count from the
    latest day in the merged history."""
    a, d = cols.index("arxiv_id"), cols.index("checked_on")
    c, f = cols.index("citations"), cols.index("influential")
    touched = {r[a] for r in written}
    latest: dict[str, list] = {}
    for r in history:
        if r[a] in touched and (r[a] not in latest or r[d] > latest[r[a]][d]):
            latest[r[a]] = r
    tc.write(db, [tc.stmt("UPDATE papers SET citation_count = ?, influential_citations = ? WHERE arxiv_id = ?",
                          (r[c], r[f], r[a])) for r in latest.values()])


def verify(primary, mirror, full: bool) -> dict:
    """Tables and rows on each side; with --full, every paper's counts."""
    tables, _ = sync_tables()
    out: dict = {"only_on_primary": [], "only_on_mirror": [], "different": {}}
    tp, tm = set(tc.tables(primary)), set(tc.tables(mirror))
    out["only_on_primary"], out["only_on_mirror"] = sorted(tp - tm), sorted(tm - tp)
    checks = [(s["remote"], s["cols"]) for s in tables.values()]
    checks += [(t, tc.columns(primary, t)) for t, _ in DAILY_KEYED if t in tp & tm]
    for table, cols in checks:
        a = sorted(map(tuple, all_rows(mirror, table, cols)), key=repr)
        b = sorted(map(tuple, all_rows(primary, table, cols)), key=repr)
        if a != b:
            out["different"][table] = {"primary": len(b), "mirror": len(a)}
    p_max, m_max = tc.max_rowid(primary, "papers"), tc.max_rowid(mirror, "papers")
    if p_max != m_max:
        out["different"]["papers"] = {"primary_max_rowid": p_max, "mirror_max_rowid": m_max}
    elif full:
        cols = ["arxiv_id", "citation_count", "influential_citations"]
        a, b = all_rows(mirror, "papers", cols), all_rows(primary, "papers", cols)
        diff = [x[0] for x, y in zip(a, b) if x != y]
        if diff or len(a) != len(b):
            out["different"]["papers"] = {"rows_that_differ": len(diff), "first": diff[:10]}
    out["equal"] = not out["different"]
    return out


def ensure_schema(db) -> None:
    """Every table either side may be written, with the columns added since."""
    from app import turso_sync
    dr, ib = _load("daily_refresh"), _load("ingest_backends")
    db.tx([{"sql": s} for s in turso_sync.REMOTE_DDL
           + [dr.HISTORY_DDL, dr.HF_PAPERS_DDL, dr.HF_HISTORY_DDL, ib.LICENCE_DDL, tc.STATE_DDL]])
    for table, column, kind in turso_sync.REMOTE_COLUMNS:
        try:
            db.tx([{"sql": f"ALTER TABLE {table} ADD COLUMN {column} {kind}"}])
        except RuntimeError as e:
            if "duplicate column" not in str(e).lower():
                raise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "plan", "apply", "verify"])
    ap.add_argument("--primary-prefix", default="TURSO")
    ap.add_argument("--mirror-prefix", default="TURSO_MIRROR")
    ap.add_argument("--space-sync-stopped", action="store_true",
                    help="confirm the Space is not pushing (TURSO_SYNC_DISABLED=1 and restarted)")
    ap.add_argument("--full", action="store_true", help="verify: compare every paper's counts too")
    args = ap.parse_args()

    from app import config  # noqa: F401  (loads .env.local, as the app does)

    def db(name, prefix):
        url, token = os.getenv(f"{prefix}_URL", ""), os.getenv(f"{prefix}_DB_TOKEN", "")
        if not (url and token):
            raise SystemExit(f"{prefix}_URL / {prefix}_DB_TOKEN not set")
        return tc.Db(name, url, token)

    primary, mirror = db("primary", args.primary_prefix), db("mirror", args.mirror_prefix)
    if primary.url == mirror.url:
        raise SystemExit("primary and mirror are the same database")

    if args.command == "check":
        out = {}
        for d in (primary, mirror):
            try:
                out[d.name] = {"url": d.url.split("//")[-1].split(".")[0], "tables": inventory(d)}
            except RuntimeError as e:
                out[d.name] = {"error": str(e)[:200]}
        print(json.dumps(out, indent=1))
        return 0 if all("error" not in v for v in out.values()) else 1

    if args.command == "verify":
        out = verify(primary, mirror, args.full)
        print(json.dumps(out, indent=1))
        return 0 if out["equal"] else 1

    write = args.command == "apply"
    if write and not args.space_sync_stopped:
        raise SystemExit("apply rewrites user tables on both sides: stop the Space pushing first "
                         "(TURSO_SYNC_DISABLED=1, restart), then pass --space-sync-stopped")
    if write:
        ensure_schema(primary)
        ensure_schema(mirror)
    out = {"mode": args.command,
           "daily": daily_tables_pass(primary, mirror, write),
           "user": user_tables_pass(primary, mirror, write)}
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
