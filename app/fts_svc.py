"""
Local BM25 sparse retrieval over the metadata sidecar.

Replaces the Zilliz sparse arm. The motivation is coverage, not speed: Zilliz
indexes only the 1.6M-paper snapshot, while dense retrieval now also covers the
202k papers published since. RRF sums 1/(k+rank) per list, so a paper present in
one list and absent from the other collects half the score of a paper in both —
at k=60 a new paper ranked #1 by dense loses to any old paper appearing above
rank 62 in both lists. The newer papers were effectively invisible.

Indexing the same corpus in both arms removes that asymmetry by construction
rather than correcting for it afterwards.

Being local is a secondary win: sparse search drops from a ~600 ms network round
trip to single-digit milliseconds, and one paid dependency goes away.

Returns the same {'arxiv_id', 'score'} shape as zilliz_svc.search_sparse, so the
fusion step is unchanged.
"""
from __future__ import annotations

import asyncio
import re
import sqlite3

from app import config, local_meta

# Title matches are weighted above abstract matches. bm25() returns a NEGATIVE
# score where more negative is better, so results sort ascending.
_TITLE_WEIGHT = 2.0
_ABSTRACT_WEIGHT = 1.0

_probed = False
_available = False


def _probe() -> bool:
    """Confirm the sidecar carries an FTS index. Cheap, and cached."""
    global _probed, _available
    if _probed:
        return _available
    _probed = True
    if not local_meta.is_available() or local_meta._conn is None:
        print("[fts_svc] no sidecar — sparse search unavailable")
        _available = False
        return False
    try:
        local_meta._conn.execute(
            "SELECT rowid FROM papers_fts LIMIT 1").fetchone()
        _available = True
        print("[fts_svc] FTS5 index available")
    except sqlite3.Error as e:
        # An older sidecar predates the index. Degrade rather than fail: the
        # dense arm alone still returns results.
        print(f"[fts_svc] no FTS index in sidecar ({e}) — sparse search off")
        _available = False
    return _available


def is_available() -> bool:
    return _probe()


def _terms(query: str) -> list[str]:
    """Bare alphanumeric terms.

    User text cannot go into MATCH as-is: FTS5 treats quotes, `*`, `AND`, `OR`,
    `NOT`, `NEAR` and `^` as syntax, so an apostrophe or a stray hyphen raises
    OperationalError and takes the whole sparse arm down. Extracting terms and
    re-quoting them means no user input is ever parsed as an operator.

    Single characters are dropped — they match a large fraction of the corpus
    and contribute nothing to BM25 ranking.
    """
    return [t for t in re.findall(r"[A-Za-z0-9]+", query) if len(t) > 1]


def _search_sync(query: str, limit: int) -> list[dict]:
    conn = local_meta._conn
    if conn is None:
        return []
    terms = _terms(query)
    if not terms:
        return []

    sql = ("SELECT p.arxiv_id, bm25(papers_fts, ?, ?) AS s "
           "FROM papers_fts JOIN papers p ON p.rowid = papers_fts.rowid "
           "WHERE papers_fts MATCH ? ORDER BY s LIMIT ?")

    # Conjunction first: precise and fast, because the term intersection is
    # small. It also returns nothing when one term is absent from the corpus,
    # which for a multi-word query is common — hence the disjunction fallback.
    attempts = [" AND ".join(f'"{t}"' for t in terms)]
    if len(terms) > 1:
        attempts.append(" OR ".join(f'"{t}"' for t in terms))

    for match in attempts:
        try:
            rows = conn.execute(sql, (_TITLE_WEIGHT, _ABSTRACT_WEIGHT,
                                      match, limit)).fetchall()
        except sqlite3.Error as e:
            print(f"[fts_svc] query failed ({e})")
            return []
        if rows:
            return [{"arxiv_id": a, "score": -float(s)} for a, s in rows]
    return []


async def search_sparse(query: str, limit: int = 50) -> list[dict]:
    """BM25 search over title + abstract.

    Takes the query TEXT, unlike zilliz_svc.search_sparse which takes BGE-M3
    lexical weights. Both are lexical retrieval and RRF is rank-based, so the
    fusion step does not care which produced the ranking.

    Scores are negated so that higher is better, matching the dense arm's
    convention. Only the ordering is used by RRF, but a mismatched sign would
    silently invert this arm if anything ever sorts on the value.
    """
    if not _probe():
        return []
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _search_sync, query, limit)
    except Exception as e:
        print(f"[fts_svc] search_sparse error: {e}")
        return []


def stats() -> dict:
    """Diagnostics for the health endpoint."""
    if not _probe() or local_meta._conn is None:
        return {"available": False}
    try:
        n = local_meta._conn.execute(
            "SELECT COUNT(*) FROM papers_fts").fetchone()[0]
        return {"available": True, "indexed": n, "backend": "sqlite-fts5"}
    except sqlite3.Error as e:
        return {"available": False, "error": str(e)[:120]}
