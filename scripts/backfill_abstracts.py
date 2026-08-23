"""
Repair the truncated abstracts in the Turso corpus.

THE PROBLEM
-----------
Measured against live Turso on 2026-08-21:

    total papers               1,799,348
    abstracts at the 500 cap   1,630,273   (90.6%)
    abstracts missing            0
    titles missing               0
    citation_count NULL          0

So the corpus is otherwise clean, and this is its one real defect — but it is a
big one, because the truncation degrades four things at once:

  * the card and the paper page show a stump, often cut mid-word;
  * `groq_svc.explain_paper` summarises that stump, so the explanation is built
    from half a paper;
  * `readability` scores the stump, so the density badge measures a fragment;
  * the cross-encoder in the search path reranks on it (PHASE7 §3.2).

WHY THIS IS A SCRIPT AND NOT A BACKGROUND JOB
---------------------------------------------
Repairing all 1.63M rows is a bulk mutation of the production metadata store
and a multi-hour run against a third party's rate-limited API. That is a
decision for a person, not something a request handler should start. So this is
explicit, resumable, and dry-run by default.

It is also ordered by usefulness rather than by id: `--strategy served` repairs
the papers the product actually shows first, so the first hour of running
delivers most of the user-visible benefit. Paper popularity is Zipfian, so a
few thousand rows cover a large share of what anyone ever reads.

USAGE
-----
    # See the scale and the estimate, change nothing:
    .venv/bin/python scripts/backfill_abstracts.py --dry-run

    # Repair the 500 most-cited truncated papers:
    .venv/bin/python scripts/backfill_abstracts.py --limit 500 --apply

    # Repair everything (long; resumable — just re-run it):
    .venv/bin/python scripts/backfill_abstracts.py --all --apply
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time

sys.path.insert(0, ".")

import xml.etree.ElementTree as ET  # noqa: E402

from app import arxiv_svc, config, http_client  # noqa: E402

# arXiv asks for no more than ONE request every three seconds for bulk use.
#
# The first run violated that badly and got 429'd from batch 11 onward, losing
# 9 of 20 batches. The cause was reusing `arxiv_svc.fetch_metadata_batch`, which
# is tuned for small interactive lookups: it splits any input into 20-id
# sub-requests fired at ~3/s. So each "batch of 100" was really a burst of five
# requests, and the 3s pause sat between bursts rather than between requests.
#
# This now issues exactly one HTTP request per batch and sleeps between them, so
# the pause means what it says. A larger id_list per request is also what makes
# the full sweep tractable: 200 ids per request is 7,183 requests, where 20 ids
# would be 71,829.
BATCH = 200
PAUSE_S = 3.0
MAX_RETRIES = 4

# Measured over real batches of 200 on 2026-08-24: 400 rows in 15s,
# 200 rows in 8s. Used for the estimate only; nothing paces off it.
SECONDS_PER_BATCH = 7.75

# Below this an abstract is not really longer than what we already hold, so
# rewriting the row buys nothing.
MIN_GAIN_CHARS = 40

# What "truncated" actually means: length EXACTLY at the cap.
#
# The first version of this used `>= 500`, which is wrong in both directions
# that matter. It counted the 193,689 rows whose abstracts are legitimately
# longer than 500 characters as damaged — overstating the job by 13% — and,
# worse, a REPAIRED row still matches `>= 500`, so the script would have
# re-selected and re-fetched its own completed work forever. It was not
# resumable, despite a test asserting that it was.
TRUNCATED = "length(abstract_preview) = 500"


async def _pipeline(stmts: list[dict], timeout: int = 120) -> list:
    """Run several statements in ONE round trip.

    The write path originally issued one UPDATE per row. At ~150ms per round
    trip that is not a detail: a full sweep of 1.63M rows would spend ~68 hours
    on HTTP alone, dwarfing the 13.6 hours of arXiv rate limiting the estimate
    was based on. Turso's pipeline API takes a batch, so a batch is what it gets.
    """
    url = config.TURSO_URL.replace("libsql://", "https://").rstrip("/")
    r = await http_client.get_client().post(
        f"{url}/v2/pipeline",
        json={"requests": [{"type": "execute", "stmt": s} for s in stmts]
                          + [{"type": "close"}]},
        headers={"Authorization": f"Bearer {config.TURSO_DB_TOKEN}",
                 "Content-Type": "application/json"},
        timeout=timeout)
    r.raise_for_status()
    out = []
    for res in r.json()["results"]:
        if res.get("type") == "error":
            raise RuntimeError(str(res.get("error"))[:300])
        resp = res.get("response", {})
        if resp.get("type") == "execute":
            out.append(resp["result"]["rows"])
    return out


async def _turso(sql: str, args: list | None = None, timeout: int = 90):
    stmt = {"sql": sql}
    if args is not None:
        stmt["args"] = args
    rows = await _pipeline([stmt], timeout=timeout)
    return rows[0] if rows else []


def _cell(v):
    if v is None:
        return {"type": "null", "value": None}
    if isinstance(v, int):
        return {"type": "integer", "value": str(v)}
    return {"type": "text", "value": str(v)}


_NS = {"atom": "http://www.w3.org/2005/Atom"}


async def fetch_abstracts(ids: list[str]) -> dict[str, str]:
    """ONE arXiv request for the whole batch, with backoff on 429.

    Deliberately not `arxiv_svc.fetch_metadata_batch`: that is tuned for small
    interactive lookups and fans any input out into 20-id sub-requests at ~3/s,
    which is what got this script rate-limited out of 9 of its first 20 batches.
    Here one batch is one request, so the pause between them means what it says.
    """
    params = {"id_list": ",".join(ids), "max_results": str(len(ids))}
    delay = PAUSE_S

    for attempt in range(MAX_RETRIES):
        try:
            r = await http_client.get_client().get(
                config.ARXIV_API_URL, params=params, timeout=60)
            if r.status_code == 429:
                # arXiv is explicitly asking us to slow down. Honour it rather
                # than burning the remaining retries at the same rate.
                delay *= 2
                print(f"    429 — backing off {delay:.0f}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})")
                await asyncio.sleep(delay)
                continue
            r.raise_for_status()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            delay *= 2
            print(f"    {type(e).__name__} — retrying in {delay:.0f}s")
            await asyncio.sleep(delay)
            continue

        out: dict[str, str] = {}
        root = ET.fromstring(r.text)
        for entry in root.findall("atom:entry", _NS):
            raw_id = (entry.findtext("atom:id", "", _NS) or "")
            summary = (entry.findtext("atom:summary", "", _NS) or "").strip()
            # arxiv_svc's own normaliser, not a hand-rolled one. Old-style ids
            # carry a category prefix (math/0309136v1) and a naive
            # rsplit("/")+split("v") mangles both halves — and CLAUDE.md §3.9
            # makes id integrity a hard rule.
            aid = arxiv_svc._normalise_id(raw_id)
            if aid and summary:
                out[aid] = " ".join(summary.split())
        return out

    return {}


async def survey() -> dict:
    total = int((await _turso("SELECT COUNT(*) FROM papers"))[0][0]["value"])
    capped = int((await _turso(
        f"SELECT COUNT(*) FROM papers WHERE {TRUNCATED}"))[0][0]["value"])
    return {"total": total, "capped": capped}


async def pick(limit: int, strategy: str) -> list[str]:
    """The truncated ids to repair, most useful first."""
    order = {
        # What the product actually surfaces. Trending and the feed both lean on
        # citation_count, so this is the closest cheap proxy for "will be read".
        "served": "ORDER BY citation_count DESC",
        # Newest first — what a freshness-driven feed shows.
        "recent": "ORDER BY update_date DESC",
        # Whatever the index yields; fastest to scan, useful for a full sweep.
        "any": "",
    }[strategy]
    rows = await _turso(
        f"SELECT arxiv_id FROM papers WHERE {TRUNCATED} "
        f"{order} LIMIT {int(limit)}")
    return [r[0]["value"] for r in rows]


async def repair(ids: list[str], apply: bool) -> dict:
    """Fetch full abstracts from arXiv and write the longer ones back."""
    stats = {"fetched": 0, "longer": 0, "written": 0, "no_gain": 0, "missing": 0}

    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        try:
            abstracts = await fetch_abstracts(chunk)
        except Exception as e:
            print(f"  batch {i // BATCH}: fetch failed ({str(e)[:80]}) — skipping")
            continue

        updates = []
        for aid in chunk:
            full = abstracts.get(aid)
            if not full:
                stats["missing"] += 1
                continue
            stats["fetched"] += 1
            if len(full) < 500 + MIN_GAIN_CHARS:
                stats["no_gain"] += 1
                continue
            stats["longer"] += 1
            updates.append((aid, full))

        if apply and updates:
            # One round trip for the whole batch, not one per row.
            try:
                await _pipeline([
                    {"sql": "UPDATE papers SET abstract_preview = ? "
                            "WHERE arxiv_id = ?",
                     "args": [_cell(full), _cell(aid)]}
                    for aid, full in updates
                ])
                stats["written"] += len(updates)
            except Exception as e:
                print(f"  batch write failed ({str(e)[:100]}) — rows unchanged")

        done = min(i + BATCH, len(ids))
        print(f"  {done}/{len(ids)}  longer={stats['longer']}  "
              f"written={stats['written']}  no-gain={stats['no_gain']}")

        if done < len(ids):
            await asyncio.sleep(PAUSE_S)

    return stats


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--all", action="store_true", help="every truncated row")
    ap.add_argument("--strategy", choices=("served", "recent", "any"),
                    default="served")
    ap.add_argument("--apply", action="store_true",
                    help="actually write. Without this nothing is modified.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not config.TURSO_URL or not config.TURSO_DB_TOKEN:
        print("TURSO_URL / TURSO_DB_TOKEN not configured")
        return

    s = await survey()
    pct = s["capped"] / s["total"] * 100 if s["total"] else 0
    print(f"corpus: {s['total']:,} papers, {s['capped']:,} truncated ({pct:.1f}%)\n")

    batches = -(-s["capped"] // BATCH)
    # MEASURED, not modelled. Three estimates were wrong before this one:
    # 13.6h counted only the arXiv pause; 22.6h added a guessed write cost but
    # still used an inflated truncation count; 10h fixed the count but kept the
    # guess. Timing real batches gives ~7.75s each — arXiv pause, the fetch, and
    # one batched write round trip.
    hours = batches * SECONDS_PER_BATCH / 3600
    print(f"full repair would be ~{batches:,} batches of {BATCH} "
          f"(~{hours:.1f} hours, arXiv rate limit + one write round trip each)\n")

    if args.dry_run or not args.apply:
        n = s["capped"] if args.all else args.limit
        print(f"DRY RUN — would repair {min(n, s['capped']):,} rows "
              f"(strategy: {args.strategy})")

        # Sample for real. The whole exercise rests on arXiv returning fuller
        # text than we hold, and that is an assumption worth testing before
        # committing thirteen hours to it.
        sample = await pick(min(20, n), args.strategy)
        if sample:
            abstracts = await fetch_abstracts(sample)
            gains = [len(v) for v in abstracts.values() if v]
            if gains:
                longer = sum(1 for g in gains if g >= 500 + MIN_GAIN_CHARS)
                print(f"\n  sampled {len(gains)} from arXiv:")
                print(f"    median full length   {sorted(gains)[len(gains)//2]:,} chars "
                      f"(stored: 500)")
                print(f"    materially longer    {longer}/{len(gains)}")
                print(f"    mean gain            "
                      f"{sum(gains)/len(gains) - 500:+,.0f} chars per abstract")
            else:
                print("\n  sample returned nothing from arXiv — check connectivity")
        print("\nre-run with --apply to write.")
        await http_client.aclose()
        return

    limit = s["capped"] if args.all else args.limit
    ids = await pick(limit, args.strategy)
    print(f"repairing {len(ids):,} rows, {args.strategy} first\n")

    t0 = time.time()
    stats = await repair(ids, apply=True)
    print(f"\ndone in {time.time() - t0:.0f}s: {stats}")
    print("re-run to continue; already-repaired rows drop out of the selection.")
    await http_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
