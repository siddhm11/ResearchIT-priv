"""Stage 1: build the pair set and the calibration samples.

* Verifies every pair-set ID, title and category against the arXiv API.
* Picks the lexical trap (title says "attention", primary q-bio.NC) and the
  unrelated negative (primary astro-ph* or math.AG) from the corpus, seed 42.
* Samples 2,000 random corpus pairs and 2,000 cs.CL/cs.LG pairs, seed 42.
* Resolves each paper's embedding TEXT from the same source production used:
  the original arxiv_comprehensive_papers.csv for the 2025-05 snapshot (which
  produced shards a/b and Zilliz), the sidecar's full abstract for papers
  ingested later (arxiv_recent).

Read-only: Qdrant/Zilliz are only queried for membership.
"""
from __future__ import annotations

import random
import re
import sqlite3
import time
import urllib.parse
import xml.etree.ElementTree as ET

from common import (A, B, CACHE, CSV, N_BASELINE_PAIRS, NAMED, SEED, SIDECAR,
                    app_import_path, log, save_json)

app_import_path()
STAGE = "pairs"
CSV_INDEX = CACHE / "csv_index.sqlite"
CL_LG = {"cs.CL", "cs.LG"}


# ── The original CSV, exactly as siddhm11/bme-upload loaded it ───────────────

def format_arxiv_id(paper_id: str) -> str:
    """Verbatim from bme-upload: restores leading zeros lost in the CSV."""
    pid = str(paper_id).strip()
    if '.' in pid:
        parts = pid.split('.')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            yymm = parts[0].zfill(4)
            target = 5 if int(yymm) >= 1501 else 4
            suffix = parts[1].ljust(target, '0')
            return f"{yymm}.{suffix}"
    return pid


def build_csv_index() -> None:
    """id -> (row, title, abstract) after bme-upload's load_data() filtering.

    `row` is the position in the cleaned frame, which bme-upload used as the
    Qdrant point id; integrity_retrieval.py checks that correspondence.
    """
    if CSV_INDEX.exists():
        return
    import pandas as pd
    t0 = time.time()
    tmp = CSV_INDEX.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp)
    db.execute("CREATE TABLE p (row INTEGER, arxiv_id TEXT, title TEXT, abstract TEXT, categories TEXT)")
    header = pd.read_csv(CSV, nrows=0).columns.tolist()
    cat_col = next((c for c in ("categories", "category") if c in header), None)
    cols = ["id", "title", "abstract"] + ([cat_col] if cat_col else [])
    row = raw = 0
    for chunk in pd.read_csv(CSV, dtype={"id": str}, usecols=cols, chunksize=200_000):
        raw += len(chunk)
        chunk["id"] = chunk["id"].apply(format_arxiv_id)
        chunk = chunk.dropna(subset=["title"])
        chunk = chunk[chunk["title"].str.strip().str.len() > 10]
        chunk = chunk.dropna(subset=["abstract"])
        chunk = chunk[chunk["abstract"].str.strip().str.len() > 50]
        recs = [(row + k, r.id, r.title, r.abstract, getattr(r, cat_col) if cat_col else None)
                for k, r in enumerate(chunk.itertuples(index=False))]
        db.executemany("INSERT INTO p VALUES (?,?,?,?,?)", recs)
        row += len(recs)
    db.execute("CREATE INDEX ix_id ON p(arxiv_id)")
    db.execute("CREATE TABLE meta (k TEXT, v TEXT)")
    db.executemany("INSERT INTO meta VALUES (?,?)",
                   [("raw_rows", str(raw)), ("clean_rows", str(row)), ("columns", ",".join(header))])
    db.commit()
    db.close()
    tmp.replace(CSV_INDEX)
    log(STAGE, f"CSV index: {raw:,} raw rows -> {row:,} after bme-upload filtering "
               f"({time.time() - t0:.0f}s)")


def csv_rows(ids: list[str]) -> dict[str, dict]:
    db = sqlite3.connect(f"file:{CSV_INDEX}?mode=ro", uri=True)
    out = {}
    for s in range(0, len(ids), 500):
        part = ids[s:s + 500]
        q = f"SELECT row, arxiv_id, title, abstract, categories FROM p WHERE arxiv_id IN ({','.join('?' * len(part))})"
        for row, aid, title, abstract, cats in db.execute(q, part):
            out.setdefault(aid, {"row": row, "title": title, "abstract": abstract, "categories": cats})
    return out


# ── arXiv API ────────────────────────────────────────────────────────────────

def arxiv_lookup(ids: list[str]) -> dict[str, dict]:
    from ingest_arxiv import (ARXIV_API, ARXIV_DELAY, NS, _arxiv_get, _permanent,
                              parse_entry, retry_wait)
    out = {}
    for s in range(0, len(ids), 50):
        part = ids[s:s + 50]
        url = f"{ARXIV_API}?" + urllib.parse.urlencode(
            {"id_list": ",".join(part), "max_results": len(part)})
        for attempt in range(6):
            try:
                with _arxiv_get(url, 60) as r:
                    root = ET.fromstring(r.read())
                break
            except Exception as e:  # noqa: BLE001 -- mirrors ingest_arxiv's retry policy
                if _permanent(e) or attempt == 5:
                    raise
                time.sleep(retry_wait(e, attempt))
        for entry in root.findall("atom:entry", NS):
            p = parse_entry(entry)
            if not p:
                continue
            raw_id = entry.findtext("atom:id", "", NS)
            p["version"] = raw_id.rsplit("v", 1)[-1] if re.search(r"v\d+$", raw_id) else None
            p["abstract_raw"] = entry.findtext("atom:summary", "", NS)
            p["published"] = entry.findtext("atom:published", "", NS)[:10]
            out[p["arxiv_id"]] = p
        time.sleep(ARXIV_DELAY)
    return out


# ── Corpus membership (read-only) ────────────────────────────────────────────

def indexed(ids: list[str]) -> dict[str, dict]:
    """Scrolls each shard directly. get_paper_vectors' fan-out turns a failed
    shard into an empty result, which would read as "not indexed"; here a
    failure raises instead."""
    from app import config, qdrant_svc
    from pymilvus import MilvusClient
    where: dict[str, str] = {}
    for backend in qdrant_svc._active_backends():
        with qdrant_svc.use_backend(backend):
            for p in qdrant_svc._scroll_vectors_by_arxiv_ids(ids):
                where.setdefault(p.payload["arxiv_id"], qdrant_svc._collection())
    z = MilvusClient(uri=config.ZILLIZ_URI, token=config.ZILLIZ_TOKEN)
    quoted = ",".join(f'"{i}"' for i in ids)
    zrows = z.query(config.ZILLIZ_COLLECTION, filter=f"arxiv_id in [{quoted}]",
                    output_fields=["arxiv_id"])
    zset = {r["arxiv_id"] for r in zrows}
    return {i: {"qdrant": i in where, "qdrant_collection": where.get(i), "zilliz": i in zset}
            for i in ids}


# ── Selection ────────────────────────────────────────────────────────────────

def primary(categories: str | None) -> str:
    return (categories or "").split()[0] if categories else ""


def pick(candidates: list[str], api_check, label: str) -> tuple[str, dict, int]:
    """Seeded shuffle of the sorted candidates; first one the arXiv API confirms."""
    order = sorted(candidates)
    random.Random(SEED).shuffle(order)
    for aid in order[:10]:
        meta = arxiv_lookup([aid]).get(aid)
        if meta and api_check(meta):
            return aid, meta, len(order)
        log(STAGE, f"{label}: {aid} rejected by arXiv API check ({meta and meta['primary_topic']})")
    raise RuntimeError(f"no {label} candidate passed the arXiv API check")


def main() -> None:
    build_csv_index()
    sc = sqlite3.connect(f"file:{SIDECAR}?mode=ro", uri=True)
    rows = sc.execute("SELECT arxiv_id, title, categories, length(abstract_preview) FROM papers").fetchall()
    log(STAGE, f"sidecar rows: {len(rows):,}")
    all_ids = sorted(r[0] for r in rows)
    cats = {r[0]: r[2] for r in rows}
    in_csv = set(csv_rows(all_ids))
    log(STAGE, f"sidecar papers present in original CSV: {len(in_csv):,}")

    # Lexical trap: "attention" as a word in the title, primary q-bio.NC, snapshot paper.
    word = re.compile(r"\battention", re.I)
    trap_cands = [r[0] for r in rows if r[1] and word.search(r[1])
                  and primary(r[2]) == "q-bio.NC" and r[0] in in_csv]
    trap, trap_meta, n_trap = pick(trap_cands, lambda m: m["primary_topic"] == "q-bio.NC", "trap")
    # Unrelated: primary astro-ph* or math.AG, snapshot paper.
    unrel_cands = [r[0] for r in rows if r[0] in in_csv and
                   (primary(r[2]).startswith("astro-ph") or primary(r[2]) == "math.AG")]
    unrel, unrel_meta, n_unrel = pick(
        unrel_cands, lambda m: m["primary_topic"].startswith("astro-ph") or m["primary_topic"] == "math.AG",
        "unrelated")

    roles = dict(NAMED)
    roles[trap] = "lexical_trap"
    roles[unrel] = "unrelated"
    ids = list(roles)
    api = arxiv_lookup(ids)
    csvd = csv_rows(ids)
    member = indexed(ids)

    papers = {}
    for aid in ids:
        m = api.get(aid)
        if m is None:
            raise RuntimeError(f"{aid} not found by the arXiv API")
        c = csvd.get(aid)
        in_corpus = member[aid]["qdrant"]
        if c is not None:
            title, abstract, source, recipe = c["title"], c["abstract"], "original_csv", "original"
        else:
            # Not in the snapshot: embed what the ingest would, flagged below.
            title, abstract, source, recipe = m["title"], m["abstract"], "arxiv_api", "ingest"
        papers[aid] = {
            "role": roles[aid], "title": title, "abstract": abstract,
            "text_source": source, "recipe": recipe, "csv_row": c["row"] if c else None,
            "indexed": member[aid], "not_indexed": not in_corpus,
            "sidecar_categories": cats.get(aid),
            "arxiv_api": {k: m[k] for k in ("title", "primary_topic", "categories", "published",
                                            "update_date", "version", "abstract", "abstract_raw")},
        }
        log(STAGE, f"{aid} {roles[aid]:24} api_primary={m['primary_topic']:10} "
                   f"indexed={member[aid]} source={source}  {m['title'][:60]}")

    selection = {
        "lexical_trap": {"arxiv_id": trap, "candidates": n_trap,
                         "rule": "title matches /\\battention/i, sidecar primary category q-bio.NC, "
                                 "paper in the original CSV snapshot; seed-42 shuffle of sorted "
                                 "candidates, first one whose arXiv API primary_category is q-bio.NC"},
        "unrelated": {"arxiv_id": unrel, "candidates": n_unrel,
                      "rule": "sidecar primary category astro-ph* or math.AG, paper in the original "
                              "CSV snapshot; seed-42 shuffle, first confirmed by the arXiv API"},
    }
    save_json("pairset.json", {"papers": papers, "selection": selection, "anchors": [A, B]})

    # ── Calibration samples ──────────────────────────────────────────────────
    exclude = set(ids)
    abs_len = {r[0]: r[3] for r in rows}

    def usable(aid: str) -> bool:
        # Recent papers need a full sidecar abstract; the 500-char cap marks
        # legacy truncated rows, whose text would not be what was embedded.
        return aid not in exclude and (aid in in_csv or (abs_len.get(aid) or 0) > 500)

    def sample(pool: list[str], tag: str) -> list[list[str]]:
        rng = random.Random(SEED)
        chosen, seen = [], set()
        for aid in rng.sample(pool, min(len(pool), 3 * 2 * N_BASELINE_PAIRS)):
            if aid not in seen and usable(aid):
                chosen.append(aid)
                seen.add(aid)
            if len(chosen) == 2 * N_BASELINE_PAIRS:
                break
        n = len(chosen) // 2
        log(STAGE, f"baseline {tag}: {n} pairs from a pool of {len(pool):,}")
        return [[chosen[k], chosen[n + k]] for k in range(n)]

    random_pairs = sample(all_ids, "random")
    cl_lg_pool = [a for a in all_ids if primary(cats[a]) in CL_LG]
    cl_lg_pairs = sample(cl_lg_pool, "cs.CL/cs.LG")

    need = sorted({a for p in random_pairs + cl_lg_pairs for a in p})
    crow = csv_rows(need)
    texts = {}
    side = {}
    rest = [a for a in need if a not in crow]
    for s in range(0, len(rest), 500):
        part = rest[s:s + 500]
        q = f"SELECT arxiv_id, title, abstract_preview FROM papers WHERE arxiv_id IN ({','.join('?' * len(part))})"
        side.update({a: (t, ab) for a, t, ab in sc.execute(q, part)})
    for a in need:
        if a in crow:
            texts[a] = {"title": crow[a]["title"], "abstract": crow[a]["abstract"],
                        "recipe": "original", "primary": primary(cats[a])}
        else:
            t, ab = side[a]
            texts[a] = {"title": t, "abstract": ab, "recipe": "ingest", "primary": primary(cats[a])}
    n_recent = sum(1 for a in need if texts[a]["recipe"] == "ingest")
    save_json("baseline.json", {"random": random_pairs, "cscl_lg": cl_lg_pairs, "texts": texts,
                                "pool_sizes": {"random": len(all_ids), "cscl_lg": len(cl_lg_pool)},
                                "n_papers": len(need), "n_from_sidecar_recent": n_recent})
    log(STAGE, f"baseline texts: {len(need):,} papers ({n_recent:,} recent, sidecar text)")


if __name__ == "__main__":
    main()
