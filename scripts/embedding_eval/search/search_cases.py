"""Judged search test: the top 10 each search setup would show, for the judge panel.

    python search_cases.py "<system>" "<system>" ...
System names: "fts", "hybrid:<retriever>", or "<retriever>+<cross-encoder>@<depth>"
(see rerank.py). Queries: a fixed stratified sample of 80 LitSearch, 40 keyword and
40 question queries. Writes CACHE/judge/search_cases.json for judge.py --cases.
"""
from __future__ import annotations

import json
import math
import random
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, SEED, SIDECAR, log  # noqa: E402
from search_eval import FETCH_K, boost_order, queries, rrf  # noqa: E402

S, J = CACHE / "search", CACHE / "judge"


def main(systems: list[str]) -> None:
    lib = json.loads((S / "library.json").read_text())
    titles = lib["titles"]
    ids = lib["ids"]
    sc = sqlite3.connect(f"file:{SIDECAR}?mode=ro", uri=True)
    cites = {}
    for s in range(0, len(ids), 900):
        part = ids[s:s + 900]
        for a, c in sc.execute(f"SELECT arxiv_id, citation_count FROM papers WHERE arxiv_id IN ({','.join('?' * len(part))})", part):
            cites[a] = c or 0
    cand = json.loads((S / "candidates.json").read_text())
    F = json.loads((S / "fts_lists.json").read_text())
    ces = {}
    Q = queries()
    rnd = random.Random(SEED)
    pick = (rnd.sample([q for q in Q if q["set"].startswith("LitSearch")], 80)
            + rnd.sample([q for q in Q if q["set"] == "synthetic keyword"], 40)
            + rnd.sample([q for q in Q if q["set"] == "synthetic question"], 40))
    cases, used = [], set()
    for q in pick:
        lists = {}
        for sysname in systems:
            if sysname == "fts":
                L = F.get(q["qid"], [])[:10]
            elif sysname.startswith("hybrid:"):
                r = sysname.split(":", 1)[1]
                pool = cand[r][q["qid"]]
                # candidates.json holds the RRF order; the boost reproduces what is shown
                fused = [(a, 1.0 / (60 + k + 1)) for k, a in enumerate(pool)]
                L = boost_order(fused, q["query"], titles, cites)[:10]
            else:
                r, rest = sysname.split("+", 1)
                ce, depth = rest.split("@")
                if ce not in ces:
                    CE = json.loads((S / f"ce_{ce}.json").read_text())
                    v = list(CE.values())
                    if v and min(v) >= 0.0 and max(v) <= 1.0:   # probability outputs: back to logits
                        CE = {k: math.log(max(p, 1e-6) / max(1 - p, 1e-6)) for k, p in CE.items()}
                    ces[ce] = CE
                win = cand[r][q["qid"]][:int(depth)]
                fused = sorted(((a, 1 / (1 + math.exp(-ces[ce].get(f"{q['qid']}|{a}", -20.0)))) for a in win),
                               key=lambda x: -x[1])
                L = boost_order(fused, q["query"], titles, cites)[:10]
            lists[sysname] = L
            used |= set(L)
        cases.append({"surface": "search", "qid": q["qid"], "query": q["query"], "group": q["set"],
                      "gold": q["gold"], "lists": lists})
    papers = {a: {"title": titles[a], "abstract": lib["abstracts"][a], "cat": lib["cats"].get(a, ""), "year": None}
              for a in used}
    (J / "search_cases.json").write_text(json.dumps({"models": systems, "cases": cases, "papers": papers}))
    n_items = sum(len({a for L in c["lists"].values() for a in L}) for c in cases)
    log("search", f"search_cases.json: {len(cases)} queries, {len(systems)} systems, {n_items:,} items to judge")


if __name__ == "__main__":
    main(sys.argv[1:])
