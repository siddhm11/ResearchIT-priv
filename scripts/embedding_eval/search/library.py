"""The search test library: every paper the search tests can return.

  bake-off round-1 universe (86,787: citation-test papers, readers' papers,
  30,000 random corpus papers) + the AI/ML set (13,834) + LitSearch gold papers.
The 30,000 random papers double as the background sample for full-corpus rank
estimates. Texts follow the bake-off recipe: original CSV where the paper is in
it, otherwise the sidecar's full abstract. Writes CACHE/search/library.json.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CACHE, SIDECAR, log  # noqa: E402
from select_pairs import csv_rows  # noqa: E402

S = CACHE / "search"


def main() -> None:
    U = json.loads((CACHE / "bakeoff" / "universe.json").read_text())
    A = json.loads((CACHE / "aiml" / "aiml_universe.json").read_text())
    L = json.loads((S / "litsearch_queries.json").read_text())
    gold = sorted({g for q in L for g in q["gold"]})
    ids = sorted(set(U["ids"]) | set(A["ids"]) | set(gold))
    titles = {**U["titles"], **A["titles"]}
    abstracts = {**U["abstracts"], **A["abstracts"]}
    need = [i for i in ids if i not in titles]
    rows = csv_rows(need)
    for i, r in rows.items():
        titles[i], abstracts[i] = r["title"].strip(), (r["abstract"] or "").strip()
    sc = sqlite3.connect(f"file:{SIDECAR}?mode=ro", uri=True)
    rest = [i for i in need if i not in rows]
    for s in range(0, len(rest), 500):
        part = rest[s:s + 500]
        for a, t, ab in sc.execute(f"SELECT arxiv_id, title, abstract_preview FROM papers WHERE arxiv_id IN "
                                   f"({','.join('?' * len(part))})", part):
            titles[a], abstracts[a] = (t or "").strip(), (ab or "").strip()
    cat = {}
    for s in range(0, len(ids), 900):
        part = ids[s:s + 900]
        for a, c in sc.execute(f"SELECT arxiv_id, categories FROM papers WHERE arxiv_id IN ({','.join('?' * len(part))})", part):
            cat[a] = (c or "").split()[0] if c else ""
    out = {"ids": ids, "background": U["background"], "aiml_seeds": A["seeds"], "aiml_stratum": A["stratum"],
           "titles": {i: " ".join((titles.get(i) or "").split()) for i in ids},
           "abstracts": {i: " ".join((abstracts.get(i) or "").split()) for i in ids}, "cats": cat}
    (S / "library.json").write_text(json.dumps(out))
    log("search", f"library {len(ids):,} papers ({len(gold)} LitSearch gold, {len(need)} newly added); "
                  f"missing abstract {sum(1 for i in ids if not out['abstracts'][i])}")


if __name__ == "__main__":
    main()
