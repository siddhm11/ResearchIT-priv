#!/usr/bin/env python3
"""
Build a self-contained world for the browser tests: no Turso, Qdrant, Groq or
arXiv involved.

    python scripts/seed_e2e.py --db /tmp/e2e.db --site /tmp/e2e_arxiv \\
        --sidecar /tmp/e2e_sidecar.sqlite

It writes three things:

  sidecar   a tiny metadata sidecar with the six fixture papers from
            tests/fixtures/arxiv_html/, plus two made-up ones: 2609.99001 (an
            abstract full of maths, and a formula carrying an XSS payload, with
            no arXiv HTML) and 2609.99002 (no plain summary seeded, so the page
            must fall back to the abstract);
  db        the app database with each fixture page already parsed and saved
            (paper_sources, full text, paragraphs) and plain summaries for three
            papers, so nothing has to be fetched or generated;
  site      a stand-in for arxiv.org/html: each page at /<id> and a generated
            PNG at every figure path, served by `python -m http.server`.

Then start the app against them with every outside service blanked, so the
values in .env.local can't leak in (load_dotenv never overrides a set name):

    DB_PATH=/tmp/e2e.db METADATA_SIDECAR_PATH=/tmp/e2e_sidecar.sqlite \\
    RESEARCHIT_ENV=test ARXIV_HTML_BASE=http://127.0.0.1:8799 \\
    TURSO_URL= TURSO_DB_TOKEN= QDRANT_URL= GROQ_API_KEY= TURSO_SYNC_DISABLED=1 \\
    PAPER_PAGE_V2=on FIGURES=on PAPER_SOURCES=on PORT=7861 python run.py
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import io
import os
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "arxiv_html"

PAPERS = {
    "2609.31507": ("cs.CV cs.RO", "Jiajun Jiang, Chunliang Hua, Zichun Chen, Yanxing Wu, Zeyuan Yang, Jie Song, Xiao Hu"),
    "2609.20888": ("cs.LG cs.CL", "Themistoklis Haris, Henry Li, Maryam Karimzadehgan"),
    "2609.27176": ("cs.CL cs.LG", "Divyansh Singh"),
    "2609.29861": ("cs.CV cs.RO", "Guangzhao Dai, Qianru Sun, Qi Wu, Bin Zhu"),
    "2609.30356": ("cs.CV", "Andrew Renninger"),
    "2609.30670": ("cs.CV cs.AI", "Yibo Ma, Qianqian Zhang, Peng Liu, Tiancheng Zhao"),
}
MATHS_ID, NO_SUMMARY_ID = "2609.99001", "2609.99002"
MATHS_ABSTRACT = (
    "We bound the error of attention by $\\epsilon \\le \\frac{1}{\\sqrt{d}}$ and show that "
    "$$\\sum_{i=1}^{n} a_i^2 = 1$$ holds for every head. A formula that tries to inject markup, "
    "$<img src=x onerror=\"window.__xss=1\">$, must stay text.")
SUMMARIES = {
    "2609.31507": "Testing how well agents follow directions across whole cities has been limited by "
                  "the cost of 3D city models. The authors build SatNav from satellite photos instead, "
                  "with 118,494 tasks in 18 cities, and find that even strong models struggle.",
    "2609.27176": "Knowing that test questions leaked into training does not say how much they helped. "
                  "LeakScale measures that effect directly by planting controlled exposures.",
    MATHS_ID: "The authors prove a bound on attention error that shrinks as the model gets wider.",
}


def load_sidecar_schema() -> str:
    spec = importlib.util.spec_from_file_location("build_metadata_sidecar", ROOT / "scripts" / "build_metadata_sidecar.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SCHEMA


def build_sidecar(path: Path, parsed: dict) -> None:
    path.unlink(missing_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(load_sidecar_schema())
    rows = []
    for aid, (cats, authors) in PAPERS.items():
        p = parsed[aid]
        rows.append((aid, p.title, authors, p.abstract, cats, "AI/ML", "2026-09-25", 3, 0))
    rows.append((MATHS_ID, "Attention error bounds in $O(1/\\sqrt{d})$", "Ada Lovelace, Alan Turing",
                 MATHS_ABSTRACT, "cs.LG", "AI/ML", "2026-09-26", 0, 0))
    rows.append((NO_SUMMARY_ID, "A paper with no plain summary", "Grace Hopper",
                 "This abstract stands alone because no plain-language summary exists for it.",
                 "cs.AI", "AI/ML", "2026-09-26", 0, 0))
    conn.executemany("INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.executemany("INSERT INTO paper_categories VALUES (?, ?, ?, ?)",
                     [(cats.split()[0], aid, 3, "2026-09-25") for aid, (cats, _) in PAPERS.items()])
    conn.commit()
    conn.close()


def png_bytes(w: int, h: int, seed: int) -> bytes:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (max(w, 40), max(h, 30)), (250, 248, 243))
    d = ImageDraw.Draw(im)
    for i in range(6):
        x = (seed * 37 + i * 61) % im.width
        d.rectangle([x, 10 + i * 8, min(im.width - 1, x + im.width // 4), 20 + i * 8 + im.height // 5],
                    outline=(40, 80 + i * 20, 120), width=3)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def build_site(site: Path, parsed: dict) -> int:
    if site.exists():
        shutil.rmtree(site)
    site.mkdir(parents=True)
    n = 0
    for aid in PAPERS:
        shutil.copy(FIXTURES / f"{aid}.html", site / aid)
        for i, fig in enumerate(parsed[aid].figures):
            for img in fig.images:
                if img.src and not img.missing:
                    target = site / img.src
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(png_bytes(img.width or 480, img.height or 240, i))
                    n += 1
    return n


async def seed_db(db_path: Path, parsed: dict) -> None:
    from app import config, db, groq_svc, paper_source_svc
    config.DB_PATH = db.DB_PATH = str(db_path)
    await db.init_db()
    for aid in PAPERS:
        await paper_source_svc._save_parsed(parsed[aid], None)
    await paper_source_svc._save_status(MATHS_ID, "no_html", "", None)
    abstracts = {aid: parsed[aid].abstract for aid in PAPERS}
    abstracts[MATHS_ID] = MATHS_ABSTRACT
    for aid, text in SUMMARIES.items():
        await db.save_explanation(groq_svc.explain_cache_key(aid, abstracts[aid]), aid, text, "seed")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--site", required=True, type=Path)
    ap.add_argument("--sidecar", required=True, type=Path)
    args = ap.parse_args(argv)

    os.environ.setdefault("TURSO_SYNC_DISABLED", "1")
    sys.path.insert(0, str(ROOT))
    from app import arxiv_html
    parsed = {aid: arxiv_html.parse((FIXTURES / f"{aid}.html").read_bytes(), aid) for aid in PAPERS}

    args.db.unlink(missing_ok=True)
    build_sidecar(args.sidecar, parsed)
    images = build_site(args.site, parsed)
    asyncio.run(seed_db(args.db, parsed))
    print(f"[seed_e2e] sidecar {args.sidecar} ({len(PAPERS) + 2} papers), db {args.db}, "
          f"site {args.site} ({images} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
