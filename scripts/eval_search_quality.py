"""
Search quality evaluation harness.

For each curated query, runs the hybrid search pipeline end-to-end
(rewrite -> encode -> dense+sparse -> RRF -> title-boost) and prints the
top 10 results with titles fetched from Turso. For known-item queries,
flags whether the expected paper landed at #1.

This is a HUMAN-JUDGMENT report, not a pass/fail test. The output is
designed to be read top-to-bottom and rated query by query.

Run:  python scripts/eval_search_quality.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

# Make the project root importable when run as `python scripts/eval_search_quality.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import hybrid_search_svc
from app import turso_svc
from app import embed_svc
from app import groq_svc


# (band, query, expected_arxiv_id_or_None)
QUERIES: list[tuple[str, str, str | None]] = [
    # ── Band A: known-item title queries ──────────────────────────────────
    # The right answer is unambiguous. Top-1 hit is the bar.
    ("A", "attention is all you need", "1706.03762"),
    ("A", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "1810.04805"),
    ("A", "Adam: A Method for Stochastic Optimization", "1412.6980"),
    ("A", "Language Models are Few-Shot Learners", "2005.14165"),
    ("A", "Deep Residual Learning for Image Recognition", "1512.03385"),

    # ── Band B: conceptual semantic queries ───────────────────────────────
    # No exact keyword match; tests whether dense retrieval rescues meaning.
    ("B", "when AI makes up fake facts", None),
    ("B", "making language models follow human preferences", None),
    ("B", "why deep networks generalize despite overparameterization", None),
    ("B", "finding similar papers using vector embeddings", None),
    ("B", "models that pretend to be aligned but aren't", None),

    # ── Band C: keyword-academic queries ──────────────────────────────────
    # Already in academic form; rewriter heuristic should skip these.
    ("C", "BGE-M3 multilingual dense retrieval", None),
    ("C", "Mamba state space model linear time", None),
    ("C", "chain of thought prompting", None),
    ("C", "FlashAttention IO-aware exact attention", None),

    # ── Band D: adversarial / edge cases ──────────────────────────────────
    ("D", "transformr", None),                                          # typo
    ("D", "GPT", None),                                                 # very short
    ("D", "bayesian deep learning monte carlo dropout uncertainty estimation", None),  # very long
    ("D", "applying CV to medical imaging", None),                      # cross-domain (CV->medical)
    ("D", "attention", None),                                           # single ambiguous word

    # ── Band E: recency-sensitive queries ─────────────────────────────────
    # Recency rerank was removed; verify recent work still surfaces.
    ("E", "Llama 3", None),
    ("E", "reasoning models 2024", None),
]


# ── Wire a thin wrapper around groq_svc.rewrite to capture what fired ────
_rewrite_log: dict[str, str] = {}
_original_rewrite = groq_svc.rewrite


async def _logging_rewrite(q: str) -> str:
    r = await _original_rewrite(q)
    _rewrite_log[q] = r
    return r


groq_svc.rewrite = _logging_rewrite


async def eval_query(
    band: str, query: str, expected_id: str | None
) -> tuple[list[str], float]:
    """Run one query end-to-end and print a formatted report."""
    t0 = time.perf_counter()
    results = await hybrid_search_svc.search(query, limit=10)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    rewrite = _rewrite_log.get(query, query)
    rewrite_fired = rewrite.strip() != query.strip()

    titles: dict[str, str] = {}
    if results:
        meta = await turso_svc.fetch_metadata_batch(results)
        titles = {aid: (m.get("title") or "(no title)") for aid, m in meta.items()}

    # ── Header ──────────────────────────────────────────────────────────────
    print()
    print(f"[{band}] {query!r}")
    if rewrite_fired:
        print(f"      rewrite: {rewrite!r}")
    else:
        print(f"      rewrite: (heuristic skipped or no change)")

    if expected_id is not None:
        if results and results[0] == expected_id:
            verdict = f"PASS  -  {expected_id} at #1"
        elif expected_id in results:
            rank = results.index(expected_id) + 1
            verdict = f"PARTIAL  -  {expected_id} at rank #{rank}"
        else:
            verdict = f"FAIL  -  {expected_id} NOT in top 10"
        print(f"      verdict: {verdict}")

    print(f"      latency: {elapsed_ms:.0f} ms  |  results: {len(results)}")

    if not results:
        print("      (no results returned)")
        return results, elapsed_ms

    for i, aid in enumerate(results, 1):
        title = titles.get(aid, "(title unavailable)")
        if len(title) > 88:
            title = title[:85] + "..."
        marker = " *" if expected_id and aid == expected_id else "  "
        print(f"  {i:2d}.{marker}{aid:13s} {title}")

    return results, elapsed_ms


async def main():
    print("=" * 100)
    print("SEARCH QUALITY EVALUATION  -  ResearchIT hybrid search pipeline")
    print("=" * 100)

    # ── Warm-up ─────────────────────────────────────────────────────────────
    # First BGE-M3 encode is ~10-15s cold. Warm before timing anything.
    print("\nWarming up BGE-M3 + Turso...")
    t0 = time.perf_counter()
    embed_svc.encode_query("warmup query for the eval harness")
    await turso_svc.fetch_metadata_batch(["1706.03762"])
    print(f"Warm-up: {(time.perf_counter()-t0)*1000:.0f} ms\n")

    band_results: dict[str, list[tuple[str, str | None, list[str], float]]] = {}

    for band, query, expected in QUERIES:
        results, latency = await eval_query(band, query, expected)
        band_results.setdefault(band, []).append((query, expected, results, latency))

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    # Band A: top-1 hit rate
    if "A" in band_results:
        a_rows = band_results["A"]
        hits = sum(1 for _, exp, res, _ in a_rows if res and res[0] == exp)
        partial = sum(
            1 for _, exp, res, _ in a_rows
            if exp in (res or []) and (not res or res[0] != exp)
        )
        misses = len(a_rows) - hits - partial
        print(f"\nBand A (known-item titles): {hits}/{len(a_rows)} top-1 hits, "
              f"{partial} partial (in top 10 but not #1), {misses} miss")
        for q, exp, res, _ in a_rows:
            if res and res[0] == exp:
                tag = "PASS"
            elif exp in (res or []):
                tag = f"PARTIAL #{res.index(exp)+1}"
            else:
                tag = "MISS"
            qshort = q if len(q) <= 60 else q[:57] + "..."
            print(f"  [{tag:10s}] {exp:14s} {qshort}")

    # Latency stats
    all_lat = [lat for rows in band_results.values() for *_, lat in rows]
    if all_lat:
        all_lat.sort()
        n = len(all_lat)
        p50 = all_lat[n // 2]
        p95 = all_lat[max(0, int(n * 0.95) - 1)]
        print(f"\nLatency (n={n}): mean {sum(all_lat)/n:.0f} ms  "
              f"p50 {p50:.0f} ms  p95 {p95:.0f} ms  "
              f"max {max(all_lat):.0f} ms")

    # Per-band coverage (how often did we get any results?)
    print("\nResults coverage by band:")
    for band, rows in sorted(band_results.items()):
        empty = sum(1 for _, _, res, _ in rows if not res)
        print(f"  Band {band}: {len(rows) - empty}/{len(rows)} returned results")


if __name__ == "__main__":
    asyncio.run(main())
