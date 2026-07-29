"""
Test script: Compare Turso DB vs arXiv API metadata fetch times.
Run: python -m tests.test_turso_timing
"""
import asyncio
import time
import sys
import os

# Ensure app module is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import turso_svc, arxiv_svc

# Sample arxiv IDs (known papers from our vector DBs)
TEST_IDS = [
    "1706.03762",   # Attention Is All You Need
    "2206.03003",   # Transformer attention medical
    "2209.15001",   # Dilated Neighborhood Attention Transformer
    "1809.04281",   # Music Transformer
    "2010.11929",   # ViT - Vision Transformer
    "1810.04805",   # BERT
    "2005.14165",   # GPT-3
    "2302.13971",   # LLaMA
    "1512.03385",   # ResNet
    "2103.00020",   # CLIP
]


async def test_turso():
    print("=" * 60)
    print("TURSO DB METADATA FETCH TEST")
    print("=" * 60)

    # Single paper
    t0 = time.perf_counter()
    result = await turso_svc.fetch_metadata(TEST_IDS[0])
    t1 = time.perf_counter()
    print(f"\n[Single] {TEST_IDS[0]} -> {(t1-t0)*1000:.0f}ms")
    if result:
        print(f"  Title:     {result['title'][:80]}")
        print(f"  Authors:   {result['authors'][:80]}")
        print(f"  Category:  {result['category']}")
        print(f"  Published: {result['published']}")
        print(f"  Year:      {result['year']}")
        print(f"  Citations: {result.get('citation_count', 'N/A')}")
        print(f"  Influential: {result.get('influential_citations', 'N/A')}")
    else:
        print("  NOT FOUND in Turso DB")

    # Batch of 10
    t0 = time.perf_counter()
    batch = await turso_svc.fetch_metadata_batch(TEST_IDS)
    t1 = time.perf_counter()
    turso_time = (t1 - t0) * 1000
    print(f"\n[Batch of {len(TEST_IDS)}] -> {turso_time:.0f}ms")
    print(f"  Found: {len(batch)}/{len(TEST_IDS)}")
    for aid, paper in batch.items():
        cites = paper.get("citation_count", 0)
        print(f"  {aid}: {paper['title'][:60]}... [{paper['category']}] (cites: {cites})")

    return turso_time, batch


async def test_arxiv():
    print("\n" + "=" * 60)
    print("ARXIV API METADATA FETCH TEST (for comparison)")
    print("=" * 60)

    t0 = time.perf_counter()
    batch = await arxiv_svc.fetch_metadata_batch(TEST_IDS)
    t1 = time.perf_counter()
    arxiv_time = (t1 - t0) * 1000
    print(f"\n[Batch of {len(TEST_IDS)}] -> {arxiv_time:.0f}ms")
    print(f"  Found: {len(batch)}/{len(TEST_IDS)}")
    for aid, paper in batch.items():
        print(f"  {aid}: {paper['title'][:60]}... [{paper['category']}]")

    return arxiv_time, batch


async def main():
    turso_time, turso_batch = await test_turso()
    arxiv_time, arxiv_batch = await test_arxiv()

    print("\n" + "=" * 60)
    print("TIMING COMPARISON")
    print("=" * 60)
    print(f"  Turso DB:   {turso_time:>8.0f}ms ({len(turso_batch)} papers)")
    print(f"  arXiv API:  {arxiv_time:>8.0f}ms ({len(arxiv_batch)} papers)")
    speedup = arxiv_time / turso_time if turso_time > 0 else float("inf")
    print(f"  Speedup:    {speedup:.1f}x faster with Turso")
    print()

    # Verify data quality: compare titles
    print("DATA QUALITY CHECK (title match):")
    for aid in TEST_IDS:
        t_title = turso_batch.get(aid, {}).get("title", "N/A")[:50]
        a_title = arxiv_batch.get(aid, {}).get("title", "N/A")[:50]
        match = "OK" if t_title.lower()[:30] == a_title.lower()[:30] else "DIFF"
        print(f"  [{match}] {aid}")


if __name__ == "__main__":
    asyncio.run(main())
