"""Side-by-side comparison: BEFORE vs AFTER citation boost.

Shows beginner vs expert results for the same topic.
Also verifies Band A (known-item) queries aren't broken.
"""
import asyncio, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import hybrid_search_svc, turso_svc, embed_svc

# Pairs: (topic, beginner_query, expert_query)
COMPARISONS = [
    ("TRANSFORMERS",
     "how do transformers work in NLP",
     "attention is all you need"),
    ("DIFFUSION",
     "what are diffusion models and how do they generate images",
     "denoising diffusion probabilistic models"),
    ("GPT-4",
     "how does GPT-4 work",
     "GPT-4 Technical Report"),
    ("RLHF",
     "what is reinforcement learning from human feedback",
     "reinforcement learning from human feedback"),
]

BAND_A = [
    ("attention is all you need", "1706.03762"),
    ("Deep Residual Learning for Image Recognition", "1512.03385"),
    ("BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "1810.04805"),
]

async def run_query(q: str):
    results = await hybrid_search_svc.search(q, limit=10)
    meta = {}
    if results:
        meta = await turso_svc.fetch_metadata_batch(results)
    return results, meta

async def main():
    print("Warming up BGE-M3...")
    embed_svc.encode_query("warmup")
    await turso_svc.fetch_metadata_batch(["1706.03762"])

    # === Band A verification ===
    print()
    print("=" * 90)
    print("BAND A VERIFICATION - Known-item queries (must still be #1)")
    print("=" * 90)
    for q, expected in BAND_A:
        results, meta = await run_query(q)
        rank = results.index(expected) + 1 if expected in results else -1
        status = "PASS" if rank == 1 else f"RANK #{rank}" if rank > 0 else "MISS"
        cites = meta.get(expected, {}).get("citation_count", 0)
        print(f"  [{status:>8}] {q[:55]:55s}  ({cites} cites)")

    # === Side-by-side comparisons ===
    print()
    print("=" * 90)
    print("SIDE-BY-SIDE: Beginner vs Expert queries (same topic)")
    print("=" * 90)

    for topic, beginner_q, expert_q in COMPARISONS:
        print(f"\n--- {topic} ---")

        # Beginner
        print(f"\n  BEGINNER: {beginner_q!r}")
        results, meta = await run_query(beginner_q)
        for i, aid in enumerate(results[:5], 1):
            m = meta.get(aid, {})
            title = (m.get("title") or "?")[:60]
            cites = m.get("citation_count", 0)
            print(f"    {i}. [{cites:>6} cites] {title}")

        # Expert
        print(f"\n  EXPERT:   {expert_q!r}")
        results, meta = await run_query(expert_q)
        for i, aid in enumerate(results[:5], 1):
            m = meta.get(aid, {})
            title = (m.get("title") or "?")[:60]
            cites = m.get("citation_count", 0)
            print(f"    {i}. [{cites:>6} cites] {title}")

    print()
    print("=" * 90)
    print("DONE")
    print("=" * 90)

if __name__ == "__main__":
    asyncio.run(main())
