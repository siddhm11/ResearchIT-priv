"""Diagnose why the Mamba paper (2312.00752) is missing from search results."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import qdrant_svc, embed_svc, zilliz_svc, hybrid_search_svc, turso_svc

MAMBA_ID = "2312.00752"


async def main():
    # Step 1: is the paper in Qdrant at all?
    vecs = await qdrant_svc.get_paper_vectors([MAMBA_ID])
    in_qdrant = MAMBA_ID in vecs
    print(f"Mamba paper {MAMBA_ID} in Qdrant: {in_qdrant}")

    # Step 2: is it in Turso?
    meta = await turso_svc.fetch_metadata_batch([MAMBA_ID])
    if MAMBA_ID in meta:
        print(f"Mamba paper in Turso: YES — title: {meta[MAMBA_ID].get('title')!r}")
    else:
        print("Mamba paper in Turso: NO")

    if not in_qdrant:
        print("\n--> Paper missing from Qdrant collection. End of investigation.")
        return

    # Step 3: where does it rank in dense, sparse, and fused?
    q = "Mamba state space model linear time"
    dense_vec, sparse_dict = embed_svc.encode_query(q)
    print(f"\nQuery: {q!r}")
    print(f"Sparse keys: {len(sparse_dict)}")

    fetch_k = 60
    dense = await qdrant_svc.search_dense(dense_vec.tolist(), limit=fetch_k)
    sparse = await zilliz_svc.search_sparse(sparse_dict, limit=fetch_k)

    dense_ids = [r["arxiv_id"] for r in dense]
    sparse_ids = [r["arxiv_id"] for r in sparse]

    if MAMBA_ID in dense_ids:
        print(f"\nDense rank: {dense_ids.index(MAMBA_ID)+1}/{fetch_k}")
    else:
        print(f"\nDense top {fetch_k}: NOT present")

    if MAMBA_ID in sparse_ids:
        print(f"Sparse rank: {sparse_ids.index(MAMBA_ID)+1}/{fetch_k}")
    else:
        print(f"Sparse top {fetch_k}: NOT present")

    fused = hybrid_search_svc._rrf_fuse(dense, sparse, k=60)
    fused_ids = [item["arxiv_id"] for item in fused]
    if MAMBA_ID in fused_ids:
        print(f"RRF fused rank: {fused_ids.index(MAMBA_ID)+1}")
    else:
        print(f"RRF fused: NOT present in top {len(fused_ids)}")

    # Show top 5 of each
    print(f"\n=== Dense top 5 ===")
    for r in dense[:5]:
        print(f"  {r['arxiv_id']}  score={r['score']:.4f}")
    print(f"\n=== Sparse top 5 ===")
    for r in sparse[:5]:
        print(f"  {r['arxiv_id']}  score={r['score']:.4f}")


asyncio.run(main())
