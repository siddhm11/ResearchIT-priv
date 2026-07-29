"""Trace where Vaswani's paper falls in the hybrid pipeline."""
import asyncio
from app import qdrant_svc, embed_svc, zilliz_svc, hybrid_search_svc

VASWANI = "1706.03762"


async def main():
    q = "attention is all you need"
    dense_vec, sparse_dict = embed_svc.encode_query(q)
    print(f"sparse keys: {len(sparse_dict)}")

    fetch_k = 60
    dense = await qdrant_svc.search_dense(dense_vec.tolist(), limit=fetch_k)
    sparse = await zilliz_svc.search_sparse(sparse_dict, limit=fetch_k)
    dense_ids = [r["arxiv_id"] for r in dense]
    sparse_ids = [r["arxiv_id"] for r in sparse]

    print(f"\nVaswani in dense top {fetch_k}: ", VASWANI in dense_ids,
          (f"(rank {dense_ids.index(VASWANI)+1})" if VASWANI in dense_ids else ""))
    print(f"Vaswani in sparse top {fetch_k}: ", VASWANI in sparse_ids,
          (f"(rank {sparse_ids.index(VASWANI)+1})" if VASWANI in sparse_ids else ""))

    fused = hybrid_search_svc._rrf_fuse(dense, sparse, k=60)
    fused_ids = [item["arxiv_id"] for item in fused]
    v_rank_rrf = fused_ids.index(VASWANI) + 1 if VASWANI in fused_ids else None
    print(f"\nVaswani rank after pure RRF: {v_rank_rrf}")

    print("\n=== Pure RRF (no recency), top 10 ===")
    for i, item in enumerate(fused[:10], 1):
        marker = " <-- VASWANI" if item["arxiv_id"] == VASWANI else ""
        print(f"  {i:2d}. {item['arxiv_id']}  rrf={item['rrf_score']:.4f}{marker}")

    ranked = hybrid_search_svc._recency_rerank([dict(x) for x in fused])
    ranked_ids = [item["arxiv_id"] for item in ranked]
    v_rank_recency = ranked_ids.index(VASWANI) + 1 if VASWANI in ranked_ids else None
    print(f"\nVaswani rank after current 0.80/0.20 recency rerank: {v_rank_recency}")

    print("\n=== Current rerank (0.80 RRF + 0.20 recency), top 10 ===")
    for i, item in enumerate(ranked[:10], 1):
        marker = " <-- VASWANI" if item["arxiv_id"] == VASWANI else ""
        print(f"  {i:2d}. {item['arxiv_id']}  final={item['final_score']:.4f}{marker}")


asyncio.run(main())
