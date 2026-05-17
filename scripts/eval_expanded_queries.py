"""
Expanded search quality evaluation — realistic user queries.

The original eval_search_quality.py uses 21 queries across 5 bands (A-E).
This script expands to 8 categories that simulate REAL users of an academic
paper search engine, not just known-item lookups and adversarial tests.

Categories:
  F: Beginner / Newcomer — "explain like I'm starting a research project"
  G: Research-in-Progress — "I know the field, looking for specific work"
  H: Implementation-Focused — "I want to BUILD something"
  I: Comparative / Survey — "compare X vs Y" or "survey of Z"
  J: Emerging / Cutting-Edge — "what's new in X?"
  K: Cross-Domain — "applying X from domain A to domain B"
  L: Vague / Exploratory — underspecified queries that real users actually type
  M: Follow-up / Refinement — queries that build on prior context

Run:  python scripts/eval_expanded_queries.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import hybrid_search_svc
from app import turso_svc
from app import embed_svc
from app import groq_svc


# ── Query definitions ────────────────────────────────────────────────────────

# (band, query, expected_arxiv_id_or_None, description)
QUERIES: list[tuple[str, str, str | None, str]] = [

    # ── Band A (original): Known-item titles ─────────────────────────────────
    ("A", "attention is all you need", "1706.03762",
     "Landmark transformer paper by Vaswani et al."),
    ("A", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "1810.04805",
     "Full BERT title — should be exact #1"),
    ("A", "Deep Residual Learning for Image Recognition", "1512.03385",
     "ResNet — the most-cited CV paper"),

    # ── Band F: Beginner / Newcomer queries ──────────────────────────────────
    # These simulate a student or newcomer who doesn't know the jargon.
    ("F", "how do transformers work in NLP",  None,
     "Newcomer asking about transformer basics"),
    ("F", "what is reinforcement learning from human feedback",  None,
     "Beginner asking about RLHF — should surface Ouyang/InstructGPT/Christiano"),
    ("F", "explain how neural networks learn",  None,
     "Very basic — should return foundational/survey papers"),
    ("F", "what are diffusion models and how do they generate images",  None,
     "Beginner asking about DDPM/Stable Diffusion family"),
    ("F", "how does GPT-4 work",  None,
     "Newcomer asking about GPT-4 — should surface the technical report"),

    # ── Band G: Research-in-Progress queries ─────────────────────────────────
    # These simulate a PhD student deep in their research.
    ("G", "contrastive learning for self-supervised visual representations",  None,
     "Should return SimCLR, MoCo, BYOL, DINO etc."),
    ("G", "knowledge distillation from large language models to smaller ones",  None,
     "Distillation pipeline — DistilBERT, TinyBERT, knowledge distillation surveys"),
    ("G", "graph neural networks for molecular property prediction",  None,
     "GNN + chemistry — SchNet, DimeNet, MPNN papers"),
    ("G", "efficient inference for large language models quantization pruning",  None,
     "LLM compression — GPTQ, AWQ, SparseGPT, pruning surveys"),
    ("G", "causal inference in observational studies with machine learning",  None,
     "Causal ML — double ML, causal forests, CATE estimation"),
    ("G", "multi-task learning with shared representations",  None,
     "MTL surveys, hard/soft parameter sharing, task relationships"),

    # ── Band H: Implementation-Focused queries ───────────────────────────────
    # These simulate someone who wants to BUILD something.
    ("H", "how to fine-tune a pre-trained language model for classification",  None,
     "Practical fine-tuning — ULMFiT, how-to-fine-tune-BERT papers"),
    ("H", "implementing attention mechanism from scratch",  None,
     "Implementation-level detail — attention tutorials, scaled dot product"),
    ("H", "best practices for training stable diffusion models",  None,
     "Practical SD training — latent diffusion, classifier-free guidance"),
    ("H", "building a retrieval augmented generation system",  None,
     "RAG — should surface the Lewis et al. RAG paper, REALM, etc."),
    ("H", "how to do distributed training with PyTorch across GPUs",  None,
     "Distributed training — ZeRO, Megatron, FSDP, DeepSpeed papers"),

    # ── Band I: Comparative / Survey queries ─────────────────────────────────
    # Users who want to understand the landscape.
    ("I", "transformer vs CNN for image classification",  None,
     "ViT vs ResNet/EfficientNet — should surface comparison papers"),
    ("I", "survey of large language models",  None,
     "LLM surveys — Zhao et al. survey, Minaee survey"),
    ("I", "comparison of object detection architectures YOLO vs DETR",  None,
     "YOLO family vs transformer-based detection"),
    ("I", "GAN vs diffusion models for image generation",  None,
     "Generative model comparison — StyleGAN, DDPM, score matching"),
    ("I", "review of federated learning privacy methods",  None,
     "FL surveys — McMahan, differential privacy in FL"),

    # ── Band J: Emerging / Cutting-Edge queries ──────────────────────────────
    # Users looking for the latest developments.
    ("J", "mixture of experts models scaling",  None,
     "MoE — Switch Transformer, Mixtral, GShard"),
    ("J", "test-time compute scaling for reasoning",  None,
     "New paradigm — o1-style reasoning, tree search at inference"),
    ("J", "multimodal large language models vision and text",  None,
     "GPT-4V, LLaVA, Flamingo, multimodal LLMs"),
    ("J", "state space models as alternative to transformers",  None,
     "S4, Mamba, H3 — structured state space models"),
    ("J", "constitutional AI and AI safety alignment techniques",  None,
     "Anthropic constitutional AI, RLHF alternatives, safety"),
    ("J", "sparse attention mechanisms for long context",  None,
     "Longformer, BigBird, sparse transformers for 100K+ context"),

    # ── Band K: Cross-Domain queries ─────────────────────────────────────────
    # Users applying ML to their specific domain.
    ("K", "deep learning for protein structure prediction",  None,
     "AlphaFold, ESMFold, protein language models"),
    ("K", "natural language processing for legal document analysis",  None,
     "Legal NLP — contract analysis, legal BERT, court opinion mining"),
    ("K", "machine learning for climate change prediction",  None,
     "Climate ML — weather forecasting, carbon modeling"),
    ("K", "using transformers for time series forecasting",  None,
     "Time series transformers — Informer, Autoformer, PatchTST"),
    ("K", "reinforcement learning for robotics manipulation",  None,
     "RL + robotics — sim-to-real transfer, dexterous manipulation"),

    # ── Band L: Vague / Exploratory queries ──────────────────────────────────
    # Underspecified queries that real users actually type.
    ("L", "AI ethics",  None,
     "Very broad — should return survey-level papers on AI ethics/fairness/bias"),
    ("L", "embedding",  None,
     "Single word — highly ambiguous. Word2Vec? Sentence embeddings? Image embeddings?"),
    ("L", "language model",  None,
     "Broad — should return influential LM papers or surveys"),
    ("L", "generate images from text",  None,
     "Casual — should surface DALL-E, Stable Diffusion, Imagen"),
    ("L", "make AI more safe",  None,
     "Very casual — should surface alignment/safety papers"),

    # ── Band M: Follow-up / Refinement queries ───────────────────────────────
    # Simulate a user who already found something and wants more.
    ("M", "improvements to the original transformer architecture",  None,
     "Post-Vaswani improvements — Reformer, Performer, ALiBi, RoPE"),
    ("M", "papers that cite ResNet and extend residual connections",  None,
     "ResNet extensions — DenseNet, ResNeXt, WideResNet, SE-Net"),
    ("M", "alternatives to RLHF for aligning language models",  None,
     "DPO, SPIN, KTO — methods that bypass reward modeling"),
    ("M", "BERT variants for low resource languages",  None,
     "mBERT, XLM-R, AfricanBERT, ArabBERT — multilingual BERT variants"),
]


# ── Wire rewrite logging ─────────────────────────────────────────────────────

_rewrite_log: dict[str, str] = {}
_original_rewrite = groq_svc.rewrite


async def _logging_rewrite(q: str) -> str:
    r = await _original_rewrite(q)
    _rewrite_log[q] = r
    return r


groq_svc.rewrite = _logging_rewrite


# ── Per-query evaluation ─────────────────────────────────────────────────────

async def eval_query(
    band: str, query: str, expected_id: str | None, description: str
) -> dict:
    """Run one query end-to-end and return structured results."""
    t0 = time.perf_counter()
    results = await hybrid_search_svc.search(query, limit=10)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    rewrite = _rewrite_log.get(query, query)
    rewrite_fired = rewrite.strip() != query.strip()

    titles: dict[str, str] = {}
    categories: dict[str, str] = {}
    if results:
        meta = await turso_svc.fetch_metadata_batch(results)
        titles = {aid: (m.get("title") or "(no title)") for aid, m in meta.items()}
        categories = {aid: (m.get("primary_topic") or "?") for aid, m in meta.items()}

    # Print formatted output
    print()
    print(f"[{band}] {query!r}")
    print(f"      intent: {description}")
    if rewrite_fired:
        print(f"      rewrite: {rewrite!r}")
    else:
        print(f"      rewrite: (skipped or no change)")

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
    else:
        for i, aid in enumerate(results, 1):
            title = titles.get(aid, "(title unavailable)")
            cat = categories.get(aid, "?")
            if len(title) > 75:
                title = title[:72] + "..."
            marker = " *" if expected_id and aid == expected_id else "  "
            print(f"  {i:2d}.{marker}{aid:14s} [{cat:20s}]  {title}")

    # Compute topic diversity
    unique_cats = set(categories.values()) - {"?"}

    return {
        "band": band,
        "query": query,
        "description": description,
        "rewrite": rewrite if rewrite_fired else None,
        "latency_ms": elapsed_ms,
        "n_results": len(results),
        "results": [
            {"rank": i+1, "arxiv_id": aid, "title": titles.get(aid, ""),
             "category": categories.get(aid, "?")}
            for i, aid in enumerate(results)
        ],
        "expected_id": expected_id,
        "expected_found": expected_id in results if expected_id else None,
        "expected_rank": results.index(expected_id) + 1 if expected_id and expected_id in results else None,
        "topic_diversity": len(unique_cats),
    }


async def main():
    print("=" * 100)
    print("EXPANDED SEARCH EVALUATION  -  Realistic User Queries")
    print(f"Total queries: {len(QUERIES)}  |  Bands: {sorted(set(b for b,_,_,_ in QUERIES))}")
    print("=" * 100)

    # Warm-up
    print("\nWarming up BGE-M3 + Turso...")
    t0 = time.perf_counter()
    embed_svc.encode_query("warmup query for the eval harness")
    await turso_svc.fetch_metadata_batch(["1706.03762"])
    print(f"Warm-up: {(time.perf_counter()-t0)*1000:.0f} ms\n")

    all_results: list[dict] = []
    band_results: dict[str, list[dict]] = {}

    for band, query, expected, description in QUERIES:
        result = await eval_query(band, query, expected, description)
        all_results.append(result)
        band_results.setdefault(band, []).append(result)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    # Band A: known-item hit rate
    if "A" in band_results:
        a_rows = band_results["A"]
        hits = sum(1 for r in a_rows if r["expected_rank"] == 1)
        total = len(a_rows)
        print(f"\nBand A (known-item): {hits}/{total} top-1 hits")

    # Per-band stats
    print("\nPer-Band Results:")
    print(f"  {'Band':<6} {'Queries':>7}  {'Avg Latency':>12}  {'Avg Results':>12}  {'Avg Topics':>11}  Description")
    print(f"  {'-'*6} {'-'*7}  {'-'*12}  {'-'*12}  {'-'*11}  {'-'*40}")

    band_labels = {
        "A": "Known-item titles",
        "F": "Beginner / Newcomer",
        "G": "Research-in-Progress",
        "H": "Implementation-Focused",
        "I": "Comparative / Survey",
        "J": "Emerging / Cutting-Edge",
        "K": "Cross-Domain",
        "L": "Vague / Exploratory",
        "M": "Follow-up / Refinement",
    }

    for band in sorted(band_results.keys()):
        rows = band_results[band]
        n = len(rows)
        avg_lat = sum(r["latency_ms"] for r in rows) / n
        avg_res = sum(r["n_results"] for r in rows) / n
        avg_div = sum(r["topic_diversity"] for r in rows) / n
        label = band_labels.get(band, "")
        print(f"  {band:<6} {n:>7}  {avg_lat:>10.0f}ms  {avg_res:>12.1f}  {avg_div:>11.1f}  {label}")

    # Overall latency
    all_lat = [r["latency_ms"] for r in all_results]
    all_lat.sort()
    n = len(all_lat)
    p50 = all_lat[n // 2]
    p95 = all_lat[max(0, int(n * 0.95) - 1)]
    print(f"\nOverall Latency (n={n}): mean {sum(all_lat)/n:.0f} ms  "
          f"p50 {p50:.0f} ms  p95 {p95:.0f} ms  max {max(all_lat):.0f} ms")

    # Rewrite analysis
    rewrites = [(r["query"], r["rewrite"]) for r in all_results if r["rewrite"]]
    skips = [r["query"] for r in all_results if not r["rewrite"]]
    print(f"\nGroq Rewriter: {len(rewrites)} fired, {len(skips)} skipped")

    # Zero-result queries
    zeros = [r["query"] for r in all_results if r["n_results"] == 0]
    if zeros:
        print(f"\nWARNING: ZERO RESULTS ({len(zeros)}):")
        for q in zeros:
            print(f"  - {q!r}")
    else:
        print(f"\nOK: All queries returned results")

    # Save JSON for comparison
    out_path = Path(__file__).parent / "expanded_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
