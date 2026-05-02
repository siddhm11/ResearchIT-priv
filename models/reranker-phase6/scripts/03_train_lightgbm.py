"""
Step 3: Train LightGBM lambdarank reranker + compare against heuristic baseline.

Produces:
  - reranker_v1.txt          — trained LightGBM model (~100KB)
  - eval_metrics.json         — nDCG@10, Recall@50, label distribution, feature importance
  - feature_importance.csv    — ranked feature importance
  - baseline_comparison.json  — LightGBM vs heuristic scorer on same eval set

Usage:
  python 03_train_lightgbm.py \
    --train-file ltr_dataset/train.parquet \
    --eval-file ltr_dataset/eval.parquet \
    --output-dir ./model_output \
    --num-boost-round 500 \
    --learning-rate 0.05

Prerequisites:
  - train.parquet + eval.parquet from Step 2
  - pip install lightgbm pyarrow numpy

The heuristic baseline replicates the EXACT scoring logic from
app/recommend/reranker.py → heuristic_score():
  score = 0.40 × lt_sim + 0.25 × st_sim + 0.15 × recency
        + 0.10 × rrf_conf - 0.15 × neg_penalty

Since pseudo-label training has no user profiles (features 20-30 = 0),
the heuristic baseline for pseudo-labels simplifies to:
  score = 0.15 × recency + 0.10 × (1 - position/max_position)

This is the fair baseline: both models see the same zero-filled user features.

Author: ResearchIT ML Pipeline — Phase 6, Step 3
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq


# ── Feature schema (must match Step 2) ───────────────────────────────────────

FEATURE_SCHEMA = [
    "qdrant_cosine_score", "candidate_position", "candidate_citation_count",
    "candidate_log_citations", "candidate_influential_citations",
    "candidate_age_days", "candidate_recency_score", "query_citation_count",
    "query_age_days", "year_diff", "same_primary_category", "co_citation_count",
    "shared_author_count", "candidate_is_newer", "query_log_citations",
    "citation_count_ratio", "age_ratio", "candidate_citations_per_year",
    "query_num_references", "candidate_num_cited_by",
    "ewma_longterm_similarity", "ewma_shortterm_similarity",
    "ewma_negative_similarity", "cluster_importance",
    "cluster_distance_to_medoid", "is_suppressed_category",
    "onboarding_category_match", "user_total_saves", "user_total_dismissals",
    "user_days_since_last_save", "user_session_save_count",
    "cosine_x_recency", "cosine_x_citations", "category_x_recency",
    "cosine_x_cocitation", "position_inverse", "citations_x_recency",
]

NUM_FEATURES = 37


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_ltr_data(parquet_path: str) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    """
    Load a parquet file into LightGBM-ready format.

    Returns:
        features:  (N, 37) float32 matrix
        labels:    (N,) int32 array (0, 1, or 2)
        groups:    list of group sizes (candidates per query)
        query_ids: list of query arXiv IDs (one per row, for analysis)
    """
    table = pq.read_table(parquet_path)

    query_ids = table.column("query_arxiv_id").to_pylist()
    labels = np.array(table.column("label").to_pylist(), dtype=np.int32)

    # Extract feature columns
    feature_arrays = []
    for fname in FEATURE_SCHEMA:
        col = table.column(fname).to_pylist()
        feature_arrays.append(col)
    features = np.column_stack(feature_arrays).astype(np.float32)

    # Compute group sizes (number of candidates per query)
    groups = []
    current_qid = None
    current_count = 0
    for qid in query_ids:
        if qid != current_qid:
            if current_qid is not None:
                groups.append(current_count)
            current_qid = qid
            current_count = 1
        else:
            current_count += 1
    if current_count > 0:
        groups.append(current_count)

    # Verify consistency
    assert sum(groups) == len(labels), f"Group sum {sum(groups)} != {len(labels)} rows"
    assert features.shape == (len(labels), NUM_FEATURES), f"Feature shape mismatch"

    return features, labels, groups, query_ids


# ── Heuristic Baseline ──────────────────────────────────────────────────────

def heuristic_baseline_score(features: np.ndarray) -> np.ndarray:
    """
    Replicate the EXACT scoring logic from app/recommend/reranker.py.

    heuristic_score():
      lt_sim   = features[:, 0]  → here: ewma_longterm_similarity (col 20) = 0
      st_sim   = features[:, 1]  → here: ewma_shortterm_similarity (col 21) = 0
      age_days = features[:, 2]  → here: candidate_age_days (col 5)
      rrf_pos  = features[:, 3]  → here: candidate_position (col 1)
      neg_sim  = features[:, 4]  → here: ewma_negative_similarity (col 22) = 0

    For pseudo-label data, EWMA features are 0, so score simplifies to:
      score = 0.15 × exp(-0.002 × age_days) + 0.10 × (1 - pos/max_pos)

    But we also include the cosine score (col 0) since that's what the
    reranker would actually see in production (it's feature 0 = lt_sim proxy).
    In the real pipeline, lt_sim IS the cosine similarity to the long-term
    profile — for pseudo-labels, the closest proxy is qdrant_cosine_score.

    So the fair pseudo-label heuristic baseline is:
      score = 0.40 × qdrant_cosine_score  (proxy for lt_sim)
            + 0.15 × recency_decay
            + 0.10 × rrf_confidence
    """
    qdrant_cosine = features[:, 0]   # qdrant_cosine_score
    position = features[:, 1]        # candidate_position
    age_days = features[:, 5]        # candidate_age_days

    # Recency: exp(-0.002 * age_days) — matches reranker.py exactly
    recency = np.exp(-0.002 * age_days)

    # RRF confidence: inverse of position (normalised)
    max_pos = position.max() + 1
    rrf_conf = 1.0 - (position / max_pos)

    scores = (
        0.40 * qdrant_cosine
        + 0.15 * recency
        + 0.10 * rrf_conf
    )
    return scores


# ── Evaluation Metrics ───────────────────────────────────────────────────────

def ndcg_at_k(labels: np.ndarray, scores: np.ndarray, groups: list[int], k: int = 10) -> float:
    """Compute mean nDCG@k across all queries."""
    ndcg_scores = []
    offset = 0
    for group_size in groups:
        group_labels = labels[offset:offset + group_size]
        group_scores = scores[offset:offset + group_size]

        # Sort by predicted score descending
        order = np.argsort(-group_scores)
        sorted_labels = group_labels[order]

        # DCG@k
        top_k = sorted_labels[:k]
        gains = (2.0 ** top_k) - 1.0
        discounts = np.log2(np.arange(len(top_k)) + 2.0)
        dcg = np.sum(gains / discounts)

        # Ideal DCG@k
        ideal_order = np.argsort(-group_labels)
        ideal_labels = group_labels[ideal_order][:k]
        ideal_gains = (2.0 ** ideal_labels) - 1.0
        ideal_discounts = np.log2(np.arange(len(ideal_labels)) + 2.0)
        idcg = np.sum(ideal_gains / ideal_discounts)

        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        # Skip queries with all-zero labels (no positives)

        offset += group_size

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def recall_at_k(labels: np.ndarray, scores: np.ndarray, groups: list[int], k: int = 50) -> float:
    """Compute mean Recall@k (fraction of positives in top-k) across all queries."""
    recalls = []
    offset = 0
    for group_size in groups:
        group_labels = labels[offset:offset + group_size]
        group_scores = scores[offset:offset + group_size]

        total_positives = np.sum(group_labels > 0)
        if total_positives == 0:
            offset += group_size
            continue

        order = np.argsort(-group_scores)
        sorted_labels = group_labels[order]
        top_k_positives = np.sum(sorted_labels[:k] > 0)
        recalls.append(top_k_positives / total_positives)

        offset += group_size

    return float(np.mean(recalls)) if recalls else 0.0


def hit_rate_at_k(labels: np.ndarray, scores: np.ndarray, groups: list[int], k: int = 10) -> float:
    """Compute HR@k: fraction of queries where at least one positive is in top-k."""
    hits = 0
    total = 0
    offset = 0
    for group_size in groups:
        group_labels = labels[offset:offset + group_size]
        group_scores = scores[offset:offset + group_size]

        if np.sum(group_labels > 0) == 0:
            offset += group_size
            continue

        order = np.argsort(-group_scores)
        sorted_labels = group_labels[order]
        if np.any(sorted_labels[:k] > 0):
            hits += 1
        total += 1
        offset += group_size

    return hits / total if total > 0 else 0.0


def mean_reciprocal_rank(labels: np.ndarray, scores: np.ndarray, groups: list[int]) -> float:
    """Compute MRR: average of 1/rank of the first positive result."""
    rr_scores = []
    offset = 0
    for group_size in groups:
        group_labels = labels[offset:offset + group_size]
        group_scores = scores[offset:offset + group_size]

        if np.sum(group_labels > 0) == 0:
            offset += group_size
            continue

        order = np.argsort(-group_scores)
        sorted_labels = group_labels[order]
        for rank, l in enumerate(sorted_labels, 1):
            if l > 0:
                rr_scores.append(1.0 / rank)
                break

        offset += group_size

    return float(np.mean(rr_scores)) if rr_scores else 0.0


def evaluate_model(
    name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    groups: list[int],
) -> dict:
    """Run all eval metrics and return as dict."""
    metrics = {
        "model": name,
        "ndcg@5": ndcg_at_k(labels, scores, groups, k=5),
        "ndcg@10": ndcg_at_k(labels, scores, groups, k=10),
        "ndcg@20": ndcg_at_k(labels, scores, groups, k=20),
        "recall@10": recall_at_k(labels, scores, groups, k=10),
        "recall@50": recall_at_k(labels, scores, groups, k=50),
        "hr@10": hit_rate_at_k(labels, scores, groups, k=10),
        "mrr": mean_reciprocal_rank(labels, scores, groups),
    }
    return metrics


# ── Main Training Pipeline ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train LightGBM lambdarank reranker for ResearchIT"
    )
    parser.add_argument("--train-file", required=True, help="train.parquet from Step 2")
    parser.add_argument("--eval-file", required=True, help="eval.parquet from Step 2")
    parser.add_argument("--output-dir", default="./model_output")
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-data-in-leaf", type=int, default=50)
    parser.add_argument("--feature-fraction", type=float, default=0.8)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    print("=" * 60)
    print("Loading training data...")
    train_features, train_labels, train_groups, train_qids = load_ltr_data(args.train_file)
    print(f"  Train: {len(train_labels)} rows, {len(train_groups)} queries")
    print(f"  Label distribution: 0={np.sum(train_labels==0)}, 1={np.sum(train_labels==1)}, 2={np.sum(train_labels==2)}")

    print("\nLoading eval data...")
    eval_features, eval_labels, eval_groups, eval_qids = load_ltr_data(args.eval_file)
    print(f"  Eval: {len(eval_labels)} rows, {len(eval_groups)} queries")
    print(f"  Label distribution: 0={np.sum(eval_labels==0)}, 1={np.sum(eval_labels==1)}, 2={np.sum(eval_labels==2)}")

    # Verify time split: no overlap between train and eval query IDs
    train_query_set = set(train_qids)
    eval_query_set = set(eval_qids)
    overlap = train_query_set & eval_query_set
    if overlap:
        print(f"  WARNING: {len(overlap)} query IDs appear in both splits!")
    else:
        print(f"  ✅ No query overlap between train/eval splits")

    # ── Baseline: heuristic scorer ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("Evaluating heuristic baseline...")

    baseline_scores = heuristic_baseline_score(eval_features)
    baseline_metrics = evaluate_model("heuristic_baseline", eval_labels, baseline_scores, eval_groups)

    print(f"\n  Heuristic Baseline Results:")
    for k, v in baseline_metrics.items():
        if k != "model":
            print(f"    {k}: {v:.4f}")

    # ── Train LightGBM ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training LightGBM lambdarank...")

    train_dataset = lgb.Dataset(
        train_features,
        label=train_labels,
        group=train_groups,
        feature_name=FEATURE_SCHEMA,
        free_raw_data=False,
    )

    eval_dataset = lgb.Dataset(
        eval_features,
        label=eval_labels,
        group=eval_groups,
        feature_name=FEATURE_SCHEMA,
        reference=train_dataset,
        free_raw_data=False,
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5, 10, 20],
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambdarank_truncation_level": 20,
        "verbose": 1,
        "seed": 42,
        "num_threads": os.cpu_count() or 4,
    }

    print(f"\n  Parameters:")
    for k, v in params.items():
        print(f"    {k}: {v}")

    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=args.early_stopping_rounds),
    ]

    t0 = time.time()
    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=args.num_boost_round,
        valid_sets=[eval_dataset],
        valid_names=["eval"],
        callbacks=callbacks,
    )
    train_time = time.time() - t0

    print(f"\n  Training completed in {train_time:.1f}s")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best nDCG@10: {model.best_score.get('eval', {}).get('ndcg@10', 'N/A')}")

    # ── Evaluate LightGBM ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Evaluating LightGBM on eval set...")

    lgb_scores = model.predict(eval_features)
    lgb_metrics = evaluate_model("lightgbm_lambdarank", eval_labels, lgb_scores, eval_groups)

    print(f"\n  LightGBM Results:")
    for k, v in lgb_metrics.items():
        if k != "model":
            print(f"    {k}: {v:.4f}")

    # ── Comparison ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON: LightGBM vs Heuristic Baseline")
    print("-" * 50)
    print(f"  {'Metric':<15} {'Heuristic':>12} {'LightGBM':>12} {'Δ':>10} {'%Δ':>8}")
    print("-" * 50)

    comparison = {}
    for metric_key in ["ndcg@5", "ndcg@10", "ndcg@20", "recall@10", "recall@50", "hr@10", "mrr"]:
        b = baseline_metrics[metric_key]
        l = lgb_metrics[metric_key]
        delta = l - b
        pct = (delta / b * 100) if b > 0 else float('inf')
        comparison[metric_key] = {
            "heuristic": round(b, 4),
            "lightgbm": round(l, 4),
            "delta": round(delta, 4),
            "pct_improvement": round(pct, 2),
        }
        marker = "✅" if delta > 0 else "⚠️" if delta == 0 else "❌"
        print(f"  {metric_key:<15} {b:>12.4f} {l:>12.4f} {delta:>+10.4f} {pct:>+7.1f}% {marker}")

    print("-" * 50)

    # ── Feature Importance ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Feature Importance (top 20):")

    importance = model.feature_importance(importance_type="gain")
    importance_pairs = sorted(
        zip(FEATURE_SCHEMA, importance),
        key=lambda x: x[1],
        reverse=True,
    )

    print(f"  {'Rank':<6} {'Feature':<35} {'Importance':>12}")
    print("-" * 55)
    for rank, (fname, imp) in enumerate(importance_pairs[:20], 1):
        bar = "█" * int(imp / max(importance) * 30) if max(importance) > 0 else ""
        print(f"  {rank:<6} {fname:<35} {imp:>12.1f}  {bar}")

    # Zero-importance features (expected: user behavior features 20-30)
    zero_features = [fname for fname, imp in importance_pairs if imp == 0]
    if zero_features:
        print(f"\n  Zero-importance features ({len(zero_features)}):")
        for fname in zero_features:
            print(f"    - {fname}")

    # ── Inference latency benchmark ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Inference Latency Benchmark:")

    # Simulate production: 100 candidates per query
    test_batch = eval_features[:100] if len(eval_features) >= 100 else eval_features

    # Warm up
    for _ in range(10):
        model.predict(test_batch)

    # Benchmark
    n_iters = 1000
    t0 = time.time()
    for _ in range(n_iters):
        model.predict(test_batch)
    total_ms = (time.time() - t0) * 1000
    per_call_ms = total_ms / n_iters

    print(f"  {len(test_batch)} candidates × {n_iters} iterations")
    print(f"  Total: {total_ms:.1f}ms")
    print(f"  Per call: {per_call_ms:.3f}ms")
    print(f"  Target: <1ms for 100 candidates → {'✅ PASS' if per_call_ms < 1.0 else '⚠️ SLOW'}")

    # ── Save outputs ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Saving outputs...")

    # Model
    model_path = output_dir / "reranker_v1.txt"
    model.save_model(str(model_path))
    model_size_kb = os.path.getsize(model_path) / 1024
    print(f"  Model: {model_path} ({model_size_kb:.1f} KB)")

    # Eval metrics
    metrics_path = output_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "baseline": baseline_metrics,
            "lightgbm": lgb_metrics,
            "comparison": comparison,
            "training": {
                "num_boost_round": args.num_boost_round,
                "best_iteration": model.best_iteration,
                "training_time_seconds": round(train_time, 1),
                "train_rows": len(train_labels),
                "train_queries": len(train_groups),
                "eval_rows": len(eval_labels),
                "eval_queries": len(eval_groups),
                "params": params,
            },
            "latency": {
                "candidates": len(test_batch),
                "per_call_ms": round(per_call_ms, 3),
                "target_ms": 1.0,
                "pass": per_call_ms < 1.0,
            },
            "feature_importance": [
                {"feature": fname, "importance": float(imp)}
                for fname, imp in importance_pairs
            ],
        }, f, indent=2)
    print(f"  Metrics: {metrics_path}")

    # Feature importance CSV
    fi_path = output_dir / "feature_importance.csv"
    with open(fi_path, "w") as f:
        f.write("rank,feature,importance\n")
        for rank, (fname, imp) in enumerate(importance_pairs, 1):
            f.write(f"{rank},{fname},{imp}\n")
    print(f"  Feature importance: {fi_path}")

    # Baseline comparison
    comp_path = output_dir / "baseline_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"  Comparison: {comp_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    primary_metric = "ndcg@10"
    b = baseline_metrics[primary_metric]
    l = lgb_metrics[primary_metric]
    delta = l - b
    pct = (delta / b * 100) if b > 0 else 0

    if delta > 0.03:
        verdict = "✅ STRONG IMPROVEMENT — deploy LightGBM"
    elif delta > 0:
        verdict = "⚠️ MARGINAL IMPROVEMENT — consider if complexity is worth it"
    else:
        verdict = "❌ NO IMPROVEMENT — keep heuristic, investigate features"

    print(f"PRIMARY METRIC: nDCG@10")
    print(f"  Heuristic: {b:.4f}")
    print(f"  LightGBM:  {l:.4f} ({delta:+.4f}, {pct:+.1f}%)")
    print(f"  Verdict:   {verdict}")
    print(f"\nModel file:  {model_path}")
    print(f"Model size:  {model_size_kb:.1f} KB")
    print(f"Latency:     {per_call_ms:.3f}ms per 100 candidates")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
