"""
Ward hierarchical clustering for multi-interest detection.

Discovers K distinct interest clusters from the user's saved paper
embeddings using Ward's method.  K is determined automatically by
a distance threshold — not predefined.

Each cluster is represented by its **medoid** (the actual paper
embedding closest to cluster center), not the centroid.  This prevents
"topic drift" into meaningless regions of embedding space.

Reference: Research-MultiInterest_Recommender_Architecture.md §2
  "PinnerSage's design choices: Ward hierarchical clustering on the
   user's interacted item embeddings, with a threshold parameter α
   controlling merge stopping — this automatically determines K per user"
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
import numpy as np
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist

from app import db

# Ward merge threshold — used as a MAXIMUM. The actual cut point is
# determined adaptively by finding the largest gap in merge distances.
# This fallback only applies if no clear gap is found.
WARD_DISTANCE_THRESHOLD = 1.5

# Absolute limits on cluster count
MIN_CLUSTERS = 1
MAX_CLUSTERS = 7   # RFC: PinnerSage uses 3-5 for typical users, cap at 7

# Minimum saved papers before clustering is meaningful
MIN_PAPERS_FOR_CLUSTERING = 5


@dataclass
class InterestCluster:
    """A single interest cluster derived from user behaviour."""
    cluster_idx: int
    medoid_paper_id: str
    medoid_embedding: np.ndarray
    paper_ids: list[str]
    importance: float  # recency-weighted sum of interactions


def _adaptive_threshold(linkage: np.ndarray) -> float:
    """
    Find the optimal cut point by detecting the largest gap in merge distances.

    The linkage matrix has shape (n-1, 4). Column 2 is the merge distance.
    We look at the differences between consecutive merge distances and pick
    the cut just below the biggest jump — that's where the most distinct
    clusters separate.

    Falls back to 0.7 × max_merge_distance if no clear gap is found.
    """
    merge_distances = linkage[:, 2]
    if len(merge_distances) < 2:
        return float(merge_distances[0]) if len(merge_distances) == 1 else WARD_DISTANCE_THRESHOLD

    # Compute gaps between consecutive merges
    gaps = np.diff(merge_distances)

    # The biggest gap indicates the most natural cluster boundary
    best_gap_idx = int(np.argmax(gaps))

    # Cut just above the merge BEFORE the biggest gap
    # (i.e., allow all merges up to but not including the big jump)
    threshold = float(merge_distances[best_gap_idx] + merge_distances[best_gap_idx + 1]) / 2.0

    # Sanity: don't let it go below first merge or above reasonable max
    min_t = float(merge_distances[0])
    max_t = float(merge_distances[-1]) * 0.7
    threshold = max(min_t, min(threshold, max_t))

    return threshold


def compute_clusters(
    paper_ids: list[str],
    embeddings: np.ndarray,
    timestamps: list[str] | None = None,
) -> list[InterestCluster]:
    """
    Cluster paper embeddings using Ward's hierarchical method.

    Args:
        paper_ids: arxiv_ids of saved papers (most recent first)
        embeddings: shape (N, 1024), the BGE-M3 vectors for each paper
        timestamps: optional ISO timestamps for recency weighting

    Returns:
        List of InterestCluster sorted by importance (highest first).
        Returns a single cluster (medoid of all) if N < MIN_PAPERS_FOR_CLUSTERING.
    """
    n = len(paper_ids)
    assert embeddings.shape == (n, 1024), f"Expected ({n}, 1024), got {embeddings.shape}"

    # Too few papers — return a single cluster with the centroid's medoid
    if n < MIN_PAPERS_FOR_CLUSTERING:
        centroid = embeddings.mean(axis=0)
        medoid_idx = _find_medoid(embeddings, centroid)
        return [InterestCluster(
            cluster_idx=0,
            medoid_paper_id=paper_ids[medoid_idx],
            medoid_embedding=embeddings[medoid_idx],
            paper_ids=list(paper_ids),
            importance=float(n),
        )]

    # L2-normalize: Ward is Euclidean-only (Murtagh & Legendre 2014).
    # On unit vectors, ‖a−b‖² = 2(1−cos(a,b)), giving cosine-Ward.
    # BGE-M3 vectors are approximately unit-norm but can drift after
    # EWMA blending or floating-point accumulation. (Doc 06, fault #4)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    embeddings = embeddings / norms

    # Compute pairwise distances and linkage
    # Ward's method minimises intra-cluster variance
    distances = pdist(embeddings, metric="euclidean")
    linkage = ward(distances)

    # Adaptive threshold: find the biggest gap in merge distances
    threshold = _adaptive_threshold(linkage)

    # Cut the dendrogram at the adaptive threshold
    labels = fcluster(linkage, t=threshold, criterion="distance")

    # Clamp cluster count
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # If too many clusters, re-cut with a maxclust constraint
    if n_clusters > MAX_CLUSTERS:
        labels = fcluster(linkage, t=MAX_CLUSTERS, criterion="maxclust")
        unique_labels = np.unique(labels)

    # Compute recency weights (position-based: most recent = highest weight)
    recency_weights = np.array([
        1.0 / (i + 1) for i in range(n)
    ], dtype=np.float64)

    # Build clusters
    clusters = []
    for cidx, label in enumerate(unique_labels):
        mask = labels == label
        cluster_embs = embeddings[mask]
        cluster_ids = [paper_ids[i] for i in range(n) if mask[i]]
        cluster_weights = recency_weights[mask]

        # Centroid for medoid computation
        centroid = cluster_embs.mean(axis=0)
        medoid_local_idx = _find_medoid(cluster_embs, centroid)

        # Importance: sum of recency weights for this cluster's papers
        importance = float(cluster_weights.sum())

        # Find the global paper_id index of the medoid
        medoid_global_indices = np.where(mask)[0]
        medoid_global_idx = medoid_global_indices[medoid_local_idx]

        clusters.append(InterestCluster(
            cluster_idx=cidx,
            medoid_paper_id=paper_ids[medoid_global_idx],
            medoid_embedding=embeddings[medoid_global_idx],
            paper_ids=cluster_ids,
            importance=importance,
        ))

    # Sort by importance (most important first)
    clusters.sort(key=lambda c: c.importance, reverse=True)
    return clusters


def _find_medoid(embeddings: np.ndarray, centroid: np.ndarray) -> int:
    """Find the index of the embedding closest to the centroid."""
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    return int(np.argmin(distances))


# ── Persistence ───────────────────────────────────────────────────────────────

async def save_clusters_to_db(user_id: str, clusters: list[InterestCluster]) -> None:
    """Persist clusters to SQLite."""
    rows = [
        {
            "cluster_idx": c.cluster_idx,
            "medoid_paper_id": c.medoid_paper_id,
            "importance": c.importance,
            "paper_ids": json.dumps(c.paper_ids),
        }
        for c in clusters
    ]
    await db.save_user_clusters(user_id, rows)


async def load_clusters_from_db(user_id: str) -> list[dict] | None:
    """Load clusters from SQLite.  Returns None if no clusters exist."""
    rows = await db.get_user_clusters(user_id)
    return rows if rows else None
