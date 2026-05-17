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
import math
from dataclasses import dataclass, field
import numpy as np
from scipy.cluster.hierarchy import ward, fcluster
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist

from app import db

# Ward merge threshold — used as a MAXIMUM. The actual cut point is
# determined adaptively by finding the largest gap in merge distances.
# This fallback only applies if no clear gap is found.
WARD_DISTANCE_THRESHOLD = 1.5

# Absolute limits on cluster count
MIN_CLUSTERS = 1
MAX_CLUSTERS = 7   # RFC: PinnerSage uses 3-5 for typical users, cap at 7

# Average papers per cluster floor — used to derive a soft cap on K from N.
# K_soft_cap = max(MIN_CLUSTERS, ceil(N / AVG_CLUSTER_SIZE_FLOOR)).
# Set to 4: at N=5 -> K_max=2, at N=10 -> K_max=3, at N=28 -> K_max=7.
# Without this, gap-based thresholding over-splits at small N: 5 same-domain
# papers were producing K=4 (3 singletons), which then got over-weighted by
# the quota floor of 3 slots per cluster.
AVG_CLUSTER_SIZE_FLOOR = 4

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

    # Clamp cluster count.
    # Two layers:
    #   1. Hard cap: never exceed MAX_CLUSTERS (=7) regardless of N.
    #   2. Soft cap: keep average cluster size >= AVG_CLUSTER_SIZE_FLOOR.
    #      This prevents the gap-detection from over-splitting small N
    #      (e.g. 5 same-domain saves were producing K=4 with 3 singletons,
    #      which then got over-weighted by the quota floor of 3 slots).
    soft_cap = max(
        MIN_CLUSTERS,
        min(MAX_CLUSTERS, math.ceil(n / AVG_CLUSTER_SIZE_FLOOR)),
    )

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters > MAX_CLUSTERS:
        labels = fcluster(linkage, t=MAX_CLUSTERS, criterion="maxclust")
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

    if n_clusters > soft_cap:
        labels = fcluster(linkage, t=soft_cap, criterion="maxclust")
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

    # Final safety net: merge any remaining singleton clusters into their
    # nearest non-singleton neighbour. The soft cap usually eliminates them,
    # but a 6-1-1-1 distribution after maxclust=4 would still leave 3.
    labels = _merge_singletons(labels, embeddings)
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


def _merge_singletons(labels: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Merge singleton clusters into their nearest non-singleton cluster.

    Why: Ward's gap-based threshold can over-split at small N, producing
    1-paper clusters that get over-weighted by the quota floor (3 slots
    per cluster regardless of importance). Merging singletons into the
    nearest non-singleton cluster preserves the multi-interest signal
    where it's real and removes spurious singletons where it's noise.

    Edge case: if every cluster is a singleton (all papers maximally
    distant), we leave the labels alone — collapsing them would erase
    a genuine multi-interest signal.
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    singleton_labels = unique_labels[counts == 1]
    non_singleton_labels = unique_labels[counts > 1]

    if len(singleton_labels) == 0:
        return labels  # nothing to merge
    if len(non_singleton_labels) == 0:
        return labels  # all singletons — keep as is

    centroids: dict[int, np.ndarray] = {}
    for ns_label in non_singleton_labels:
        ns_mask = labels == ns_label
        centroids[int(ns_label)] = embeddings[ns_mask].mean(axis=0)

    new_labels = labels.copy()
    for s_label in singleton_labels:
        s_idx = int(np.where(labels == s_label)[0][0])
        s_emb = embeddings[s_idx]
        best_label = int(s_label)
        best_dist = float("inf")
        for ns_label, centroid in centroids.items():
            d = float(np.linalg.norm(s_emb - centroid))
            if d < best_dist:
                best_dist = d
                best_label = ns_label
        new_labels[s_idx] = best_label

    return new_labels


# ── Cluster ID stabilisation (Phase 4.2) ─────────────────────────────────────

# Hungarian matches below this cosine similarity are rejected as "unrelated".
# Doc 06 §"Clustering specifics": a genuinely new interest must not steal an
# old cluster's identity just because Hungarian found the least-bad assignment.
CLUSTER_MATCH_MIN_COSINE = 0.5


def stabilize_cluster_ids(
    new_clusters: list[InterestCluster],
    old_clusters: list[InterestCluster],
    min_cosine_sim: float = CLUSTER_MATCH_MIN_COSINE,
) -> list[InterestCluster]:
    """
    Preserve cluster identity across reclusters using the Hungarian algorithm.

    Every time the user saves a paper we recluster from scratch.  Without
    stabilisation, cluster indices shuffle (NLP was 0, now it's 2), breaking
    future analytics and UI labels.

    Algorithm:
      1. Build cost matrix: cost[i][j] = 1 - cosine_sim(new_medoid_i, old_medoid_j)
      2. Solve with scipy linear_sum_assignment (O(K³), trivial for K ≤ 7)
      3. Matched pairs with cosine_sim >= min_cosine_sim inherit the old idx
      4. Weak matches (cosine_sim < min_cosine_sim) and unmatched new clusters
         get the next available index

    Args:
        new_clusters:   freshly computed clusters (cluster_idx values ignored)
        old_clusters:   clusters from the previous recluster (stable reference)
        min_cosine_sim: reject matches below this cosine similarity (default 0.5)

    Returns:
        new_clusters with stable cluster_idx values assigned.
    """
    if not old_clusters or not new_clusters:
        return new_clusters

    new_embs = np.array([c.medoid_embedding for c in new_clusters], dtype=np.float32)
    old_embs = np.array([c.medoid_embedding for c in old_clusters], dtype=np.float32)

    # L2-normalise before cosine similarity
    def _safe_norm(embs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.where(norms < 1e-10, 1.0, norms)

    new_embs = _safe_norm(new_embs)
    old_embs = _safe_norm(old_embs)

    # Cosine similarity → cost matrix (n_new × n_old)
    sim = new_embs @ old_embs.T
    cost = 1.0 - sim

    # Hungarian assignment — works on rectangular matrices
    row_ind, col_ind = linear_sum_assignment(cost)

    # Accept only pairs whose cosine similarity clears the threshold.
    # Weak matches would steal an old cluster's identity for an unrelated topic.
    new_to_stable: dict[int, int] = {}
    for r, c in zip(row_ind, col_ind):
        if float(sim[r, c]) >= min_cosine_sim:
            new_to_stable[int(r)] = old_clusters[int(c)].cluster_idx

    used_ids: set[int] = set(new_to_stable.values())
    next_id = 0

    result: list[InterestCluster] = []
    for i, cluster in enumerate(new_clusters):
        if i in new_to_stable:
            stable_idx = new_to_stable[i]
        else:
            # No strong match — assign next free index
            while next_id in used_ids:
                next_id += 1
            stable_idx = next_id
            used_ids.add(stable_idx)
            next_id += 1

        result.append(InterestCluster(
            cluster_idx=stable_idx,
            medoid_paper_id=cluster.medoid_paper_id,
            medoid_embedding=cluster.medoid_embedding,
            paper_ids=cluster.paper_ids,
            importance=cluster.importance,
        ))

    return result


# ── Persistence ───────────────────────────────────────────────────────────────

async def save_clusters_to_db(user_id: str, clusters: list[InterestCluster]) -> None:
    """Persist clusters to SQLite (including medoid embedding for Bug B fallback)."""
    rows = [
        {
            "cluster_idx": c.cluster_idx,
            "medoid_paper_id": c.medoid_paper_id,
            "importance": c.importance,
            "paper_ids": json.dumps(c.paper_ids),
            "medoid_embedding_blob": c.medoid_embedding.astype(np.float32).tobytes(),
        }
        for c in clusters
    ]
    await db.save_user_clusters(user_id, rows)


async def load_clusters_from_db(user_id: str) -> list[dict] | None:
    """Load clusters from SQLite.  Returns None if no clusters exist."""
    rows = await db.get_user_clusters(user_id)
    return rows if rows else None
