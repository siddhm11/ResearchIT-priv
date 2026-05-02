"""
Quick-start: Load and use the ResearchIT Phase 6 reranker.

Usage:
    python load_model.py

Or import in your code:
    from load_model import load_reranker, predict_scores
"""
import numpy as np

def load_reranker(model_path: str = "production_model/reranker_v1.txt"):
    """Load the LightGBM reranker model."""
    import lightgbm as lgb
    model = lgb.Booster(model_file=model_path)
    assert model.num_feature() == 37, f"Expected 37 features, got {model.num_feature()}"
    return model

def predict_scores(model, features: np.ndarray) -> np.ndarray:
    """
    Predict reranking scores for candidates.
    
    Args:
        model: LightGBM Booster
        features: (N, 37) float32 array — see feature_schema.json for column order
        
    Returns:
        (N,) float64 array — higher score = more relevant
    """
    assert features.shape[1] == 37, f"Expected 37 features, got {features.shape[1]}"
    return model.predict(features)

# Feature schema (must match this exact order)
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

if __name__ == "__main__":
    model = load_reranker()
    print(f"Model loaded: {model.num_trees()} trees, {model.num_feature()} features")
    
    # Test with dummy input
    dummy = np.zeros((10, 37), dtype=np.float32)
    scores = predict_scores(model, dummy)
    print(f"Dummy scores: {scores[:5]}")
    print("✅ Model works!")
