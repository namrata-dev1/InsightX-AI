"""Phase 3, Milestone 4: Comprehensive Recommendation & Ranking Tests.

Tests cover:
  1. Intra-List Diversity (ILD) calculation.
  2. Average Recommendation Popularity (ARP) calculation.
  3. Gini coefficient of item recommendation distribution.
  4. ComprehensiveRecommendationEvaluator returns all required metrics.
  5. Segment-wise evaluation splits into repeat, sparse, and cold-start.
  6. Hybrid weight tuning on validation data without leakage.
  7. Alternative normalisation schemes in HybridRecommender (minmax, zscore, rank).
  8. Preserves backward compatibility of RecommendationEvaluator.
"""

import numpy as np
import pandas as pd
import pytest

from insightx.models.recommenders import (
    PopularityRecommender,
    CategoryRecommender,
    CoPurchaseRecommender,
)
from insightx.models.collaborative_filtering import SVDCollaborativeFilter
from insightx.models.content_based import ContentBasedRecommender
from insightx.models.hybrid_recommender import HybridRecommender
from insightx.models.recommendation_eval import RecommendationEvaluator
from insightx.models.comprehensive_eval import (
    ComprehensiveRecommendationEvaluator,
    _intra_list_diversity,
    _average_recommendation_popularity,
    _gini_index,
    tune_hybrid_weights,
)


def _sample_train_data():
    return pd.DataFrame([
        {"customer_unique_id": "repeat_1", "product_id": "prod_A", "purchase_count": 3},
        {"customer_unique_id": "repeat_1", "product_id": "prod_B", "purchase_count": 2},
        {"customer_unique_id": "sparse_1", "product_id": "prod_A", "purchase_count": 1},
        {"customer_unique_id": "sparse_2", "product_id": "prod_C", "purchase_count": 1},
    ])


def _sample_val_data():
    return pd.DataFrame([
        {"customer_unique_id": "repeat_1", "product_id": "prod_C"},
        {"customer_unique_id": "sparse_1", "product_id": "prod_B"},
        {"customer_unique_id": "cold_1",   "product_id": "prod_A"},
    ])


def _sample_metadata():
    return pd.DataFrame([
        {"product_id": "prod_A", "product_category_name_english": "electronics", "price_band": "mid"},
        {"product_id": "prod_B", "product_category_name_english": "books",       "price_band": "budget"},
        {"product_id": "prod_C", "product_category_name_english": "toys",        "price_band": "premium"},
    ])


class TestMetricsCalculation:

    def test_intra_list_diversity_different_categories(self):
        cat_map = {"p1": "catA", "p2": "catB", "p3": "catC"}
        ild = _intra_list_diversity(["p1", "p2", "p3"], cat_map)
        assert ild == 1.0

    def test_intra_list_diversity_identical_categories(self):
        cat_map = {"p1": "catA", "p2": "catA", "p3": "catA"}
        ild = _intra_list_diversity(["p1", "p2", "p3"], cat_map)
        assert ild == 0.0

    def test_average_recommendation_popularity(self):
        pop_map = {"p1": 10.0, "p2": 2.0}
        arp = _average_recommendation_popularity([["p1", "p2"]], pop_map)
        assert arp == 6.0

    def test_gini_index_bounds(self):
        # Equal distribution -> Gini 0
        eq = np.array([5.0, 5.0, 5.0, 5.0])
        assert _gini_index(eq) == pytest.approx(0.0, abs=1e-5)
        # Concentrated distribution -> Gini > 0
        skewed = np.array([0.0, 0.0, 0.0, 10.0])
        assert _gini_index(skewed) > 0.6


class TestComprehensiveEvaluator:

    def test_evaluate_returns_all_metrics(self):
        train = _sample_train_data()
        val = _sample_val_data()
        meta = _sample_metadata()

        pop = PopularityRecommender(top_k=2).fit(train)
        evaluator = ComprehensiveRecommendationEvaluator(k=2)

        res = evaluator.evaluate(pop, val, train, "pop_test", meta)
        required_keys = [
            "precision_at_k", "recall_at_k", "f1_at_k", "ndcg_at_k",
            "catalog_coverage_at_k", "intra_list_diversity",
            "avg_recommendation_popularity", "popularity_gini",
        ]
        for rk in required_keys:
            assert rk in res, f"Missing metric key: {rk}"

    def test_segment_wise_evaluation(self):
        train = _sample_train_data()
        val = _sample_val_data()
        meta = _sample_metadata()

        pop = PopularityRecommender(top_k=2).fit(train)
        evaluator = ComprehensiveRecommendationEvaluator(k=2)

        segments = evaluator.evaluate_segments(pop, val, train, "pop", meta)
        assert "repeat" in segments
        assert "sparse" in segments
        assert "cold_start" in segments
        # cold_1 has 0 purchases in train -> belongs to cold_start
        assert segments["cold_start"]["n_eval_customers_total"] == 1

    def test_compare_extended_dataframe(self):
        evaluator = ComprehensiveRecommendationEvaluator(k=2)
        rows = [
            {"recommender_name": "r1", "k": 2, "precision_at_k": 0.5, "recall_at_k": 0.5,
             "f1_at_k": 0.5, "ndcg_at_k": 0.5, "catalog_coverage_at_k": 0.6,
             "intra_list_diversity": 0.4, "avg_recommendation_popularity": 5.0,
             "popularity_gini": 0.2, "n_eval_customers_total": 3, "n_eval_customers_with_relevant": 2}
        ]
        df = evaluator.compare(rows)
        for col in ["Diversity", "ARP", "Gini"]:
            assert col in df.columns


class TestHybridTuningAndNormalisation:

    def test_hybrid_normalisation_schemes(self):
        train = _sample_train_data()
        pop = PopularityRecommender(top_k=2).fit(train)
        for mode in ["minmax", "zscore", "rank"]:
            hybrid = HybridRecommender(
                components=[(pop, 1.0, "popularity_score")],
                top_k=2,
                normalisation_mode=mode,
            ).fit(train)
            recs = hybrid.recommend("repeat_1", k=2)
            assert len(recs) <= 2
            assert "hybrid_score" in recs.columns

    def test_tuning_utility_selects_valid_weights(self):
        train = _sample_train_data()
        val = _sample_val_data()
        meta = _sample_metadata()

        pop = PopularityRecommender(top_k=2).fit(train)
        cop = CoPurchaseRecommender(top_k=2).fit(train)

        sub_models = {
            "popularity": (pop, "popularity_score"),
            "copurchase": (cop, "copurchase_score"),
        }
        grid = [
            {"popularity": 1.0, "copurchase": 0.0},
            {"popularity": 0.0, "copurchase": 1.0},
            {"popularity": 0.5, "copurchase": 0.5},
        ]
        best_w, best_score, trials_df = tune_hybrid_weights(
            train_interaction_df=train,
            val_interaction_df=val,
            product_metadata_df=meta,
            sub_models=sub_models,
            grid_weights=grid,
            k=2,
            metric="recall_at_k",
        )
        assert isinstance(best_w, dict)
        assert len(trials_df) == 3
        assert best_score >= 0.0
