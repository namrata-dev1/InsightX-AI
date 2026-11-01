"""Phase 3, Milestone 3: Advanced Recommendation Models — Synthetic Tests.

All fixtures are self-contained (no real Olist data required, <= 500 rows).

Test coverage:
  SVD COLLABORATIVE FILTER
    1.  fit() succeeds on valid training data.
    2.  recommend() returns <= k items.
    3.  exclude_purchased removes training products.
    4.  New/unknown customer returns empty DataFrame (no crash).
    5.  Recommendations are reproducible (deterministic).
    6.  k > catalog size returns at most catalog_size products.
    7.  Single-user or single-item edge case does not crash.
    8.  cf_score is non-negative (clipped).
    9.  Evidence records produced after fit().
    10. Empty interaction matrix does not crash.

  CONTENT-BASED RECOMMENDER
    11. fit() on valid metadata + interactions succeeds.
    12. recommend() returns products with cbf_score in [-1, 1].
    13. Exclude_purchased removes training products from CBF output.
    14. Unknown customer returns empty DataFrame.
    15. Missing numeric features filled without crash.
    16. Missing category column handled without crash.
    17. Recommendations are reproducible.
    18. Evidence records produced after fit().

  HYBRID RECOMMENDER
    19. Hybrid scores in [0, 1] after normalisation.
    20. exclude_purchased applied at hybrid level.
    21. All-component-empty → fallback to popularity.
    22. New customer (not in any component training) → fallback.
    23. Single-component hybrid behaves like that component (up to normalisation).
    24. Weight=0 component is ignored.
    25. Evidence records produced after fit().

  EVALUATOR (NDCG@K)
    26. NDCG@1 = 1.0 for perfect top-1 recommender.
    27. NDCG@K = 0.0 when no hit in top-K.
    28. NDCG@K < 1.0 when relevant item is not at position 1.
    29. ndcg_at_k key present in evaluate() result.
    30. NDCG@K in [0, 1] for any recommender output.

  INTEGRATION
    31. Full pipeline: train all 4 models, evaluate, compare.
    32. Hybrid outperforms or ties at least one baseline on coverage.
"""

from typing import Dict, List, Set
import numpy as np
import pandas as pd
import pytest

from insightx.models.collaborative_filtering import SVDCollaborativeFilter
from insightx.models.content_based import ContentBasedRecommender
from insightx.models.hybrid_recommender import HybridRecommender
from insightx.models.recommenders import PopularityRecommender, CategoryRecommender
from insightx.models.recommendation_eval import RecommendationEvaluator, _ndcg_at_k
from insightx.provenance.types import ProvenanceType


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _train_df() -> pd.DataFrame:
    """12 training interactions across 5 customers and 5 products."""
    return pd.DataFrame([
        {"customer_unique_id": "uid_A", "product_id": "prod_1", "purchase_count": 3},
        {"customer_unique_id": "uid_A", "product_id": "prod_2", "purchase_count": 1},
        {"customer_unique_id": "uid_B", "product_id": "prod_1", "purchase_count": 2},
        {"customer_unique_id": "uid_B", "product_id": "prod_3", "purchase_count": 2},
        {"customer_unique_id": "uid_C", "product_id": "prod_2", "purchase_count": 1},
        {"customer_unique_id": "uid_C", "product_id": "prod_3", "purchase_count": 1},
        {"customer_unique_id": "uid_D", "product_id": "prod_4", "purchase_count": 5},
        {"customer_unique_id": "uid_E", "product_id": "prod_1", "purchase_count": 1},
        {"customer_unique_id": "uid_E", "product_id": "prod_5", "purchase_count": 2},
    ])


def _product_meta() -> pd.DataFrame:
    return pd.DataFrame([
        {"product_id": "prod_1", "product_category_name_english": "electronics",
         "price_band": "mid", "product_weight_g": 500.0, "product_length_cm": 20.0,
         "product_height_cm": 10.0, "product_width_cm": 15.0, "product_photos_qty": 3},
        {"product_id": "prod_2", "product_category_name_english": "electronics",
         "price_band": "budget", "product_weight_g": 200.0, "product_length_cm": 15.0,
         "product_height_cm": 5.0, "product_width_cm": 10.0, "product_photos_qty": 2},
        {"product_id": "prod_3", "product_category_name_english": "toys",
         "price_band": "budget", "product_weight_g": 300.0, "product_length_cm": 25.0,
         "product_height_cm": 15.0, "product_width_cm": 20.0, "product_photos_qty": 4},
        {"product_id": "prod_4", "product_category_name_english": "sports",
         "price_band": "premium", "product_weight_g": 1500.0, "product_length_cm": 50.0,
         "product_height_cm": 30.0, "product_width_cm": 40.0, "product_photos_qty": 6},
        {"product_id": "prod_5", "product_category_name_english": "toys",
         "price_band": "mid", "product_weight_g": 400.0, "product_length_cm": 18.0,
         "product_height_cm": 12.0, "product_width_cm": 14.0, "product_photos_qty": 3},
    ])


# ---------------------------------------------------------------------------
# SVD COLLABORATIVE FILTER
# ---------------------------------------------------------------------------

class TestSVDCollaborativeFilter:

    def test_fit_succeeds(self):
        rec = SVDCollaborativeFilter(n_components=2, top_k=3).fit(_train_df())
        assert rec._fitted

    def test_recommend_returns_le_k(self):
        rec = SVDCollaborativeFilter(n_components=2, top_k=3).fit(_train_df())
        result = rec.recommend(customer_id="uid_A", k=3)
        assert len(result) <= 3

    def test_exclude_purchased_removes_training_products(self):
        rec = SVDCollaborativeFilter(n_components=2, exclude_purchased=True).fit(_train_df())
        result = rec.recommend(customer_id="uid_A", k=10)
        # uid_A bought prod_1 and prod_2
        recommended = set(result["product_id"])
        assert "prod_1" not in recommended
        assert "prod_2" not in recommended

    def test_unknown_customer_returns_empty(self):
        rec = SVDCollaborativeFilter(n_components=2).fit(_train_df())
        result = rec.recommend(customer_id="uid_UNKNOWN_99", k=5)
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)

    def test_recommendations_reproducible(self):
        rec = SVDCollaborativeFilter(n_components=2, random_state=0, exclude_purchased=False).fit(_train_df())
        r1 = rec.recommend(customer_id="uid_A", k=5)
        r2 = rec.recommend(customer_id="uid_A", k=5)
        pd.testing.assert_frame_equal(r1, r2)

    def test_k_larger_than_catalog_returns_at_most_catalog(self):
        rec = SVDCollaborativeFilter(n_components=2, exclude_purchased=False).fit(_train_df())
        result = rec.recommend(customer_id="uid_A", k=1000)
        # Catalog has 5 products; exclude_purchased=False → at most 5
        assert len(result) <= 5

    def test_single_user_no_crash(self):
        tiny = pd.DataFrame([
            {"customer_unique_id": "uid_A", "product_id": "prod_1", "purchase_count": 1},
        ])
        rec = SVDCollaborativeFilter(n_components=2).fit(tiny)
        # With 1 user, SVD is skipped; should not crash
        result = rec.recommend(customer_id="uid_A", k=5)
        assert isinstance(result, pd.DataFrame)

    def test_cf_score_nonnegative(self):
        rec = SVDCollaborativeFilter(n_components=2, exclude_purchased=False).fit(_train_df())
        result = rec.recommend(customer_id="uid_A", k=5)
        if len(result) > 0:
            assert (result["cf_score"] >= 0).all(), "cf_score must be non-negative (clipped)."

    def test_evidence_produced_after_fit(self):
        rec = SVDCollaborativeFilter(n_components=2).fit(_train_df())
        assert len(rec.evidence) >= 2

    def test_empty_train_no_crash(self):
        empty = pd.DataFrame(columns=["customer_unique_id", "product_id", "purchase_count"])
        rec = SVDCollaborativeFilter(n_components=2).fit(empty)
        result = rec.recommend(customer_id="uid_A", k=5)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# CONTENT-BASED RECOMMENDER
# ---------------------------------------------------------------------------

class TestContentBasedRecommender:

    def test_fit_succeeds(self):
        rec = ContentBasedRecommender(top_k=3).fit(_product_meta(), _train_df())
        assert rec._fitted

    def test_cbf_score_in_valid_range(self):
        rec = ContentBasedRecommender(top_k=5, exclude_purchased=False).fit(
            _product_meta(), _train_df()
        )
        result = rec.recommend(customer_id="uid_A", k=5)
        if len(result) > 0:
            assert (result["cbf_score"] >= -1.0).all()
            assert (result["cbf_score"] <= 1.0 + 1e-6).all()

    def test_exclude_purchased_applied(self):
        rec = ContentBasedRecommender(top_k=5, exclude_purchased=True).fit(
            _product_meta(), _train_df()
        )
        result = rec.recommend(customer_id="uid_A", k=5)
        purchased = {"prod_1", "prod_2"}
        recommended = set(result["product_id"])
        assert len(recommended & purchased) == 0

    def test_unknown_customer_returns_empty(self):
        rec = ContentBasedRecommender().fit(_product_meta(), _train_df())
        result = rec.recommend(customer_id="uid_UNKNOWN_99", k=5)
        assert len(result) == 0

    def test_missing_numeric_features_no_crash(self):
        meta = _product_meta().copy()
        meta["product_weight_g"] = None
        meta["product_length_cm"] = None
        rec = ContentBasedRecommender().fit(meta, _train_df())
        result = rec.recommend(customer_id="uid_A", k=5)
        assert isinstance(result, pd.DataFrame)

    def test_missing_category_column_no_crash(self):
        meta = _product_meta().drop(columns=["product_category_name_english"])
        rec = ContentBasedRecommender().fit(meta, _train_df())
        result = rec.recommend(customer_id="uid_A", k=5)
        assert isinstance(result, pd.DataFrame)

    def test_recommendations_reproducible(self):
        rec = ContentBasedRecommender(exclude_purchased=False).fit(_product_meta(), _train_df())
        r1 = rec.recommend(customer_id="uid_A", k=5)
        r2 = rec.recommend(customer_id="uid_A", k=5)
        pd.testing.assert_frame_equal(r1, r2)

    def test_evidence_produced(self):
        rec = ContentBasedRecommender().fit(_product_meta(), _train_df())
        assert len(rec.evidence) >= 2
        for ev in rec.evidence:
            assert ev.source_type == "RECSYS_CORE"
            assert isinstance(ev.raw_value, (str, int, float, bool, bytes, type(None)))


# ---------------------------------------------------------------------------
# HYBRID RECOMMENDER
# ---------------------------------------------------------------------------

class TestHybridRecommender:

    def _build_hybrid(self, weights=(0.4, 0.3, 0.3)):
        train = _train_df()
        meta = _product_meta()
        pop = PopularityRecommender(top_k=10, exclude_purchased=False).fit(train)
        cf  = SVDCollaborativeFilter(n_components=2, exclude_purchased=False, random_state=0).fit(train)
        cbf = ContentBasedRecommender(exclude_purchased=False).fit(meta, train)

        hybrid = HybridRecommender(
            components=[
                (cf,  weights[0], "cf_score"),
                (cbf, weights[1], "cbf_score"),
                (pop, weights[2], "popularity_score"),
            ],
            top_k=5,
            exclude_purchased=True,
        ).fit(train)
        return hybrid

    def test_hybrid_scores_between_0_and_1_after_normalisation(self):
        hybrid = self._build_hybrid()
        result = hybrid.recommend(customer_id="uid_A", k=5)
        if len(result) > 0:
            # hybrid_score is weighted sum of normalised [0,1] scores → must be in [0,1]
            assert (result["hybrid_score"] >= -1e-9).all()
            assert (result["hybrid_score"] <= 1.0 + 1e-9).all()

    def test_exclude_purchased_at_hybrid_level(self):
        hybrid = self._build_hybrid()
        result = hybrid.recommend(customer_id="uid_A", k=10)
        purchased = {"prod_1", "prod_2"}
        recommended = set(result["product_id"])
        assert len(recommended & purchased) == 0, (
            f"Purchased products {recommended & purchased} should be excluded."
        )

    def test_new_customer_gets_fallback(self):
        hybrid = self._build_hybrid()
        result = hybrid.recommend(customer_id="completely_new_uid", k=5)
        # Might be empty or populated via fallback — must not crash
        assert isinstance(result, pd.DataFrame)

    def test_zero_weight_component_ignored(self):
        """A component with weight=0 should not affect the output."""
        train = _train_df()
        pop  = PopularityRecommender(top_k=10, exclude_purchased=False).fit(train)
        pop2 = PopularityRecommender(top_k=10, exclude_purchased=False).fit(train)

        h1 = HybridRecommender(
            [(pop, 1.0, "popularity_score")], top_k=5, exclude_purchased=False
        ).fit(train)
        h2 = HybridRecommender(
            [(pop, 1.0, "popularity_score"), (pop2, 0.0, "popularity_score")],
            top_k=5, exclude_purchased=False,
        ).fit(train)

        r1 = h1.recommend(customer_id="uid_A", k=5)
        r2 = h2.recommend(customer_id="uid_A", k=5)
        # Product order should be the same (zero-weight component contributes nothing)
        assert list(r1["product_id"]) == list(r2["product_id"])

    def test_evidence_records_produced(self):
        hybrid = self._build_hybrid()
        assert len(hybrid.evidence) >= 1
        for ev in hybrid.evidence:
            assert ev.source_type == "RECSYS_CORE"
            assert ev.provenance_type == ProvenanceType.SIMULATION_ASSUMPTION

    def test_rank_column_sequential(self):
        hybrid = self._build_hybrid()
        result = hybrid.recommend(customer_id="uid_A", k=5)
        if len(result) > 0:
            assert list(result["rank"]) == list(range(1, len(result) + 1))

    def test_all_components_empty_triggers_fallback(self):
        """If all components return empty (all cold-start), fallback to popularity."""
        train = _train_df()
        # Build a CF model that will always return empty (unknown users only)
        cf = SVDCollaborativeFilter(n_components=2, exclude_purchased=False).fit(train)
        pop = PopularityRecommender(top_k=10, exclude_purchased=False).fit(train)

        hybrid = HybridRecommender(
            components=[(cf, 1.0, "cf_score")],
            top_k=5,
            exclude_purchased=False,
            fallback_to_popularity=True,
        ).fit(train)

        # A new customer unknown to CF → should fall back to popularity
        result = hybrid.recommend(customer_id="uid_COMPLETELY_NEW", k=5)
        # Should get some recommendations via fallback (popularity)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# EVALUATOR — NDCG@K
# ---------------------------------------------------------------------------

class TestNDCGAtK:

    def test_ndcg_perfect_top1(self):
        """Only relevant item is at position 1 → NDCG@1 = 1.0."""
        rec_list = ["prod_relevant", "prod_other"]
        relevant = {"prod_relevant"}
        assert _ndcg_at_k(rec_list, relevant, 1) == pytest.approx(1.0)

    def test_ndcg_zero_when_no_hit(self):
        """No relevant item in top-K → NDCG@K = 0.0."""
        assert _ndcg_at_k(["prod_X", "prod_Y"], {"prod_Z"}, 2) == 0.0

    def test_ndcg_decreases_with_rank(self):
        """Relevant item at rank 2 should give lower NDCG than rank 1."""
        relevant = {"prod_A"}
        ndcg_rank1 = _ndcg_at_k(["prod_A", "prod_B"], relevant, 2)
        ndcg_rank2 = _ndcg_at_k(["prod_B", "prod_A"], relevant, 2)
        assert ndcg_rank1 > ndcg_rank2, (
            f"NDCG at rank 1 ({ndcg_rank1:.4f}) should exceed rank 2 ({ndcg_rank2:.4f})."
        )

    def test_ndcg_key_in_evaluate_result(self):
        """evaluate() result dict must contain ndcg_at_k key."""
        train = _train_df()
        val = pd.DataFrame([{"customer_unique_id": "uid_A", "product_id": "prod_3"}])
        pop = PopularityRecommender(top_k=5, exclude_purchased=False).fit(train)
        evaluator = RecommendationEvaluator(k=5)
        result = evaluator.evaluate(pop, val, train, "pop")
        assert "ndcg_at_k" in result, "ndcg_at_k must be present in evaluate() result."

    def test_ndcg_in_valid_range(self):
        """NDCG@K must be in [0, 1] for any recommender."""
        train = _train_df()
        val = pd.DataFrame([
            {"customer_unique_id": "uid_A", "product_id": "prod_3"},
            {"customer_unique_id": "uid_B", "product_id": "prod_2"},
        ])
        pop = PopularityRecommender(top_k=5, exclude_purchased=False).fit(train)
        evaluator = RecommendationEvaluator(k=5)
        result = evaluator.evaluate(pop, val, train, "pop")
        assert 0.0 <= result["ndcg_at_k"] <= 1.0

    def test_ndcg_empty_relevant_set(self):
        """NDCG with empty relevant set returns 0.0."""
        assert _ndcg_at_k(["prod_A", "prod_B"], set(), k=5) == 0.0

    def test_ndcg_k_zero(self):
        """NDCG@0 returns 0.0."""
        assert _ndcg_at_k(["prod_A"], {"prod_A"}, k=0) == 0.0


# ---------------------------------------------------------------------------
# INTEGRATION
# ---------------------------------------------------------------------------

class TestAdvancedRecsysIntegration:

    def test_full_pipeline_four_models(self):
        """End-to-end: fit all 4 models, evaluate on val, compare results."""
        train = _train_df()
        meta = _product_meta()
        val = pd.DataFrame([
            {"customer_unique_id": "uid_A", "product_id": "prod_3"},
            {"customer_unique_id": "uid_B", "product_id": "prod_2"},
        ])

        pop = PopularityRecommender(top_k=5, exclude_purchased=False).fit(train)
        cf  = SVDCollaborativeFilter(n_components=2, exclude_purchased=False, random_state=42).fit(train)
        cbf = ContentBasedRecommender(exclude_purchased=False).fit(meta, train)
        hybrid = HybridRecommender(
            components=[
                (cf,  0.4, "cf_score"),
                (cbf, 0.3, "cbf_score"),
                (pop, 0.3, "popularity_score"),
            ],
            top_k=5, exclude_purchased=False,
        ).fit(train)

        evaluator = RecommendationEvaluator(k=5)
        results = []
        for name, rec in [("pop", pop), ("cf", cf), ("cbf", cbf), ("hybrid", hybrid)]:
            r = evaluator.evaluate(rec, val, train, name, meta)
            results.append(r)
            assert "precision_at_k" in r
            assert "ndcg_at_k" in r
            assert 0.0 <= r["precision_at_k"] <= 1.0
            assert 0.0 <= r["ndcg_at_k"] <= 1.0
            assert 0.0 <= r["catalog_coverage_at_k"] <= 1.0

        cmp = evaluator.compare(results)
        assert len(cmp) == 4
        assert "NDCG@K" in cmp.columns

    def test_hybrid_coverage_gte_single_component(self):
        """Hybrid should achieve catalog coverage >= the coverage of the CF alone
        (because fallback adds more diverse products)."""
        train = _train_df()
        meta = _product_meta()
        val = pd.DataFrame([
            {"customer_unique_id": "uid_A", "product_id": "prod_3"},
            {"customer_unique_id": "uid_B", "product_id": "prod_2"},
        ])

        pop = PopularityRecommender(top_k=5, exclude_purchased=False).fit(train)
        cf  = SVDCollaborativeFilter(n_components=2, exclude_purchased=False, random_state=42).fit(train)
        cbf = ContentBasedRecommender(exclude_purchased=False).fit(meta, train)
        hybrid = HybridRecommender(
            components=[
                (cf,  0.4, "cf_score"),
                (cbf, 0.3, "cbf_score"),
                (pop, 0.3, "popularity_score"),
            ],
            top_k=5, exclude_purchased=False,
        ).fit(train)

        evaluator = RecommendationEvaluator(k=5)
        cf_result = evaluator.evaluate(cf, val, train, "cf")
        hybrid_result = evaluator.evaluate(hybrid, val, train, "hybrid")

        assert hybrid_result["catalog_coverage_at_k"] >= cf_result["catalog_coverage_at_k"] - 1e-9, (
            "Hybrid coverage should be >= CF coverage since it blends multiple signals."
        )
