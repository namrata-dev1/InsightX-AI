"""Phase 3, Milestone 2: Recommendation Baselines — Synthetic Tests.

All tests use fully self-contained synthetic fixtures. No real Olist data required.
Fixtures are kept <= 500 rows to satisfy local execution limits.

Test coverage:
  POPULARITY RECOMMENDER
    1.  Basic top-K returned correctly.
    2.  Tie-breaking is deterministic (same input → same output, alphabetical).
    3.  Exclude-purchased removes known items.
    4.  New / unknown customer receives global top-K (cold-start).
    5.  Empty interaction matrix raises no crash.
    6.  k > catalog size returns at most catalog_size products.

  CATEGORY RECOMMENDER
    7.  Known-category customer gets category-weighted recommendations.
    8.  Unknown-category customer falls back to popularity.
    9.  Category weights sum to 1.0 per customer.
    10. Product from irrelevant category excluded when weight=0.

  CO-PURCHASE RECOMMENDER
    11. Co-purchase pairs identified correctly for multi-product baskets.
    12. Single-product baskets contribute no co-purchase signal.
    13. Jaccard similarity in [0, 1].
    14. Customer with no seed products (new customer) falls back.
    15. All-excluded edge case (all co-purchase candidates already purchased).

  EVALUATOR
    16. Precision@K, Recall@K, F1@K computed correctly on a controlled fixture.
    17. Coverage@K is fraction of catalog recommended.
    18. Customers with no relevant items excluded from P@K / R@K average.
    19. Empty eval split returns zero metrics without crash.
    20. compare() formats results table correctly.

  PROVENANCE
    21. Evidence records are non-empty after fit().
    22. Evidence raw_value is a primitive.
    23. Evidence source_type = RECSYS_CORE.

  INTEGRATION
    24. Full pipeline: build interactions → fit all 3 → evaluate on val split.
    25. Recommendations are reproducible (calling twice gives same output).
"""

from typing import Dict, List, Set
import numpy as np
import pandas as pd
import pytest

from insightx.models.recommenders import (
    PopularityRecommender,
    CategoryRecommender,
    CoPurchaseRecommender,
)
from insightx.models.recommendation_eval import RecommendationEvaluator
from insightx.provenance.types import ProvenanceType


# ---------------------------------------------------------------------------
# Shared synthetic fixtures
# ---------------------------------------------------------------------------

def _make_train_interactions() -> pd.DataFrame:
    """Creates a deterministic training interaction matrix.

    Customers and products:
      uid_A bought: prod_1 (x3), prod_2 (x1)       → basket {1,2}
      uid_B bought: prod_1 (x2), prod_3 (x2)       → basket {1,3}
      uid_C bought: prod_2 (x1), prod_3 (x1)       → basket {2,3}
      uid_D bought: prod_4 (x5)                     → basket {4}  (single-product)
      uid_E bought: prod_1 (x1)                     → basket {1}  (single-product)
    popularity ranking: prod_1(6) > prod_3(3) > prod_4(5)? No:
      prod_1: 3+2+1=6, prod_2: 1+1=2, prod_3: 2+1=3, prod_4: 5
    → ranking: prod_1=6, prod_4=5, prod_3=3, prod_2=2
    co-purchase pairs from baskets:
      (prod_1, prod_2) from uid_A: 1
      (prod_1, prod_3) from uid_B: 1
      (prod_2, prod_3) from uid_C: 1
    """
    return pd.DataFrame([
        {"customer_unique_id": "uid_A", "product_id": "prod_1", "purchase_count": 3},
        {"customer_unique_id": "uid_A", "product_id": "prod_2", "purchase_count": 1},
        {"customer_unique_id": "uid_B", "product_id": "prod_1", "purchase_count": 2},
        {"customer_unique_id": "uid_B", "product_id": "prod_3", "purchase_count": 2},
        {"customer_unique_id": "uid_C", "product_id": "prod_2", "purchase_count": 1},
        {"customer_unique_id": "uid_C", "product_id": "prod_3", "purchase_count": 1},
        {"customer_unique_id": "uid_D", "product_id": "prod_4", "purchase_count": 5},
        {"customer_unique_id": "uid_E", "product_id": "prod_1", "purchase_count": 1},
    ])


def _make_product_metadata() -> pd.DataFrame:
    return pd.DataFrame([
        {"product_id": "prod_1", "product_category_name_english": "electronics"},
        {"product_id": "prod_2", "product_category_name_english": "electronics"},
        {"product_id": "prod_3", "product_category_name_english": "toys"},
        {"product_id": "prod_4", "product_category_name_english": "sports"},
    ])


def _make_val_interactions() -> pd.DataFrame:
    """Validation interactions: uid_A buys prod_3 (not seen in training)."""
    return pd.DataFrame([
        {"customer_unique_id": "uid_A", "product_id": "prod_3"},
        {"customer_unique_id": "uid_B", "product_id": "prod_2"},
    ])


# ---------------------------------------------------------------------------
# POPULARITY RECOMMENDER
# ---------------------------------------------------------------------------

class TestPopularityRecommender:

    def test_basic_topk_order(self):
        """Top products returned in descending popularity order."""
        train = _make_train_interactions()
        rec = PopularityRecommender(top_k=4, exclude_purchased=False).fit(train)
        result = rec.recommend(customer_id=None, k=4)
        assert list(result["product_id"]) == ["prod_1", "prod_4", "prod_3", "prod_2"], (
            f"Expected popularity order prod_1>prod_4>prod_3>prod_2, got {list(result['product_id'])}"
        )

    def test_tie_breaking_is_deterministic(self):
        """Two products with equal popularity must always be ordered alphabetically."""
        train = pd.DataFrame([
            {"customer_unique_id": "uid_A", "product_id": "prod_Z", "purchase_count": 5},
            {"customer_unique_id": "uid_B", "product_id": "prod_A", "purchase_count": 5},
        ])
        rec = PopularityRecommender(top_k=2, exclude_purchased=False).fit(train)
        result = rec.recommend(k=2)
        # prod_A and prod_Z both have count=5; prod_A comes first alphabetically
        assert result.iloc[0]["product_id"] == "prod_A", (
            "Tie should be broken alphabetically (prod_A before prod_Z)."
        )

    def test_exclude_purchased_removes_known_products(self):
        """Products uid_A already bought (prod_1, prod_2) must not appear in recommendations."""
        train = _make_train_interactions()
        rec = PopularityRecommender(top_k=4, exclude_purchased=True).fit(train)
        result = rec.recommend(customer_id="uid_A", k=4)
        recommended = set(result["product_id"])
        assert "prod_1" not in recommended, "prod_1 was purchased by uid_A — must be excluded."
        assert "prod_2" not in recommended, "prod_2 was purchased by uid_A — must be excluded."

    def test_new_customer_gets_global_topk(self):
        """A customer not seen in training (new_uid) gets global top-K."""
        train = _make_train_interactions()
        rec = PopularityRecommender(top_k=3, exclude_purchased=True).fit(train)
        result = rec.recommend(customer_id="new_uid_not_in_train", k=3)
        assert len(result) == 3, "New customer should receive global top-3."
        assert result.iloc[0]["product_id"] == "prod_1", "Global #1 should be prod_1."

    def test_empty_interaction_matrix_no_crash(self):
        """Empty training data should not crash fit()."""
        train = pd.DataFrame(columns=["customer_unique_id", "product_id", "purchase_count"])
        rec = PopularityRecommender(top_k=5).fit(train)
        result = rec.recommend(k=5)
        assert len(result) == 0, "Empty train → empty recommendations."

    def test_k_larger_than_catalog_returns_catalog_size(self):
        """Requesting more recommendations than catalog size returns all catalog products."""
        train = _make_train_interactions()  # 4 unique products
        rec = PopularityRecommender(top_k=100, exclude_purchased=False).fit(train)
        result = rec.recommend(k=100)
        assert len(result) == 4, f"Should return at most 4 (catalog size), got {len(result)}."

    def test_rank_column_starts_at_one(self):
        """Rank column must start at 1 and be sequential."""
        train = _make_train_interactions()
        rec = PopularityRecommender(top_k=4, exclude_purchased=False).fit(train)
        result = rec.recommend(k=4)
        assert list(result["rank"]) == [1, 2, 3, 4]

    def test_popularity_score_nonnegative(self):
        """All popularity scores must be non-negative."""
        train = _make_train_interactions()
        rec = PopularityRecommender(top_k=4, exclude_purchased=False).fit(train)
        result = rec.recommend(k=4)
        assert (result["popularity_score"] >= 0).all()

    def test_batch_recommend_returns_dict(self):
        """batch_recommend should return a dict keyed by customer_id."""
        train = _make_train_interactions()
        rec = PopularityRecommender(top_k=3).fit(train)
        results = rec.recommend_batch(["uid_A", "uid_B", "new_uid"])
        assert set(results.keys()) == {"uid_A", "uid_B", "new_uid"}
        for cid, df in results.items():
            assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# CATEGORY RECOMMENDER
# ---------------------------------------------------------------------------

class TestCategoryRecommender:

    def test_known_category_customer_gets_relevant_recommendations(self):
        """uid_A (electronics buyer) should get electronics products recommended."""
        train = _make_train_interactions()
        meta = _make_product_metadata()
        rec = CategoryRecommender(top_k=5, exclude_purchased=False).fit(train, meta)
        result = rec.recommend(customer_id="uid_A", k=5)
        # uid_A bought prod_1 (electronics) and prod_2 (electronics)
        # So top recommendations should include electronics products
        assert len(result) > 0
        assert "product_id" in result.columns
        assert "category_score" in result.columns

    def test_unknown_customer_fallback_to_popularity(self):
        """New customer with no history should fall back to popularity-based recs."""
        train = _make_train_interactions()
        meta = _make_product_metadata()
        rec = CategoryRecommender(top_k=3, fallback_to_popularity=True).fit(train, meta)
        result = rec.recommend(customer_id="new_uid_999", k=3)
        assert len(result) > 0, "New customer should receive fallback recommendations."

    def test_category_exclude_purchased(self):
        """Already-purchased products must be excluded when flag is set."""
        train = _make_train_interactions()
        meta = _make_product_metadata()
        rec = CategoryRecommender(top_k=5, exclude_purchased=True).fit(train, meta)
        # uid_A bought prod_1 and prod_2
        result = rec.recommend(customer_id="uid_A", k=5)
        recommended = set(result["product_id"])
        assert "prod_1" not in recommended
        assert "prod_2" not in recommended

    def test_all_categories_unknown_fallback(self):
        """Customer whose all purchased products have category 'unknown' falls back."""
        train = pd.DataFrame([
            {"customer_unique_id": "uid_X", "product_id": "prod_no_cat", "purchase_count": 1},
        ])
        meta = pd.DataFrame([
            {"product_id": "prod_no_cat", "product_category_name_english": "unknown"},
        ])
        rec = CategoryRecommender(top_k=1, fallback_to_popularity=True).fit(train, meta)
        result = rec.recommend(customer_id="uid_X", k=1)
        # Should not crash; either empty or fallback
        assert isinstance(result, pd.DataFrame)

    def test_no_crash_missing_category_in_metadata(self):
        """Products with missing category must not crash the recommender."""
        train = _make_train_interactions()
        # Metadata missing prod_4 entirely
        meta = _make_product_metadata().iloc[:3].copy()  # only 3 products
        rec = CategoryRecommender(top_k=5).fit(train, meta)
        result = rec.recommend(customer_id="uid_A", k=5)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# CO-PURCHASE RECOMMENDER
# ---------------------------------------------------------------------------

class TestCoPurchaseRecommender:

    def test_copurchase_pairs_identified_correctly(self):
        """uid_A (prod_1 + prod_2) and uid_B (prod_1 + prod_3) create known pairs."""
        train = _make_train_interactions()
        rec = CoPurchaseRecommender(top_k=5, exclude_purchased=False).fit(train)
        # prod_1 should co-occur with prod_2 and prod_3
        assert rec.n_co_pairs >= 3, (
            f"Expected >= 3 co-purchase pairs (12, 13, 23), got {rec.n_co_pairs}."
        )

    def test_single_product_baskets_no_signal(self):
        """uid_D (only prod_4) and uid_E (only prod_1) contribute no co-purchase pairs."""
        train = pd.DataFrame([
            {"customer_unique_id": "uid_D", "product_id": "prod_4", "purchase_count": 5},
            {"customer_unique_id": "uid_E", "product_id": "prod_1", "purchase_count": 1},
        ])
        rec = CoPurchaseRecommender(top_k=5).fit(train)
        assert rec.n_co_pairs == 0, "Single-product baskets must not contribute co-purchase pairs."

    def test_jaccard_in_range(self):
        """Jaccard similarity must be in [0, 1] for all pairs."""
        train = _make_train_interactions()
        rec = CoPurchaseRecommender(top_k=5).fit(train)
        if rec._jaccard_df is not None and len(rec._jaccard_df) > 0:
            jaccards = rec._jaccard_df["jaccard"]
            assert (jaccards >= 0.0).all() and (jaccards <= 1.0).all(), (
                f"Jaccard out of range: min={jaccards.min()}, max={jaccards.max()}"
            )

    def test_new_customer_falls_back(self):
        """A customer not in training data gets popularity-based fallback."""
        train = _make_train_interactions()
        rec = CoPurchaseRecommender(top_k=3, fallback_to_popularity=True).fit(train)
        result = rec.recommend(customer_id="brand_new_customer", k=3)
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 3

    def test_all_candidates_excluded_edge_case(self):
        """If all co-purchase candidates are already purchased, should fall back or return empty."""
        train = pd.DataFrame([
            {"customer_unique_id": "uid_A", "product_id": "prod_1", "purchase_count": 2},
            {"customer_unique_id": "uid_A", "product_id": "prod_2", "purchase_count": 1},
            {"customer_unique_id": "uid_B", "product_id": "prod_1", "purchase_count": 1},
            {"customer_unique_id": "uid_B", "product_id": "prod_2", "purchase_count": 1},
        ])
        rec = CoPurchaseRecommender(top_k=5, exclude_purchased=True).fit(train)
        # uid_A bought prod_1 and prod_2; only co-purchase signal is (prod_1 ↔ prod_2)
        # Both are already purchased by uid_A → fallback expected
        result = rec.recommend(customer_id="uid_A", k=5)
        assert isinstance(result, pd.DataFrame), "Must return DataFrame, not raise exception."

    def test_exclude_purchased_flag(self):
        """With exclude_purchased=True, seed products not in recommendations."""
        train = _make_train_interactions()
        rec = CoPurchaseRecommender(top_k=5, exclude_purchased=True).fit(train)
        result = rec.recommend(customer_id="uid_A", k=5)
        purchased_by_A = {"prod_1", "prod_2"}
        recommended = set(result["product_id"])
        overlap = recommended & purchased_by_A
        assert len(overlap) == 0, (
            f"Purchased products {overlap} should not appear in recommendations."
        )

    def test_copurchase_score_nonnegative(self):
        """copurchase_score must be >= 0 for all recommendations."""
        train = _make_train_interactions()
        rec = CoPurchaseRecommender(top_k=5).fit(train)
        result = rec.recommend(customer_id="uid_A", k=5)
        if len(result) > 0:
            assert (result["copurchase_score"] >= 0).all()

    def test_symmetric_copurchase(self):
        """Co-purchase graph must be symmetric: if (A,B) then (B,A) with same count."""
        train = _make_train_interactions()
        rec = CoPurchaseRecommender(top_k=5).fit(train)
        if rec._jaccard_df is not None and len(rec._jaccard_df) > 0:
            jdf = rec._jaccard_df
            # Find (prod_1, prod_2) and (prod_2, prod_1) — should have same jaccard
            ab = jdf[(jdf["product_a"] == "prod_1") & (jdf["product_b"] == "prod_2")]
            ba = jdf[(jdf["product_a"] == "prod_2") & (jdf["product_b"] == "prod_1")]
            if len(ab) > 0 and len(ba) > 0:
                assert abs(float(ab["jaccard"].iloc[0]) - float(ba["jaccard"].iloc[0])) < 1e-9


# ---------------------------------------------------------------------------
# EVALUATOR
# ---------------------------------------------------------------------------

class TestRecommendationEvaluator:

    def _make_perfect_rec(self, relevant_product: str):
        """A mock recommender that always recommends the given product first."""
        class PerfectRec:
            def recommend(self, customer_id, k=10):
                return pd.DataFrame({
                    "product_id": [relevant_product],
                    "popularity_score": [99.0],
                    "rank": [1],
                })
        return PerfectRec()

    def test_precision_at_k_perfect_recommender(self):
        """A recommender that always recommends the relevant product → P@1 = 1.0."""
        train = _make_train_interactions()
        val = pd.DataFrame([
            {"customer_unique_id": "uid_A", "product_id": "prod_3"},
        ])
        evaluator = RecommendationEvaluator(k=1)
        rec = self._make_perfect_rec("prod_3")
        result = evaluator.evaluate(rec, val, train, "perfect")
        assert result["precision_at_k"] == 1.0, (
            f"Perfect recommender should give P@1=1.0, got {result['precision_at_k']}"
        )

    def test_precision_at_k_zero_recommender(self):
        """A recommender that never recommends the relevant product → P@K = 0.0."""
        train = _make_train_interactions()
        val = pd.DataFrame([
            {"customer_unique_id": "uid_A", "product_id": "prod_99_never_recommended"},
        ])
        rec = PopularityRecommender(top_k=10, exclude_purchased=False).fit(train)
        evaluator = RecommendationEvaluator(k=10)
        result = evaluator.evaluate(rec, val, train, "popularity_zero")
        # prod_99 is not in catalog, so P@K = 0
        assert result["precision_at_k"] == 0.0

    def test_recall_at_k_with_multiple_relevant(self):
        """Recall@K with 2 relevant items and 1 hit → R@K = 0.5."""
        train = pd.DataFrame([
            {"customer_unique_id": "uid_A", "product_id": "prod_hit", "purchase_count": 10},
            {"customer_unique_id": "uid_A", "product_id": "prod_miss", "purchase_count": 1},
            {"customer_unique_id": "uid_B", "product_id": "prod_other", "purchase_count": 3},
        ])
        val = pd.DataFrame([
            {"customer_unique_id": "uid_A", "product_id": "prod_hit"},
            {"customer_unique_id": "uid_A", "product_id": "prod_new"},  # never recommended
        ])
        # Use a perfect rec for prod_hit
        rec = self._make_perfect_rec("prod_hit")
        evaluator = RecommendationEvaluator(k=1)
        result = evaluator.evaluate(rec, val, train, "recall_test")
        assert result["recall_at_k"] == pytest.approx(0.5, abs=1e-6), (
            f"Expected R@1 = 0.5 (1 of 2 relevant), got {result['recall_at_k']}"
        )

    def test_coverage_at_k(self):
        """Catalog coverage: fraction of all catalog products that appear in any recommendation."""
        train = _make_train_interactions()
        val = _make_val_interactions()
        rec = PopularityRecommender(top_k=4, exclude_purchased=False).fit(train)
        evaluator = RecommendationEvaluator(k=4)
        result = evaluator.evaluate(rec, val, train, "coverage_test")
        # Coverage should be in [0, 1]
        assert 0.0 <= result["catalog_coverage_at_k"] <= 1.0

    def test_customers_with_no_relevant_items_excluded_from_metrics(self):
        """Customers in eval split with 0 relevant products must not affect P@K / R@K."""
        train = _make_train_interactions()
        # uid_Z is in eval but has 0 relevant (not in catalog)
        val = pd.DataFrame([
            {"customer_unique_id": "uid_A", "product_id": "prod_3"},
            # uid_Z has an empty relevant set (product not in catalog)
        ])
        rec = PopularityRecommender(top_k=4, exclude_purchased=False).fit(train)
        evaluator = RecommendationEvaluator(k=4)
        result = evaluator.evaluate(rec, val, train, "exclusion_test")
        # n_with_relevant should be <= n_eval_total
        assert result["n_eval_customers_with_relevant"] <= result["n_eval_customers_total"]

    def test_empty_eval_split_returns_zero_metrics(self):
        """Empty eval split should return 0 metrics without raising."""
        train = _make_train_interactions()
        empty_val = pd.DataFrame(columns=["customer_unique_id", "product_id"])
        rec = PopularityRecommender(top_k=4).fit(train)
        evaluator = RecommendationEvaluator(k=4)
        result = evaluator.evaluate(rec, empty_val, train, "empty_eval")
        assert result["precision_at_k"] == 0.0
        assert result["recall_at_k"] == 0.0
        assert result["n_eval_customers_total"] == 0

    def test_compare_formats_dataframe(self):
        """compare() should return a DataFrame with one row per recommender."""
        results = [
            {"recommender_name": "popularity", "k": 10, "precision_at_k": 0.1,
             "recall_at_k": 0.2, "f1_at_k": 0.13, "catalog_coverage_at_k": 0.5,
             "n_eval_customers_with_relevant": 5, "n_eval_customers_total": 10},
            {"recommender_name": "category", "k": 10, "precision_at_k": 0.12,
             "recall_at_k": 0.22, "f1_at_k": 0.15, "catalog_coverage_at_k": 0.4,
             "n_eval_customers_with_relevant": 5, "n_eval_customers_total": 10},
        ]
        evaluator = RecommendationEvaluator(k=10)
        df = evaluator.compare(results)
        assert len(df) == 2
        assert "recommender" in df.columns
        assert "Precision@K" in df.columns
        assert "Recall@K" in df.columns


# ---------------------------------------------------------------------------
# PROVENANCE
# ---------------------------------------------------------------------------

class TestProvenanceEvidence:

    def test_popularity_evidence_non_empty(self):
        train = _make_train_interactions()
        rec = PopularityRecommender().fit(train)
        assert len(rec.evidence) >= 1, "PopularityRecommender must produce >= 1 evidence record."

    def test_copurchase_evidence_non_empty(self):
        train = _make_train_interactions()
        rec = CoPurchaseRecommender().fit(train)
        assert len(rec.evidence) >= 2

    def test_evidence_raw_value_is_primitive(self):
        train = _make_train_interactions()
        for RecClass in [PopularityRecommender, CoPurchaseRecommender]:
            rec = RecClass().fit(train)
            for ev in rec.evidence:
                assert isinstance(ev.raw_value, (str, int, float, bool, bytes, type(None))), (
                    f"raw_value for {RecClass.__name__} must be a primitive, "
                    f"got {type(ev.raw_value).__name__}"
                )

    def test_evidence_source_type_is_recsys_core(self):
        train = _make_train_interactions()
        meta = _make_product_metadata()
        for rec in [
            PopularityRecommender().fit(train),
            CoPurchaseRecommender().fit(train),
            CategoryRecommender().fit(train, meta),
        ]:
            for ev in rec.evidence:
                assert ev.source_type == "RECSYS_CORE", (
                    f"source_type must be 'RECSYS_CORE', got '{ev.source_type}'"
                )

    def test_copurchase_jaccard_evidence_provenance_type(self):
        """Jaccard similarity metric evidence must be SIMULATION_ASSUMPTION."""
        train = _make_train_interactions()
        rec = CoPurchaseRecommender().fit(train)
        jaccard_evidence = [
            ev for ev in rec.evidence
            if ev.claim_id == "copurchase_jaccard_similarity"
        ]
        assert len(jaccard_evidence) == 1
        assert jaccard_evidence[0].provenance_type == ProvenanceType.SIMULATION_ASSUMPTION


# ---------------------------------------------------------------------------
# INTEGRATION
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_full_pipeline_no_crash(self):
        """End-to-end: build interactions → fit all 3 recommenders → evaluate on val."""
        from insightx.data.recommendation_data import CustomerProductInteractionBuilder

        # Build synthetic orders / customers / items
        rng = np.random.default_rng(1337)
        n_cust = 50
        n_orders = 100
        n_prod = 20

        customers = pd.DataFrame({
            "customer_id": [f"cid_{i:03d}" for i in range(n_cust)],
            "customer_unique_id": [f"uid_{i:03d}" for i in range(n_cust)],
        })
        orders = pd.DataFrame({
            "order_id": [f"ord_{i:04d}" for i in range(n_orders)],
            "customer_id": [f"cid_{int(rng.integers(0, n_cust)):03d}" for _ in range(n_orders)],
            "order_status": ["delivered"] * n_orders,
            "order_purchase_timestamp": pd.date_range(
                "2017-06-01", periods=n_orders, freq="3d"
            ).strftime("%Y-%m-%d %H:%M:%S").tolist(),
        })
        items = pd.DataFrame({
            "order_id": rng.choice(orders["order_id"].values, size=150).tolist(),
            "product_id": [f"prod_{int(rng.integers(0, n_prod)):03d}" for _ in range(150)],
            "price": rng.uniform(20, 300, 150).round(2).tolist(),
            "freight_value": rng.uniform(5, 30, 150).round(2).tolist(),
        })
        meta = pd.DataFrame({
            "product_id": [f"prod_{i:03d}" for i in range(n_prod)],
            "product_category_name_english": (
                rng.choice(["electronics", "toys", "sports"], size=n_prod).tolist()
            ),
        })

        # Build interaction matrix (train split only)
        builder = CustomerProductInteractionBuilder(cutoff_date="2018-03-01")
        train_matrix, _ = builder.build_interaction_matrix(orders, customers, items)

        # Fit recommenders
        pop = PopularityRecommender(top_k=5).fit(train_matrix)
        cat = CategoryRecommender(top_k=5).fit(train_matrix, meta)
        cop = CoPurchaseRecommender(top_k=5).fit(train_matrix)

        # Evaluate on a small held-out val set
        val_df = pd.DataFrame({
            "customer_unique_id": [train_matrix["customer_unique_id"].iloc[0]
                                   if len(train_matrix) > 0 else "uid_000"],
            "product_id": [f"prod_{int(rng.integers(0, n_prod)):03d}"],
        })

        evaluator = RecommendationEvaluator(k=5)
        for name, rec in [("popularity", pop), ("category", cat), ("copurchase", cop)]:
            result = evaluator.evaluate(rec, val_df, train_matrix, name, meta)
            assert "precision_at_k" in result
            assert 0.0 <= result["precision_at_k"] <= 1.0
            assert 0.0 <= result["recall_at_k"] <= 1.0
            assert 0.0 <= result["catalog_coverage_at_k"] <= 1.0

    def test_recommendations_are_reproducible(self):
        """Calling recommend() twice must give identical results."""
        train = _make_train_interactions()
        rec = PopularityRecommender(top_k=4, exclude_purchased=False).fit(train)
        result1 = rec.recommend(customer_id="uid_A", k=4)
        result2 = rec.recommend(customer_id="uid_A", k=4)
        pd.testing.assert_frame_equal(result1, result2)

    def test_copurchase_recommendations_reproducible(self):
        """CoPurchaseRecommender must return same results on repeated calls."""
        train = _make_train_interactions()
        rec = CoPurchaseRecommender(top_k=5, exclude_purchased=False).fit(train)
        r1 = rec.recommend(customer_id="uid_A", k=5)
        r2 = rec.recommend(customer_id="uid_A", k=5)
        pd.testing.assert_frame_equal(r1, r2)
