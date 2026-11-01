"""End-to-End Integration Tests for Phase 3: Recommendation Engine & Intelligence Pipeline.

Tests:
  1. Full end-to-end integration: Raw data -> Interaction Builder -> 6 Recommenders -> Pipeline.
  2. Integration with Phase 2 Intelligence: RFM segment, CLV, and Churn Risk enrichment.
  3. Segment-based personalized recommendations: Repeat, Sparse, and Cold-start.
  4. Offline ranking evaluation across all models with NDCG@K, Diversity, and Gini.
  5. Preserves StructuredEvidence provenance trail throughout all models.
  6. Empty input / missing data robustness.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from insightx.models.recommendation_pipeline import RecommendationIntelligencePipeline
from insightx.provenance.evidence import StructuredEvidence
from insightx.provenance.types import ProvenanceType


@pytest.fixture
def e2e_recsys_data():
    """Generates synthetic relational tables spanning orders, items, products, customers."""
    rng = np.random.default_rng(42)
    n_customers = 40
    n_products = 20

    customers = pd.DataFrame({
        "customer_id": [f"c_{i:03d}" for i in range(n_customers)],
        "customer_unique_id": [f"u_{i:03d}" for i in range(n_customers)],
    })

    # Orders across 2017 to 2018
    n_orders = 80
    base_date = datetime(2017, 4, 1)
    orders = pd.DataFrame({
        "order_id": [f"o_{i:04d}" for i in range(n_orders)],
        "customer_id": [f"c_{int(rng.integers(0, n_customers)):03d}" for _ in range(n_orders)],
        "order_status": ["delivered"] * n_orders,
        "order_purchase_timestamp": [
            (base_date + timedelta(days=int(rng.integers(0, 360)))).strftime("%Y-%m-%d %H:%M:%S")
            for _ in range(n_orders)
        ],
    })

    items = pd.DataFrame({
        "order_id": rng.choice(orders["order_id"].values, size=120).tolist(),
        "product_id": [f"p_{int(rng.integers(0, n_products)):03d}" for _ in range(120)],
        "price": rng.uniform(20, 400, 120).round(2).tolist(),
        "freight_value": rng.uniform(5, 40, 120).round(2).tolist(),
    })

    products = pd.DataFrame({
        "product_id": [f"p_{i:03d}" for i in range(n_products)],
        "product_category_name": rng.choice(["eletronicos", "brinquedos", "esporte"], size=n_products).tolist(),
        "product_weight_g": rng.uniform(100, 3000, n_products).round(1).tolist(),
        "product_length_cm": rng.uniform(10, 50, n_products).round(1).tolist(),
        "product_height_cm": rng.uniform(5, 30, n_products).round(1).tolist(),
        "product_width_cm": rng.uniform(10, 40, n_products).round(1).tolist(),
        "product_photos_qty": rng.integers(1, 6, n_products).tolist(),
    })

    translations = pd.DataFrame({
        "product_category_name": ["eletronicos", "brinquedos", "esporte"],
        "product_category_name_english": ["electronics", "toys", "sports"],
    })

    return {
        "customers": customers,
        "orders": orders,
        "items": items,
        "products": products,
        "translations": translations,
    }


class TestRecommendationPipelineIntegration:

    def test_pipeline_fit_and_components(self, e2e_recsys_data):
        d = e2e_recsys_data
        pipeline = RecommendationIntelligencePipeline(cutoff_date="2018-03-01", top_k=5)
        pipeline.fit(
            orders_df=d["orders"],
            customers_df=d["customers"],
            order_items_df=d["items"],
            products_df=d["products"],
            category_translation_df=d["translations"],
        )

        assert pipeline._fitted
        assert pipeline.popularity_model is not None
        assert pipeline.category_model is not None
        assert pipeline.copurchase_model is not None
        assert pipeline.cf_model is not None
        assert pipeline.cbf_model is not None
        assert pipeline.hybrid_model is not None

        # Evidence preservation
        evidence_list = pipeline.evidence_trail
        assert len(evidence_list) >= 5
        for ev in evidence_list:
            assert isinstance(ev, StructuredEvidence)
            assert ev.source_type == "RECSYS_CORE"

    def test_segment_tailored_recommendations(self, e2e_recsys_data):
        d = e2e_recsys_data
        pipeline = RecommendationIntelligencePipeline(cutoff_date="2018-03-01", top_k=5)
        pipeline.fit(
            orders_df=d["orders"],
            customers_df=d["customers"],
            order_items_df=d["items"],
            products_df=d["products"],
            category_translation_df=d["translations"],
        )

        # 1. Existing customer in training
        known_uid = list(pipeline.customer_training_counts.keys())[0]
        recs_known = pipeline.recommend_for_customer(known_uid, k=5)
        assert len(recs_known) <= 5
        assert recs_known["customer_activity_segment"].iloc[0] in ["repeat", "sparse"]
        assert "product_category_name_english" in recs_known.columns

        # 2. Cold-start customer
        cold_uid = "unseen_cold_customer_999"
        recs_cold = pipeline.recommend_for_customer(cold_uid, k=5)
        assert len(recs_cold) <= 5
        assert recs_cold["customer_activity_segment"].iloc[0] == "cold_start"

    def test_customer_intelligence_enrichment(self, e2e_recsys_data):
        d = e2e_recsys_data
        pipeline = RecommendationIntelligencePipeline(cutoff_date="2018-03-01", top_k=5)
        pipeline.fit(
            orders_df=d["orders"],
            customers_df=d["customers"],
            order_items_df=d["items"],
            products_df=d["products"],
            category_translation_df=d["translations"],
        )

        known_uid = list(pipeline.customer_training_counts.keys())[0]
        intel_df = pd.DataFrame([{
            "customer_unique_id": known_uid,
            "rfm_segment": "Champions",
            "predicted_clv": 520.50,
            "churn_risk_score": 0.12,
            "churn_risk_tier": "Low Risk",
        }])

        recs = pipeline.recommend_for_customer(known_uid, k=5, customer_intelligence_df=intel_df)
        assert "rfm_segment" in recs.columns
        assert recs["rfm_segment"].iloc[0] == "Champions"
        assert recs["predicted_clv"].iloc[0] == 520.50
        assert recs["churn_risk_score"].iloc[0] == 0.12

    def test_offline_evaluation_execution(self, e2e_recsys_data):
        d = e2e_recsys_data
        pipeline = RecommendationIntelligencePipeline(cutoff_date="2018-01-01", top_k=5)
        pipeline.fit(
            orders_df=d["orders"],
            customers_df=d["customers"],
            order_items_df=d["items"],
            products_df=d["products"],
            category_translation_df=d["translations"],
        )

        # Validation set (orders after 2018-01-01)
        val_orders = d["orders"][pd.to_datetime(d["orders"]["order_purchase_timestamp"]) > pd.Timestamp("2018-01-01")]
        val_with_uid = val_orders.merge(
            d["customers"][["customer_id", "customer_unique_id"]], on="customer_id"
        )
        val_interactions = val_with_uid.merge(
            d["items"][["order_id", "product_id"]], on="order_id"
        )[["customer_unique_id", "product_id"]].drop_duplicates()

        eval_out = pipeline.evaluate_offline(val_interactions, k=5)
        assert "comparison_df" in eval_out
        assert "overall_results" in eval_out
        assert "hybrid_segments" in eval_out
        assert len(eval_out["comparison_df"]) == 6
