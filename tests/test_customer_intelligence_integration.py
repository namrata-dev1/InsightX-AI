"""End-to-End Integration Tests for InsightX-AI 3.0 Phase 2: Customer Behaviour Intelligence."""

import numpy as np
import pandas as pd
import pytest

from insightx.features.engineering import CustomerFeaturePipeline
from insightx.features.rfm import RFMAnalyzer
from insightx.insights.behaviour import CustomerBehaviourInsights
from insightx.models.churn import ChurnRiskPredictor
from insightx.models.clv import CLVEngine
from insightx.models.explainability import ModelExplainer
from insightx.models.segmentation import CustomerSegmentation
from insightx.provenance.evidence import StructuredEvidence
from insightx.provenance.types import ProvenanceType


@pytest.fixture
def end_to_end_synthetic_data():
    """Generates synthetic relational tables simulating a multi-customer e-commerce cohort."""
    np.random.seed(42)
    n_customers = 50
    customer_unique_ids = [f"u_{i:03d}" for i in range(n_customers)]

    # Generate 1 to 4 orders per customer to simulate real e-commerce distribution
    customers_records = []
    orders_records = []
    items_records = []
    payments_records = []
    reviews_records = []

    order_counter = 0
    cutoff_ts = pd.to_datetime("2018-03-01 00:00:00")

    for cid in customer_unique_ids:
        # 85% single order, 15% repeat order
        n_orders = 1 if np.random.rand() > 0.15 else np.random.choice([2, 3, 4])
        state = np.random.choice(["SP", "RJ", "MG", "RS", "PR"])

        for o_idx in range(n_orders):
            order_counter += 1
            cust_token = f"c_{order_counter:04d}"
            ord_id = f"ord_{order_counter:04d}"

            # Orders placed between Jan 2017 and May 2018 (spanning before and after cutoff 2018-03-01)
            days_offset = np.random.randint(10, 500)
            purchase_ts = pd.to_datetime("2017-01-01") + pd.Timedelta(days=days_offset)
            delivery_ts = purchase_ts + pd.Timedelta(days=np.random.randint(3, 14))
            estimated_ts = purchase_ts + pd.Timedelta(days=12)

            customers_records.append({
                "customer_id": cust_token,
                "customer_unique_id": cid,
                "customer_state": state,
            })

            orders_records.append({
                "order_id": ord_id,
                "customer_id": cust_token,
                "order_status": "delivered",
                "order_purchase_timestamp": str(purchase_ts),
                "order_delivered_customer_date": str(delivery_ts),
                "order_estimated_delivery_date": str(estimated_ts),
            })

            # Items
            price = round(float(np.random.uniform(20.0, 300.0)), 2)
            freight = round(float(np.random.uniform(10.0, 40.0)), 2)
            items_records.append({
                "order_id": ord_id,
                "order_item_id": 1,
                "product_id": f"prod_{np.random.randint(1, 20)}",
                "price": price,
                "freight_value": freight,
            })

            # Payments
            payments_records.append({
                "order_id": ord_id,
                "payment_sequential": 1,
                "payment_type": np.random.choice(["credit_card", "boleto", "voucher"]),
                "payment_installments": int(np.random.choice([1, 2, 3, 6])),
                "payment_value": round(price + freight, 2),
            })

            # Reviews (some submitted after cutoff for orders placed before cutoff)
            if np.random.rand() > 0.10:  # 90% review rate
                rev_ts = delivery_ts + pd.Timedelta(days=1)
                reviews_records.append({
                    "review_id": f"rev_{order_counter:04d}",
                    "order_id": ord_id,
                    "review_score": int(np.random.choice([1, 2, 3, 4, 5], p=[0.08, 0.05, 0.10, 0.25, 0.52])),
                    "review_creation_date": str(rev_ts),
                })

    return {
        "customers": pd.DataFrame(customers_records),
        "orders": pd.DataFrame(orders_records),
        "order_items": pd.DataFrame(items_records),
        "payments": pd.DataFrame(payments_records),
        "reviews": pd.DataFrame(reviews_records),
        "cutoff_date": "2018-03-01 00:00:00",
    }


def test_end_to_end_customer_intelligence_pipeline(end_to_end_synthetic_data):
    """Verifies that all Phase 2 modules connect seamlessly from raw tables to executive insights."""
    cutoff = end_to_end_synthetic_data["cutoff_date"]

    # 1. Feature Engineering
    pipeline = CustomerFeaturePipeline()
    features_df = pipeline.build_customer_feature_matrix(
        orders_df=end_to_end_synthetic_data["orders"],
        customers_df=end_to_end_synthetic_data["customers"],
        order_items_df=end_to_end_synthetic_data["order_items"],
        payments_df=end_to_end_synthetic_data["payments"],
        reviews_df=end_to_end_synthetic_data["reviews"],
        cutoff_date=cutoff,
    )
    assert len(features_df) > 0
    assert "customer_unique_id" in features_df.columns
    assert "recency_days" in features_df.columns
    assert "total_monetary" in features_df.columns

    # 2. Forward Target Generation
    targets_df = pipeline.create_forward_targets(
        orders_df=end_to_end_synthetic_data["orders"],
        customers_df=end_to_end_synthetic_data["customers"],
        cutoff_date=cutoff,
        horizon_days=90,
        lookback_days=365,
    )
    assert len(targets_df) > 0
    assert "churn_in_horizon" in targets_df.columns

    # Merge features with targets
    modeling_df = features_df.merge(targets_df, on="customer_unique_id", how="inner")
    assert len(modeling_df) == len(features_df)

    # 3. RFM Analysis
    rfm_analyzer = RFMAnalyzer(reference_date=cutoff)
    rfm_raw = rfm_analyzer.calculate_rfm(
        orders_df=end_to_end_synthetic_data["orders"],
        customers_df=end_to_end_synthetic_data["customers"],
        payments_df=end_to_end_synthetic_data["payments"],
        reference_date=cutoff,
    )
    scored_rfm = rfm_analyzer.assign_rfm_scores(rfm_raw, quantiles=4)
    assert "r_score" in scored_rfm.columns
    assert "f_score" in scored_rfm.columns
    assert "m_score" in scored_rfm.columns
    assert "rfm_segment" in scored_rfm.columns

    # 4. Customer Segmentation (K-Means)
    seg_features = ["recency_days", "frequency", "total_monetary", "avg_installments", "freight_ratio"]
    seg_model = CustomerSegmentation(n_clusters=3, random_state=42)
    clusters = seg_model.fit_predict(modeling_df[seg_features])
    stability = seg_model.evaluate_stability(modeling_df[seg_features], n_clusters=3)
    profiles = seg_model.profile_clusters(modeling_df[seg_features], clusters)

    assert len(clusters) == len(modeling_df)
    assert stability["mean_ari"] >= 0.70
    assert "persona_name" in profiles.columns

    # 5. Customer Lifetime Value (CLV)
    clv_engine = CLVEngine()
    historical_val = clv_engine.calculate_historical_value(modeling_df)
    forward_clv = clv_engine.estimate_forward_clv(historical_val, horizon_days=365)
    audit = clv_engine.evaluate_probabilistic_model_suitability(modeling_df)

    assert "realized_net_profit" in forward_clv.columns
    assert "forward_projected_gross_clv" in forward_clv.columns
    assert "forward_projected_net_clv" in forward_clv.columns
    assert "clv_tier" in forward_clv.columns
    assert "bgnbd_suitable" in audit

    # 6. Churn Risk Modeling
    active_cohort = forward_clv[forward_clv["is_active_in_lookback"] == True].copy()
    if len(active_cohort) >= 10:
        churn_predictor = ChurnRiskPredictor(model_type="logistic_regression", random_state=42)
        X = active_cohort[seg_features]
        y = active_cohort["churn_in_horizon"]

        churn_predictor.train(X, y)
        probs = churn_predictor.predict_proba(X)
        assert probs.shape[0] == len(X)
        assert probs.shape[1] == 2
        active_cohort["predicted_churn_probability"] = probs[:, 1]

        # 7. Model Explainability & Behavioural Insights
        explainer = ModelExplainer(model=churn_predictor, feature_names=seg_features)
        explainer.fit_explainer()
        global_imp = explainer.compute_global_importance(X)
        assert len(global_imp) == len(seg_features)

        instance_exp = explainer.explain_instance(X.iloc[0])
        assert "predicted_churn_probability" in instance_exp
        assert "causality_disclaimer" in instance_exp

        # 8. Behavioural Insights & Evidence Generation
        insights_engine = CustomerBehaviourInsights(dataset_id="olist-test-integration")
        portfolio_df = active_cohort.merge(scored_rfm[["customer_unique_id", "rfm_segment"]], on="customer_unique_id", how="left")
        portfolio_df["rfm_segment"] = portfolio_df["rfm_segment"].fillna("General")

        insights = insights_engine.generate_segment_clv_insights(portfolio_df)
        assert "segment_summary" in insights
        assert len(insights["evidence"]) >= 1
        assert isinstance(insights["evidence"][0], StructuredEvidence)
        assert insights["evidence"][0].provenance_type == ProvenanceType.SIMULATION_ASSUMPTION

        driver_insights = insights_engine.generate_churn_drivers_insight(global_imp, top_n=2)
        assert "causality_statement" in driver_insights
        assert "CORRELATION vs CAUSATION" in driver_insights["causality_statement"]

        playbook = insights_engine.generate_retention_playbook("Champions", 0.85, "Platinum")
        assert playbook["retention_priority"] == "URGENT_INTERVENTION"


def test_single_class_churn_cohort_edge_case():
    """Verifies that ChurnRiskPredictor handles a 100% churned cohort without crashing."""
    X = pd.DataFrame({
        "recency_days": [200.0, 250.0, 300.0, 350.0],
        "frequency": [1, 1, 1, 1],
        "total_monetary": [50.0, 40.0, 60.0, 30.0],
        "avg_installments": [1.0, 1.0, 1.0, 1.0],
        "freight_ratio": [0.2, 0.3, 0.25, 0.2],
    })
    y = pd.Series([1, 1, 1, 1])  # 100% churned

    predictor = ChurnRiskPredictor(model_type="logistic_regression", random_state=42)
    predictor.train(X, y)

    probs = predictor.predict_proba(X)
    assert probs.shape == (4, 2)
    # Probability of churn should be 1.0
    assert np.all(probs[:, 1] == 1.0)
    assert np.all(probs[:, 0] == 0.0)
