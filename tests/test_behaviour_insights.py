"""Comprehensive Unit Tests for Behavioural Insights and Model Explainability Engine."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from insightx.insights.behaviour import CustomerBehaviourInsights
from insightx.insights.generator import InsightGenerator
from insightx.models.explainability import ModelExplainer
from insightx.provenance.evidence import StructuredEvidence
from insightx.provenance.types import ProvenanceType


@pytest.fixture
def trained_dummy_model():
    """Provides a simple trained Logistic Regression model and feature dataset."""
    np.random.seed(42)
    X = pd.DataFrame({
        "recency_days": [10.0, 50.0, 180.0, 300.0, 25.0, 200.0],
        "frequency": [3, 2, 1, 1, 2, 1],
        "total_monetary": [500.0, 200.0, 50.0, 30.0, 250.0, 40.0],
        "avg_review_score": [5.0, 4.0, 2.0, 1.0, 4.5, 2.5],
    })
    y = np.array([0, 0, 1, 1, 0, 1])  # 1 = churn, 0 = retained

    model = LogisticRegression(random_state=42)
    model.fit(X.values, y)

    return {"model": model, "X": X, "y": y, "feature_names": list(X.columns)}


def test_model_explainer_global_importance(trained_dummy_model):
    """Verifies that ModelExplainer calculates ranked feature importance with directions."""
    explainer = ModelExplainer(
        model=trained_dummy_model["model"],
        feature_names=trained_dummy_model["feature_names"],
    )
    explainer.fit_explainer()

    importance_df = explainer.compute_global_importance(trained_dummy_model["X"])

    assert "feature" in importance_df.columns
    assert "importance_score" in importance_df.columns
    assert "direction" in importance_df.columns
    assert len(importance_df) == len(trained_dummy_model["feature_names"])
    # Should be sorted descending by importance
    assert importance_df["importance_score"].iloc[0] >= importance_df["importance_score"].iloc[-1]


def test_model_explainer_instance_explanation(trained_dummy_model):
    """Verifies that individual customer predictions are accompanied by top drivers and mitigators."""
    explainer = ModelExplainer(
        model=trained_dummy_model["model"],
        feature_names=trained_dummy_model["feature_names"],
    )
    explainer.fit_explainer()

    sample_customer = trained_dummy_model["X"].iloc[0]  # Active VIP customer
    explanation = explainer.explain_instance(sample_customer, top_n=2)

    assert "predicted_churn_probability" in explanation
    assert 0.0 <= explanation["predicted_churn_probability"] <= 1.0
    assert "top_risk_drivers" in explanation
    assert "top_mitigating_factors" in explanation
    assert "causality_disclaimer" in explanation
    assert "NOT establish causal" in explanation["causality_disclaimer"]


def test_model_explainer_handles_nan_features(trained_dummy_model):
    """Verifies that an instance with NaN feature values is explained without crashing."""
    explainer = ModelExplainer(
        model=trained_dummy_model["model"],
        feature_names=trained_dummy_model["feature_names"],
    )
    explainer.fit_explainer()

    nan_row = pd.Series({
        "recency_days": np.nan,
        "frequency": 1,
        "total_monetary": np.nan,
        "avg_review_score": 3.0,
    })

    explanation = explainer.explain_instance(nan_row)
    assert "predicted_churn_probability" in explanation
    assert len(explanation["top_risk_drivers"]) + len(explanation["top_mitigating_factors"]) > 0


def test_behaviour_insights_segment_clv():
    """Verifies segment cross-tabulation and StructuredEvidence creation."""
    portfolio_df = pd.DataFrame({
        "customer_unique_id": [f"u_{i}" for i in range(10)],
        "rfm_segment": ["Champions"] * 3 + ["Loyal Customers"] * 3 + ["Lost"] * 4,
        "forward_projected_gross_clv": [500.0, 450.0, 600.0, 200.0, 150.0, 180.0, 20.0, 10.0, 15.0, 10.0],
        "predicted_churn_probability": [0.05, 0.10, 0.08, 0.30, 0.35, 0.25, 0.95, 0.90, 0.92, 0.98],
    })

    engine = CustomerBehaviourInsights(dataset_id="olist-test-v1")
    insights = engine.generate_segment_clv_insights(portfolio_df)

    assert "segment_summary" in insights
    assert insights["top_valuable_segment"] == "Champions"
    assert "evidence" in insights
    assert len(insights["evidence"]) >= 1

    ev = insights["evidence"][0]
    assert isinstance(ev, StructuredEvidence)
    assert ev.provenance_type == ProvenanceType.SIMULATION_ASSUMPTION
    assert "Champions" in ev.evidence_to_claim_linkage


def test_behaviour_insights_churn_drivers_evidence():
    """Verifies that top churn drivers are linked to evidence with strict non-causal language."""
    importance_df = pd.DataFrame({
        "feature": ["recency_days", "low_review_ratio", "avg_installments"],
        "importance_score": [0.85, 0.42, 0.15],
    })

    engine = CustomerBehaviourInsights(dataset_id="olist-test-v1")
    insights = engine.generate_churn_drivers_insight(importance_df, top_n=2)

    assert "top_churn_predictive_features" in insights
    assert len(insights["top_churn_predictive_features"]) == 2
    assert "causality_statement" in insights
    assert "CORRELATION vs CAUSATION" in insights["causality_statement"]

    ev = insights["evidence"]
    assert isinstance(ev, StructuredEvidence)
    assert ev.unit == "feature_ranking"


def test_retention_playbook_generation():
    """Verifies differentiated actions based on value and churn risk tiers."""
    engine = CustomerBehaviourInsights()

    vip_at_risk = engine.generate_retention_playbook(
        rfm_segment="Champions", predicted_churn_prob=0.85, clv_tier="Platinum"
    )
    assert vip_at_risk["retention_priority"] == "URGENT_INTERVENTION"

    engaged_vip = engine.generate_retention_playbook(
        rfm_segment="Loyal Customers", predicted_churn_prob=0.15, clv_tier="Gold"
    )
    assert engaged_vip["retention_priority"] == "LOYALTY_NURTURING"

    low_value_churn = engine.generate_retention_playbook(
        rfm_segment="Lost", predicted_churn_prob=0.95, clv_tier="Bronze"
    )
    assert low_value_churn["retention_priority"] == "AUTOMATED_RE_ENGAGEMENT"


def test_insight_generator_backward_compatibility():
    """Verifies that legacy InsightGenerator methods remain functional and produce briefs."""
    gen = InsightGenerator()
    recs = gen.generate_segment_recommendations({
        "VIPs": {"monetary": 500.0},
        "OneTimers": {"monetary": 50.0},
    })
    assert len(recs) == 2

    brief = gen.generate_executive_brief({
        "total_revenue": 1500000.0,
        "total_orders": 100000,
        "repeat_rate": 0.031,
    })
    assert "R$1,500,000.00" in brief
    assert "3.10%" in brief
