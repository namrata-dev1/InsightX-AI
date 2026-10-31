"""Comprehensive Unit Tests for Churn Risk Prediction Engine."""

import numpy as np
import pandas as pd
import pytest

from insightx.models.churn import ChurnRiskPredictor, HeuristicChurnBaseline
from insightx.models.repeat_purchase import RepeatPurchasePredictor


@pytest.fixture
def synthetic_churn_cohort():
    """Generates synthetic customer cohort with imbalanced churn targets (~85% churned, ~15% retained)."""
    np.random.seed(42)
    n = 200

    # Churned customers: high recency, low review score, single order
    n_churn = 170
    churn_df = pd.DataFrame({
        "recency_days": np.random.uniform(120, 360, n_churn),
        "frequency": np.ones(n_churn, dtype=int),
        "total_orders": np.ones(n_churn, dtype=int),
        "total_monetary": np.random.uniform(20, 150, n_churn),
        "avg_order_value": np.random.uniform(20, 150, n_churn),
        "avg_basket_size": np.ones(n_churn),
        "freight_ratio": np.random.uniform(0.2, 0.5, n_churn),
        "avg_installments": np.random.uniform(1, 3, n_churn),
        "credit_card_share": np.random.uniform(0.3, 0.8, n_churn),
        "boleto_share": np.random.uniform(0.1, 0.4, n_churn),
        "voucher_share": np.zeros(n_churn),
        "avg_review_score": np.random.uniform(1.0, 3.5, n_churn),
        "low_review_ratio": np.random.uniform(0.3, 0.8, n_churn),
        "on_time_delivery_rate": np.random.uniform(0.5, 0.8, n_churn),
        "distinct_categories_count": np.ones(n_churn, dtype=int),
        "churn_label": np.ones(n_churn, dtype=int),
    })

    # Retained customers: low recency, multiple orders, high review score
    n_ret = 30
    ret_df = pd.DataFrame({
        "recency_days": np.random.uniform(5, 60, n_ret),
        "frequency": np.random.choice([2, 3, 4], n_ret),
        "total_orders": np.random.choice([2, 3, 4], n_ret),
        "total_monetary": np.random.uniform(150, 600, n_ret),
        "avg_order_value": np.random.uniform(80, 200, n_ret),
        "avg_basket_size": np.random.uniform(1.5, 3.0, n_ret),
        "freight_ratio": np.random.uniform(0.1, 0.25, n_ret),
        "avg_installments": np.random.uniform(2, 6, n_ret),
        "credit_card_share": np.random.uniform(0.7, 1.0, n_ret),
        "boleto_share": np.random.uniform(0.0, 0.2, n_ret),
        "voucher_share": np.random.uniform(0.0, 0.1, n_ret),
        "avg_review_score": np.random.uniform(4.0, 5.0, n_ret),
        "low_review_ratio": np.zeros(n_ret),
        "on_time_delivery_rate": np.random.uniform(0.9, 1.0, n_ret),
        "distinct_categories_count": np.random.choice([1, 2, 3], n_ret),
        "churn_label": np.zeros(n_ret, dtype=int),
    })

    return pd.concat([churn_df, ret_df], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)


def test_heuristic_churn_baseline():
    """Verifies that rule-based inactivity baseline correctly flags high recency as churn."""
    heuristic = HeuristicChurnBaseline(inactivity_threshold_days=180.0)
    df = pd.DataFrame({"recency_days": [20.0, 150.0, 180.0, 250.0]})

    preds = heuristic.predict(df)
    assert list(preds) == [0, 0, 1, 1]

    probs = heuristic.predict_proba(df)
    assert probs.shape == (4, 2)
    # Probability of churn should increase with recency
    assert probs[3, 1] > probs[0, 1]


def test_logistic_regression_churn_training_and_evaluation(synthetic_churn_cohort):
    """Verifies training, prediction, and imbalanced metrics evaluation for Logistic Regression."""
    X = synthetic_churn_cohort.drop(columns=["churn_label"])
    y = synthetic_churn_cohort["churn_label"]

    predictor = ChurnRiskPredictor(model_type="logistic_regression", random_state=42)
    predictor.train(X.iloc[:150], y.iloc[:150])

    metrics = predictor.evaluate(X.iloc[150:], y.iloc[150:], top_k_pct=0.10)

    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "brier_score" in metrics
    assert "precision_at_top_10_pct" in metrics
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["roc_auc"] >= 0.70  # Separable synthetic signals


def test_random_forest_churn_training(synthetic_churn_cohort):
    """Verifies Random Forest churn classification model."""
    X = synthetic_churn_cohort.drop(columns=["churn_label"])
    y = synthetic_churn_cohort["churn_label"]

    rf_predictor = ChurnRiskPredictor(model_type="random_forest", random_state=42)
    rf_predictor.train(X.iloc[:150], y.iloc[:150])

    probs = rf_predictor.predict_proba(X.iloc[150:])
    assert probs.shape == (50, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_repeat_purchase_predictor_regression(synthetic_churn_cohort):
    """Verifies that RepeatPurchasePredictor remains fully functional and backward-compatible."""
    synth_df = synthetic_churn_cohort.rename(columns={"churn_label": "is_repeat_buyer"})
    # Flip label: 0 -> repeat buyer 1, 1 -> non-repeat 0
    synth_df["is_repeat_buyer"] = 1 - synth_df["is_repeat_buyer"]

    predictor = RepeatPurchasePredictor(model_type="logistic_regression", random_state=42)
    X_train, X_test, y_train, y_test = predictor.prepare_data(synth_df, target_col="is_repeat_buyer")

    predictor.train(X_train, y_train)
    metrics = predictor.evaluate(X_test, y_test)

    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
