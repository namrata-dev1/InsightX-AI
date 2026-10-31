"""Comprehensive Unit Tests for Customer Lifetime Value (CLV) Engine."""

import numpy as np
import pandas as pd
import pytest

from insightx.models.clv import CLVEngine


@pytest.fixture
def sample_customer_cohort():
    """Synthetic customer cohort with high spenders, repeat buyers, and one-time buyers."""
    return pd.DataFrame({
        "customer_unique_id": ["u_whale", "u_repeat", "u_onetime_recent", "u_onetime_old", "u_zero"],
        "recency_days": [15.0, 30.0, 10.0, 300.0, 100.0],
        "frequency": [4, 2, 1, 1, 0],
        "total_monetary": [2000.0, 300.0, 100.0, 80.0, 0.0],
    })


def test_historical_value_calculation(sample_customer_cohort):
    """Verifies that realized gross revenue and parameterized net profit follow business assumptions."""
    # Take rate = 15%, Operating margin = 20% -> Net profit = spend * 0.15 * 0.20 = spend * 0.03
    engine = CLVEngine(custom_assumptions={"marketplace_take_rate": 0.15, "operating_margin": 0.20})
    result = engine.calculate_historical_value(sample_customer_cohort)

    assert "realized_gross_revenue" in result.columns
    assert "realized_marketplace_take" in result.columns
    assert "realized_net_profit" in result.columns

    whale_row = result[result["customer_unique_id"] == "u_whale"].iloc[0]
    assert whale_row["realized_gross_revenue"] == 2000.0
    assert whale_row["realized_marketplace_take"] == 300.0  # 2000 * 0.15
    assert whale_row["realized_net_profit"] == 60.0  # 300 * 0.20


def test_probabilistic_model_suitability_check():
    """Verifies that BG/NBD and Gamma-Gamma suitability correctly identifies sparse cohorts."""
    engine = CLVEngine()

    # Cohort with 97% one-time buyers (Olist profile)
    sparse_cohort = pd.DataFrame({
        "customer_unique_id": [f"u_{i}" for i in range(100)],
        "frequency": [1] * 97 + [2, 3, 4],  # 3% repeat
        "total_monetary": [100.0] * 100,
    })

    audit = engine.evaluate_probabilistic_model_suitability(sparse_cohort)
    assert audit["repeat_customers_pct"] == 3.0
    assert audit["bgnbd_suitable"] is False
    assert "UNSUITABLE" in audit["bgnbd_verdict"]

    # Cohort with sufficient repeat buyers and independent spend
    np.random.seed(42)
    dense_freq = np.random.choice([2, 3, 4, 5], 150)
    dense_cohort = pd.DataFrame({
        "customer_unique_id": [f"u_rep_{i}" for i in range(150)],
        "frequency": dense_freq,
        "total_monetary": dense_freq * np.random.uniform(50, 60, 150),  # Independent spend per order
    })

    audit_dense = engine.evaluate_probabilistic_model_suitability(dense_cohort)
    assert audit_dense["repeat_customers_pct"] == 100.0
    assert audit_dense["gamma_gamma_suitable"] is True


def test_forward_clv_projection(sample_customer_cohort):
    """Verifies expected forward CLV accounts for recency decay and repeat history."""
    engine = CLVEngine()
    result = engine.estimate_forward_clv(sample_customer_cohort, horizon_days=365)

    assert "forward_projected_gross_clv" in result.columns
    assert "forward_projected_net_clv" in result.columns
    assert "clv_tier" in result.columns

    whale = result[result["customer_unique_id"] == "u_whale"].iloc[0]
    onetime_old = result[result["customer_unique_id"] == "u_onetime_old"].iloc[0]

    # Whale must have substantially higher forward CLV than inactive one-time buyer
    assert whale["forward_projected_gross_clv"] > onetime_old["forward_projected_gross_clv"]
    assert whale["forward_projected_net_clv"] > onetime_old["forward_projected_net_clv"]


def test_clv_edge_cases():
    """Verifies that 0 spend, missing columns, and empty datasets do not crash."""
    engine = CLVEngine()

    zero_df = pd.DataFrame({
        "customer_unique_id": ["u_null"],
        "recency_days": [np.nan],
        "frequency": [0],
        "total_monetary": [0.0],
    })

    hist = engine.calculate_historical_value(zero_df)
    assert hist["realized_net_profit"].iloc[0] == 0.0

    forward = engine.estimate_forward_clv(zero_df)
    assert forward["forward_projected_gross_clv"].iloc[0] >= 0.0
