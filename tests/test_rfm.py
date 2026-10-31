"""Comprehensive Unit Tests for RFM Engine."""

import numpy as np
import pandas as pd
import pytest

from insightx.features.rfm import RFMAnalyzer


@pytest.fixture
def sample_rfm_data():
    """Provides synthetic customer order history with controlled RFM patterns."""
    customers_df = pd.DataFrame({
        "customer_id": ["c1", "c2", "c3", "c4_o1", "c4_o2", "c4_o3", "c5"],
        "customer_unique_id": ["u_alice", "u_bob", "u_charlie", "u_david", "u_david", "u_david", "u_eve"],
    })

    # Reference cutoff: 2018-05-01
    orders_df = pd.DataFrame({
        "order_id": ["ord_1", "ord_2", "ord_3", "ord_4a", "ord_4b", "ord_4c", "ord_5"],
        "customer_id": ["c1", "c2", "c3", "c4_o1", "c4_o2", "c4_o3", "c5"],
        "order_status": ["delivered"] * 7,
        "order_purchase_timestamp": [
            "2018-04-20 10:00:00",  # Alice: 11 days ago (Very recent)
            "2018-03-01 10:00:00",  # Bob: 61 days ago
            "2017-06-01 10:00:00",  # Charlie: ~334 days ago (Old)
            "2018-01-10 10:00:00",  # David order 1
            "2018-02-15 10:00:00",  # David order 2
            "2018-04-10 10:00:00",  # David order 3 (Recent repeat)
            "2018-05-15 10:00:00",  # Eve: AFTER cutoff (Must be excluded)
        ],
    })

    payments_df = pd.DataFrame({
        "order_id": ["ord_1", "ord_2", "ord_3", "ord_4a", "ord_4b", "ord_4c", "ord_5"],
        "payment_value": [
            500.0,   # Alice: High spend
            150.0,   # Bob: Medium spend
            30.0,    # Charlie: Low spend
            100.0,   # David 1
            150.0,   # David 2
            250.0,   # David 3 -> David total 500.0
            1000.0,  # Eve
        ],
    })

    return {
        "customers": customers_df,
        "orders": orders_df,
        "payments": payments_df,
        "reference_date": "2018-05-01 00:00:00",
    }


def test_rfm_calculation_strictly_per_customer_unique_id(sample_rfm_data):
    """Verifies that multi-order customers aggregate strictly per customer_unique_id."""
    analyzer = RFMAnalyzer()
    rfm_df = analyzer.calculate_rfm(
        orders_df=sample_rfm_data["orders"],
        customers_df=sample_rfm_data["customers"],
        payments_df=sample_rfm_data["payments"],
        reference_date=sample_rfm_data["reference_date"],
    )

    # Eve is after cutoff, so exactly 4 customers (alice, bob, charlie, david)
    assert len(rfm_df) == 4
    assert set(rfm_df["customer_unique_id"]) == {"u_alice", "u_bob", "u_charlie", "u_david"}

    david_row = rfm_df[rfm_df["customer_unique_id"] == "u_david"].iloc[0]
    assert david_row["frequency"] == 3
    assert david_row["monetary"] == 500.0
    # Latest purchase was April 10, cutoff is May 1 -> ~20.58 days
    assert 20.0 <= david_row["recency"] <= 21.0


def test_point_in_time_reference_date_enforcement(sample_rfm_data):
    """Verifies that orders occurring after the reference cutoff are excluded."""
    analyzer = RFMAnalyzer()
    rfm_df = analyzer.calculate_rfm(
        orders_df=sample_rfm_data["orders"],
        customers_df=sample_rfm_data["customers"],
        payments_df=sample_rfm_data["payments"],
        reference_date="2018-05-01 00:00:00",
    )
    assert "u_eve" not in rfm_df["customer_unique_id"].values


def test_frequency_adaptive_discrete_scoring():
    """Verifies discrete integer frequency rules (1 to 4) prevent quantile bin-edge collapse."""
    analyzer = RFMAnalyzer()
    # Synthetic cohort reflecting Olist: mostly 1s with few 2s and 3s
    synth_rfm = pd.DataFrame({
        "customer_unique_id": [f"u_{i}" for i in range(100)],
        "recency": np.linspace(1, 300, 100),
        "frequency": [1] * 96 + [2, 2, 3, 5],  # 96% 1s
        "monetary": np.linspace(20, 1000, 100),
    })

    scored = analyzer.assign_rfm_scores(synth_rfm, quantiles=4)

    assert "f_score" in scored.columns
    # Check that frequency 1 gets score 1, 2 gets 2, 3 gets 3, 5 gets 4
    assert set(scored[scored["frequency"] == 1]["f_score"]) == {1}
    assert set(scored[scored["frequency"] == 2]["f_score"]) == {2}
    assert set(scored[scored["frequency"] == 3]["f_score"]) == {3}
    assert set(scored[scored["frequency"] == 5]["f_score"]) == {4}


def test_recency_and_monetary_monotonicity():
    """Verifies that lower recency -> higher R-score, and higher monetary -> higher M-score."""
    analyzer = RFMAnalyzer()
    synth_rfm = pd.DataFrame({
        "customer_unique_id": ["u_best", "u_mid", "u_worst"],
        "recency": [5.0, 50.0, 300.0],
        "frequency": [3, 2, 1],
        "monetary": [1000.0, 300.0, 20.0],
    })

    scored = analyzer.assign_rfm_scores(synth_rfm, quantiles=3)

    best_row = scored[scored["customer_unique_id"] == "u_best"].iloc[0]
    worst_row = scored[scored["customer_unique_id"] == "u_worst"].iloc[0]

    assert best_row["r_score"] > worst_row["r_score"]
    assert best_row["m_score"] > worst_row["m_score"]
    assert best_row["rfm_score"] > worst_row["rfm_score"]


def test_tie_breaking_on_duplicate_recency_monetary():
    """Verifies that duplicate values in recency or monetary do not crash or produce NaNs."""
    analyzer = RFMAnalyzer()
    # All identical recency and monetary values
    synth_rfm = pd.DataFrame({
        "customer_unique_id": [f"u_{i}" for i in range(20)],
        "recency": [30.0] * 20,
        "frequency": [1] * 20,
        "monetary": [150.0] * 20,
    })

    scored = analyzer.assign_rfm_scores(synth_rfm, quantiles=4)
    assert not scored["r_score"].isna().any()
    assert not scored["m_score"].isna().any()
    assert scored["r_score"].min() >= 1 and scored["r_score"].max() <= 4


def test_rfm_segment_taxonomy_assignment():
    """Verifies segment naming rules."""
    analyzer = RFMAnalyzer()
    test_cases = pd.DataFrame({
        "customer_unique_id": ["c_champ", "c_loyal", "c_recent_high", "c_lost"],
        "r_score": [4, 3, 4, 1],
        "f_score": [4, 2, 1, 1],
        "m_score": [4, 3, 4, 1],
    })

    segments = test_cases.apply(analyzer._map_segment, axis=1).tolist()
    assert segments[0] == "Champions"
    assert segments[1] == "Loyal Customers"
    assert segments[2] == "Recent High Spenders"
    assert segments[3] == "Lost"


def test_empty_dataframe_handled_safely():
    """Verifies that an empty DataFrame produces empty output without error."""
    analyzer = RFMAnalyzer()
    empty_df = pd.DataFrame(columns=["order_id", "customer_id", "order_purchase_timestamp", "order_status"])
    cust_df = pd.DataFrame(columns=["customer_id", "customer_unique_id"])

    res = analyzer.calculate_rfm(empty_df, cust_df)
    assert len(res) == 0

    scored = analyzer.assign_rfm_scores(res)
    assert len(scored) == 0
