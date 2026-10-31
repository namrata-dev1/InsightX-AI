"""Tests for Temporal Splitter, Point-in-Time Leakage Validator, and Repeat Rate Calculation."""

import pandas as pd
import pytest

from insightx.data.splitter import TemporalSplitter
from insightx.data.validator import (
    TemporalLeakageError,
    TemporalLeakageValidator,
    calculate_empirical_repeat_rate,
)


@pytest.fixture
def synthetic_orders_stream() -> pd.DataFrame:
    """Provides a synthetic stream of orders spanning 2017 to late 2018."""
    return pd.DataFrame({
        "order_id": [f"ord_{i}" for i in range(1, 9)],
        "customer_id": [f"cust_{i}" for i in range(1, 9)],
        "order_status": [
            "delivered", "delivered", "delivered", "delivered",
            "delivered", "delivered", "delivered", "cancelled"
        ],
        "order_purchase_timestamp": [
            "2017-02-15 10:00:00",  # Train
            "2017-06-20 14:00:00",  # Train
            "2018-01-10 09:30:00",  # Train
            "2018-03-15 11:00:00",  # Val
            "2018-04-20 16:45:00",  # Val
            "2018-06-10 12:00:00",  # Test
            "2018-07-25 18:00:00",  # Test
            "2018-08-01 10:00:00",  # Test (cancelled)
        ],
    })


@pytest.fixture
def synthetic_customers() -> pd.DataFrame:
    """Provides customers mapping customer_id to customer_unique_id with repeat buyers."""
    return pd.DataFrame({
        "customer_id": [f"cust_{i}" for i in range(1, 9)],
        # cust_1 (train) and cust_6 (test) are the SAME real person (u_repeat)
        # cust_2 (train) and cust_4 (val) are the SAME real person (u_repeat_val)
        "customer_unique_id": [
            "u_repeat",     # ord_1 (Train)
            "u_val_repeat", # ord_2 (Train)
            "u_single_1",   # ord_3 (Train)
            "u_val_repeat", # ord_4 (Val)
            "u_single_2",   # ord_5 (Val)
            "u_repeat",     # ord_6 (Test)
            "u_single_3",   # ord_7 (Test)
            "u_single_4",   # ord_8 (Test)
        ],
    })


def test_temporal_splitter_boundaries_and_overlap(
    synthetic_orders_stream: pd.DataFrame,
    synthetic_customers: pd.DataFrame,
):
    """Verifies that temporal splitter produces non-overlapping windows while preserving repeat customers."""
    splitter = TemporalSplitter(
        train_start="2017-01-01 00:00:00",
        train_end="2018-03-01 00:00:00",
        val_end="2018-05-31 23:59:59",
        test_end="2018-08-31 23:59:59",
    )
    splits, meta = splitter.split_orders(
        synthetic_orders_stream,
        customers_df=synthetic_customers,
    )

    assert meta.train_rows == 3
    assert meta.val_rows == 2
    assert meta.test_rows == 3

    # Customer continuity: u_repeat purchased in train AND test
    assert meta.customer_overlap_train_test == 1
    # Customer continuity: u_val_repeat purchased in train AND val
    assert meta.customer_overlap_train_val == 1

    # Verify chronological integrity
    assert TemporalLeakageValidator.verify_chronological_splits(splits) is True


def test_chronological_split_leakage_detected():
    """Verifies that TemporalLeakageValidator catches inverted or bleeding splits."""
    bad_splits = {
        "train": pd.DataFrame({"order_purchase_timestamp": ["2018-04-01"]}),  # Future!
        "validation": pd.DataFrame({"order_purchase_timestamp": ["2018-03-15"]}),
        "test": pd.DataFrame({"order_purchase_timestamp": ["2018-06-01"]}),
    }
    with pytest.raises(TemporalLeakageError, match="Temporal leak detected"):
        TemporalLeakageValidator.verify_chronological_splits(bad_splits)


def test_point_in_time_feature_leakage():
    """Verifies that future-peeking feature rows are detected and rejected."""
    cutoff = "2018-03-01 00:00:00"

    # Clean features: all derived from events <= cutoff
    clean_features = pd.DataFrame({
        "customer_unique_id": ["u1", "u2"],
        "total_spend": [120.0, 45.0],
        "latest_order_timestamp": ["2018-01-15 10:00:00", "2018-02-28 23:59:00"],
    })
    assert TemporalLeakageValidator.verify_point_in_time(clean_features, cutoff) is True

    # Leaking features: incorporates an order from May 2018
    leaking_features = pd.DataFrame({
        "customer_unique_id": ["u1", "u2"],
        "total_spend": [120.0, 95.0],
        "latest_order_timestamp": ["2018-01-15 10:00:00", "2018-05-10 12:00:00"],
    })
    with pytest.raises(TemporalLeakageError, match="Point-in-time violation: 1 feature rows"):
        TemporalLeakageValidator.verify_point_in_time(leaking_features, cutoff)


def test_empirical_repeat_rate_calculation(
    synthetic_orders_stream: pd.DataFrame,
    synthetic_customers: pd.DataFrame,
):
    """Verifies dynamic empirical repeat-purchase rate calculation.
    
    In synthetic data:
    - ord_1..ord_7 are delivered.
    - ord_8 is cancelled (must be excluded).
    - u_repeat has 2 delivered orders (ord_1, ord_6) -> Repeat!
    - u_val_repeat has 2 delivered orders (ord_2, ord_4) -> Repeat!
    - u_single_1 (ord_3), u_single_2 (ord_5), u_single_3 (ord_7) have 1 delivered order.
    - Total delivered customers = 5.
    - Repeat customers = 2.
    - Expected rate = 2 / 5 = 0.40.
    """
    res = calculate_empirical_repeat_rate(synthetic_orders_stream, synthetic_customers)
    assert res["total_customers_count"] == 5
    assert res["repeat_customers_count"] == 2
    assert res["repeat_rate"] == 0.40
    assert res["order_status_filter"] == "delivered"
    assert res["granularity"] == "customer_unique_id"
    assert "observation_window_start" in res
