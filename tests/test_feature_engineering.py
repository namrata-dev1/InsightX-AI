"""Unit tests for Point-in-Time Customer Feature Engineering Pipeline."""

import numpy as np
import pandas as pd
import pytest

from insightx.data.validator import TemporalLeakageValidator
from insightx.features.engineering import CustomerFeaturePipeline, FeaturePipeline


@pytest.fixture
def synthetic_behavior_data():
    """Provides a synthetic relational fixture with controlled timestamps."""
    customers_df = pd.DataFrame({
        "customer_id": ["c1_o1", "c1_o2", "c2_o1", "c3_o1"],
        "customer_unique_id": ["u_alice", "u_alice", "u_bob", "u_charlie"],
        "customer_state": ["SP", "SP", "RJ", "MG"],
    })

    # u_alice: 2 orders (Jan 10, Feb 20).
    # u_bob: 1 order (Feb 25), delivered AFTER cutoff (March 5).
    # u_charlie: 1 order (March 15 - AFTER cutoff).
    orders_df = pd.DataFrame({
        "order_id": ["ord_1", "ord_2", "ord_3", "ord_4"],
        "customer_id": ["c1_o1", "c1_o2", "c2_o1", "c3_o1"],
        "order_status": ["delivered", "delivered", "delivered", "delivered"],
        "order_purchase_timestamp": [
            "2018-01-10 10:00:00",
            "2018-02-20 14:00:00",
            "2018-02-25 12:00:00",
            "2018-03-15 09:00:00",
        ],
        "order_delivered_customer_date": [
            "2018-01-15 12:00:00",
            "2018-02-24 16:00:00",
            "2018-03-05 18:00:00",  # Delivered after 2018-03-01 cutoff!
            "2018-03-20 11:00:00",
        ],
        "order_estimated_delivery_date": [
            "2018-01-20 00:00:00",
            "2018-02-28 00:00:00",
            "2018-03-04 00:00:00",
            "2018-03-25 00:00:00",
        ],
    })

    # Order items: multi-item order for ord_1
    order_items_df = pd.DataFrame({
        "order_id": ["ord_1", "ord_1", "ord_2", "ord_3"],
        "order_item_id": [1, 2, 1, 1],
        "product_id": ["prod_a", "prod_b", "prod_a", "prod_c"],
        "price": [50.0, 30.0, 100.0, 75.0],
        "freight_value": [10.0, 5.0, 15.0, 12.0],
    })

    # Payments: multi-payment split for ord_2 (credit card + voucher)
    payments_df = pd.DataFrame({
        "order_id": ["ord_1", "ord_2", "ord_2", "ord_3"],
        "payment_sequential": [1, 1, 2, 1],
        "payment_type": ["credit_card", "credit_card", "voucher", "boleto"],
        "payment_installments": [1, 3, 1, 1],
        "payment_value": [95.0, 80.0, 35.0, 87.0],
    })

    # Reviews: ord_2 has review created AFTER 2018-03-01 cutoff
    reviews_df = pd.DataFrame({
        "review_id": ["rev_1", "rev_2"],
        "order_id": ["ord_1", "ord_2"],
        "review_score": [5, 1],
        "review_creation_date": [
            "2018-01-16 09:00:00",  # Before cutoff
            "2018-03-04 10:00:00",  # AFTER cutoff!
        ],
    })

    return {
        "customers": customers_df,
        "orders": orders_df,
        "order_items": order_items_df,
        "payments": payments_df,
        "reviews": reviews_df,
    }


def test_customer_unique_id_aggregation(synthetic_behavior_data):
    """Verifies that features aggregate strictly per customer_unique_id, not customer_id."""
    pipeline = CustomerFeaturePipeline()
    features = pipeline.build_customer_feature_matrix(
        orders_df=synthetic_behavior_data["orders"],
        customers_df=synthetic_behavior_data["customers"],
        order_items_df=synthetic_behavior_data["order_items"],
        payments_df=synthetic_behavior_data["payments"],
        cutoff_date="2018-03-01 00:00:00",
    )

    # u_alice (2 orders) and u_bob (1 order) were purchased before cutoff
    # u_charlie purchased after cutoff, so only 2 unique customers should exist
    assert len(features) == 2
    assert set(features["customer_unique_id"]) == {"u_alice", "u_bob"}

    alice_row = features[features["customer_unique_id"] == "u_alice"].iloc[0]
    assert alice_row["total_orders"] == 2
    assert alice_row["frequency"] == 2  # Both orders delivered before cutoff
    assert alice_row["total_items"] == 3  # ord_1 had 2 items, ord_2 had 1 item


def test_point_in_time_leakage_prevention(synthetic_behavior_data):
    """Verifies that orders after cutoff date are strictly excluded."""
    pipeline = CustomerFeaturePipeline()
    cutoff = "2018-03-01 00:00:00"

    features = pipeline.build_customer_feature_matrix(
        orders_df=synthetic_behavior_data["orders"],
        customers_df=synthetic_behavior_data["customers"],
        cutoff_date=cutoff,
    )

    # Validate using Phase 1 validator
    assert TemporalLeakageValidator.verify_point_in_time(
        features, cutoff_date=cutoff, timestamp_col="latest_order_timestamp"
    )
    # Ensure u_charlie (ordered March 15) is not present
    assert "u_charlie" not in features["customer_unique_id"].values


def test_future_review_leakage_prevented(synthetic_behavior_data):
    """Verifies reviews created after cutoff are excluded even if order was purchased before cutoff."""
    pipeline = CustomerFeaturePipeline()

    # ord_2 was placed Feb 20, but its 1-star review was created March 4
    # Cutoff March 1: review for ord_2 must NOT be included (only ord_1 review score 5 exists)
    features_march1 = pipeline.build_customer_feature_matrix(
        orders_df=synthetic_behavior_data["orders"],
        customers_df=synthetic_behavior_data["customers"],
        reviews_df=synthetic_behavior_data["reviews"],
        cutoff_date="2018-03-01 00:00:00",
    )
    alice_row = features_march1[features_march1["customer_unique_id"] == "u_alice"].iloc[0]
    assert alice_row["avg_review_score"] == 5.0  # Only review 1 is visible

    # Cutoff March 10: review for ord_2 IS now visible, so average review is (5 + 1)/2 = 3.0
    features_march10 = pipeline.build_customer_feature_matrix(
        orders_df=synthetic_behavior_data["orders"],
        customers_df=synthetic_behavior_data["customers"],
        reviews_df=synthetic_behavior_data["reviews"],
        cutoff_date="2018-03-10 00:00:00",
    )
    alice_march10 = features_march10[features_march10["customer_unique_id"] == "u_alice"].iloc[0]
    assert alice_march10["avg_review_score"] == 3.0


def test_in_transit_order_at_cutoff_not_delivered(synthetic_behavior_data):
    """Verifies that an order delivered after cutoff is marked in-transit as of cutoff."""
    pipeline = CustomerFeaturePipeline()

    # ord_3 (u_bob) was placed Feb 25, delivered March 5.
    # At cutoff 2018-03-01, it was NOT yet delivered.
    features = pipeline.build_customer_feature_matrix(
        orders_df=synthetic_behavior_data["orders"],
        customers_df=synthetic_behavior_data["customers"],
        cutoff_date="2018-03-01 00:00:00",
    )
    bob_row = features[features["customer_unique_id"] == "u_bob"].iloc[0]
    assert bob_row["total_orders"] == 1
    assert bob_row["frequency"] == 0  # 0 delivered orders as of cutoff


def test_multi_payment_deduplication(synthetic_behavior_data):
    """Verifies payment splits (e.g. credit card + voucher) are correctly aggregated."""
    pipeline = CustomerFeaturePipeline()
    features = pipeline.build_customer_feature_matrix(
        orders_df=synthetic_behavior_data["orders"],
        customers_df=synthetic_behavior_data["customers"],
        payments_df=synthetic_behavior_data["payments"],
        cutoff_date="2018-03-01 00:00:00",
    )

    alice_row = features[features["customer_unique_id"] == "u_alice"].iloc[0]
    # ord_1 = 95.0, ord_2 = 80.0 + 35.0 = 115.0 -> total 210.0
    assert alice_row["total_monetary"] == 210.0
    assert alice_row["voucher_share"] > 0.0


def test_forward_targets_creation(synthetic_behavior_data):
    """Verifies clean forward-window target generation without label peek."""
    pipeline = CustomerFeaturePipeline()
    targets = pipeline.create_forward_targets(
        orders_df=synthetic_behavior_data["orders"],
        customers_df=synthetic_behavior_data["customers"],
        cutoff_date="2018-03-01 00:00:00",
        horizon_days=90,
        lookback_days=365,
    )

    # Alice bought delivered orders in lookback window and none in forward window -> active=True, repeat=0, churn=1
    alice_t = targets[targets["customer_unique_id"] == "u_alice"].iloc[0]
    assert alice_t["is_active_in_lookback"] is True or alice_t["is_active_in_lookback"] == 1
    assert alice_t["repeat_in_horizon"] == 0
    assert alice_t["churn_in_horizon"] == 1

    # Charlie bought on March 15 (forward window)
    charlie_t = targets[targets["customer_unique_id"] == "u_charlie"].iloc[0]
    assert charlie_t["is_active_in_lookback"] is False or charlie_t["is_active_in_lookback"] == 0
    assert charlie_t["repeat_in_horizon"] == 1
    assert charlie_t["churn_in_horizon"] == 0


def test_feature_pipeline_backward_compatible_alias():
    """Verifies that FeaturePipeline is an alias to CustomerFeaturePipeline."""
    assert FeaturePipeline is CustomerFeaturePipeline
