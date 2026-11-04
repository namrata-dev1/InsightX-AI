"""Deterministic Unit and Integration Tests for OlistDatasetAdapter.

Tests:
  1. Required column schema checking (detects missing required columns).
  2. Coercion to timezone-naive UTC datetime objects.
  3. Preservation of unparseable datetimes as NaT with audit logging.
  4. Detection of duplicate primary keys.
  5. Numeric downcasting (float64 -> float32) reducing memory footprint.
  6. Referential integrity checking (orphan items, orders, products).
  7. Non-destructive behavior (no silent row drops).
  8. Backward compatibility of OlistDataLoader with use_adapter=False and use_adapter=True.
"""

import numpy as np
import pandas as pd
import pytest

from insightx.data.adapter import OlistDatasetAdapter, QualityReport
from insightx.data.loader import OlistDataLoader


@pytest.fixture
def sample_raw_tables():
    """Generates synthetic raw Olist tables with deliberate data quality anomalies."""
    orders = pd.DataFrame([
        {
            "order_id": "ord_001",
            "customer_id": "cust_001",
            "order_status": "delivered",
            "order_purchase_timestamp": "2018-01-15 10:30:00",
            "order_delivered_customer_date": "2018-01-20 14:00:00",
        },
        {
            "order_id": "ord_002",
            "customer_id": "cust_002",
            "order_status": "shipped",
            "order_purchase_timestamp": "2018-02-10 11:00:00",
            "order_delivered_customer_date": None,
        },
        {
            "order_id": "ord_003_bad_ts",
            "customer_id": "cust_003",
            "order_status": "delivered",
            "order_purchase_timestamp": "invalid_timestamp_string",
            "order_delivered_customer_date": None,
        },
        {
            # Duplicate order_id
            "order_id": "ord_001",
            "customer_id": "cust_001",
            "order_status": "delivered",
            "order_purchase_timestamp": "2018-01-15 10:30:00",
            "order_delivered_customer_date": "2018-01-20 14:00:00",
        },
    ])

    customers = pd.DataFrame([
        {"customer_id": "cust_001", "customer_unique_id": "uniq_001"},
        {"customer_id": "cust_002", "customer_unique_id": "uniq_002"},
        # cust_003 missing -> tests orphan customer_id in orders
    ])

    order_items = pd.DataFrame([
        {
            "order_id": "ord_001",
            "order_item_id": 1,
            "product_id": "prod_001",
            "seller_id": "seller_001",
            "price": 100.50,
            "freight_value": 15.20,
        },
        {
            # Orphan order_id not in orders
            "order_id": "ord_999_orphan",
            "order_item_id": 1,
            "product_id": "prod_001",
            "seller_id": "seller_001",
            "price": 50.00,
            "freight_value": 10.00,
        },
        {
            # Orphan product_id not in products
            "order_id": "ord_002",
            "order_item_id": 1,
            "product_id": "prod_999_missing_catalog",
            "seller_id": "seller_002",
            "price": 25.00,
            "freight_value": 5.00,
        },
    ])

    products = pd.DataFrame([
        {"product_id": "prod_001", "product_category_name": "eletronicos"},
    ])

    return {
        "orders": orders,
        "customers": customers,
        "order_items": order_items,
        "products": products,
    }


class TestOlistDatasetAdapter:

    def test_schema_validation_required_columns(self):
        adapter = OlistDatasetAdapter()
        incomplete_df = pd.DataFrame({"order_id": ["o1"], "order_status": ["delivered"]})
        adapted_df, report = adapter.adapt_table("orders", incomplete_df, strict=False)

        assert not report.is_valid
        assert any("order_purchase_timestamp" in err for err in report.errors)
        assert any("customer_id" in err for err in report.errors)

        with pytest.raises(ValueError):
            adapter.adapt_table("orders", incomplete_df, strict=True)

    def test_datetime_coercion_timezone_naive(self, sample_raw_tables):
        adapter = OlistDatasetAdapter(enforce_naive_timestamps=True)
        adapted_orders, report = adapter.adapt_table("orders", sample_raw_tables["orders"])

        ts_series = adapted_orders["order_purchase_timestamp"]
        assert pd.api.types.is_datetime64_any_dtype(ts_series)
        # Ensure timezone-naive
        assert getattr(ts_series.dt, "tz", None) is None

        # Invalid timestamp string coerced to NaT
        assert pd.isna(adapted_orders.loc[2, "order_purchase_timestamp"])
        assert report.invalid_timestamp_counts.get("order_purchase_timestamp", 0) == 1

    def test_duplicate_primary_key_audited(self, sample_raw_tables):
        adapter = OlistDatasetAdapter()
        adapted_orders, report = adapter.adapt_table("orders", sample_raw_tables["orders"])

        # ord_001 is duplicated twice
        assert report.duplicate_key_counts == 2
        assert any("duplicate rows" in w for w in report.warnings)
        # Verify raw data not silently dropped
        assert len(adapted_orders) == len(sample_raw_tables["orders"])

    def test_float_downcasting(self, sample_raw_tables):
        raw_items = sample_raw_tables["order_items"]
        assert raw_items["price"].dtype == "float64"

        adapter = OlistDatasetAdapter(downcast_floats=True)
        adapted_items, _ = adapter.adapt_table("order_items", raw_items)
        assert adapted_items["price"].dtype == "float32"
        assert adapted_items["freight_value"].dtype == "float32"

    def test_referential_integrity_audit(self, sample_raw_tables):
        adapter = OlistDatasetAdapter()
        adapted_tables, reports, orphan_summary = adapter.adapt_all(sample_raw_tables)

        assert orphan_summary.get("items_without_valid_order") == 1
        assert orphan_summary.get("orders_without_valid_customer") == 1
        assert orphan_summary.get("items_without_valid_product") == 1

    def test_loader_backward_compatibility(self, tmp_path):
        # Create minimal CSV in tmp_path
        csv_file = tmp_path / "olist_orders_dataset.csv"
        csv_file.write_text("order_id,customer_id,order_status,order_purchase_timestamp\no1,c1,delivered,2018-01-01 10:00:00\n")

        loader = OlistDataLoader(raw_data_dir=tmp_path)

        # 1. Default: returns standard DataFrame
        df_default = loader.load_table("olist_orders_dataset.csv", use_adapter=False)
        assert isinstance(df_default, pd.DataFrame)
        assert len(df_default) == 1

        # 2. use_adapter=True: returns (adapted_df, QualityReport)
        adapted_df, rep = loader.load_table("olist_orders_dataset.csv", use_adapter=True)
        assert isinstance(adapted_df, pd.DataFrame)
        assert isinstance(rep, QualityReport)
        assert pd.api.types.is_datetime64_any_dtype(adapted_df["order_purchase_timestamp"])
