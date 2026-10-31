"""Tests for SQLCatalog view management and KPI queries."""

import pandas as pd
import pytest

from insightx.data.sql_catalog import SQLCatalog


@pytest.fixture
def catalog() -> SQLCatalog:
    cat = SQLCatalog()
    cat.register_view(
        "v_orders",
        pd.DataFrame({
            "order_id": ["o1", "o2", "o3"],
            "customer_id": ["c1", "c2", "c3"],
            "order_status": ["delivered", "delivered", "delivered"],
            "order_purchase_timestamp": ["2018-01-01 10:00:00", "2018-01-15 11:00:00", "2018-02-01 12:00:00"],
            "order_delivered_customer_date": ["2018-01-05 10:00:00", "2018-01-20 11:00:00", "2018-02-06 12:00:00"],
            "order_estimated_delivery_date": ["2018-01-10 10:00:00", "2018-01-25 11:00:00", "2018-02-12 12:00:00"],
        }),
    )
    cat.register_view(
        "v_order_payments",
        pd.DataFrame({
            "order_id": ["o1", "o2", "o3"],
            "payment_value": [100.0, 150.0, 200.0],
            "payment_type": ["credit_card", "boleto", "credit_card"],
        }),
    )
    cat.register_view(
        "v_customers",
        pd.DataFrame({
            "customer_id": ["c1", "c2", "c3"],
            "customer_unique_id": ["u1", "u2", "u1"],
            "customer_state": ["SP", "RJ", "SP"],
        }),
    )
    cat.register_view(
        "v_order_items",
        pd.DataFrame({
            "order_id": ["o1", "o2", "o3"],
            "order_item_id": [1, 1, 1],
            "product_id": ["p1", "p2", "p1"],
            "price": [90.0, 140.0, 190.0],
        }),
    )
    cat.register_view(
        "v_products",
        pd.DataFrame({
            "product_id": ["p1", "p2"],
            "product_category_name": ["beleza_saude", "relogios_presentes"],
        }),
    )
    yield cat
    cat.close()


def test_monthly_revenue_kpi_query(catalog: SQLCatalog):
    """Verifies executive monthly revenue query calculation."""
    df = catalog.get_monthly_revenue_kpis()
    assert len(df) == 2  # 2018-01 and 2018-02
    assert "total_revenue" in df.columns
    assert df.loc[df["order_month"] == "2018-01", "total_orders"].values[0] == 2
    assert df.loc[df["order_month"] == "2018-01", "total_revenue"].values[0] == 250.0


def test_top_categories_kpi_query(catalog: SQLCatalog):
    """Verifies top product categories query calculation."""
    df = catalog.get_top_categories_kpis()
    assert len(df) == 2
    assert "total_sales" in df.columns
    assert df.iloc[0]["category"] == "beleza_saude"


def test_delivery_performance_kpi_query(catalog: SQLCatalog):
    """Verifies delivery performance query calculation."""
    df = catalog.get_delivery_performance_kpis()
    assert len(df) == 2  # SP and RJ
    assert "avg_delivery_days" in df.columns
