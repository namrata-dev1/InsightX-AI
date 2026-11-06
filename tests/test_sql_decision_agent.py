"""Tests for Natural Language to SQL Decision Agent.

Verifies:
1. Natural language intent mapping to valid DuckDB SQL.
2. Full query execution and plain-English business explanation generation.
3. Strict SQLGuard security enforcement against adversarial injection and DDL/DML.
4. StructuredEvidence provenance generation.
5. Resilient error handling on invalid queries and schema violations.
"""

from datetime import datetime
import pandas as pd
import pytest

from insightx.agent.sql_decision_agent import SQLDecisionAgent, SQLDecisionResult
from insightx.agent.sql_guard import SQLGuard, SQLSecurityError
from insightx.provenance.types import ProvenanceType


@pytest.fixture
def agent():
    """Initializes a standard SQLDecisionAgent."""
    return SQLDecisionAgent()


@pytest.fixture
def mock_tables():
    """Provides minimal synthetic DataFrames matching view schemas."""
    df_orders = pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "customer_id": ["c1", "c2", "c3"],
        "order_status": ["delivered", "delivered", "canceled"],
        "order_purchase_timestamp": [
            "2017-05-01 10:00:00",
            "2017-06-15 11:00:00",
            "2017-07-20 12:00:00",
        ],
        "order_approved_at": ["2017-05-01", "2017-06-15", "2017-07-20"],
        "order_delivered_carrier_date": ["2017-05-03", "2017-06-18", "2017-07-22"],
        "order_delivered_customer_date": ["2017-05-10 10:00:00", "2017-06-25 11:00:00", None],
        "order_estimated_delivery_date": ["2017-05-15", "2017-06-30", "2017-07-28"],
    })

    df_items = pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "order_item_id": [1, 1, 1],
        "product_id": ["p1", "p2", "p3"],
        "seller_id": ["s1", "s2", "s1"],
        "shipping_limit_date": ["2017-05-05", "2017-06-20", "2017-07-25"],
        "price": [100.0, 150.0, 50.0],
        "freight_value": [15.0, 20.0, 10.0],
    })

    df_payments = pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "payment_sequential": [1, 1, 1],
        "payment_type": ["credit_card", "boleto", "credit_card"],
        "payment_installments": [2, 1, 3],
        "payment_value": [115.0, 170.0, 60.0],
    })

    df_customers = pd.DataFrame({
        "customer_id": ["c1", "c2", "c3"],
        "customer_unique_id": ["u1", "u2", "u3"],
        "customer_zip_code_prefix": [1001, 1002, 1003],
        "customer_city": ["sao paulo", "rio de janeiro", "sao paulo"],
        "customer_state": ["SP", "RJ", "SP"],
    })

    df_products = pd.DataFrame({
        "product_id": ["p1", "p2", "p3"],
        "product_category_name": ["health_beauty", "computers_accessories", "health_beauty"],
        "product_name_lenght": [50, 45, 55],
        "product_description_lenght": [200, 300, 250],
        "product_photos_qty": [2, 3, 1],
        "product_weight_g": [500, 1200, 600],
        "product_length_cm": [20, 30, 25],
        "product_height_cm": [10, 15, 12],
        "product_width_cm": [15, 20, 18],
    })

    df_reviews = pd.DataFrame({
        "review_id": ["r1", "r2", "r3"],
        "order_id": ["o1", "o2", "o3"],
        "review_score": [5, 4, 1],
        "review_comment_title": ["Great", "Good", "Bad"],
        "review_comment_message": ["Fast", "Nice", "Delayed"],
        "review_creation_date": ["2017-05-11", "2017-06-26", "2017-07-23"],
        "review_answer_timestamp": ["2017-05-12", "2017-06-27", "2017-07-24"],
    })

    df_sellers = pd.DataFrame({
        "seller_id": ["s1", "s2"],
        "seller_zip_code_prefix": [2001, 2002],
        "seller_city": ["sao paulo", "curitiba"],
        "seller_state": ["SP", "PR"],
    })

    return {
        "v_orders": df_orders,
        "v_order_items": df_items,
        "v_order_payments": df_payments,
        "v_customers": df_customers,
        "v_products": df_products,
        "v_order_reviews": df_reviews,
        "v_sellers": df_sellers,
    }


def test_agent_supported_intents(agent):
    """Verifies retrieval of pre-configured business analytical query templates."""
    intents = agent.get_supported_intents()
    assert isinstance(intents, list)
    assert len(intents) >= 6
    intent_names = [i["intent"] for i in intents]
    assert "monthly_revenue" in intent_names
    assert "top_categories" in intent_names
    assert "delivery_performance" in intent_names
    assert "payment_breakdown" in intent_names


def test_nl_to_sql_monthly_revenue(agent):
    """Verifies natural language translation for revenue trend."""
    intent, sql = agent.translate_to_sql("What is our monthly revenue trend in 2017?")
    assert intent == "monthly_revenue"
    assert "v_orders" in sql
    assert "v_order_payments" in sql
    assert "2017" in sql


def test_nl_to_sql_top_categories(agent):
    """Verifies category sales translation with custom limit."""
    intent, sql = agent.translate_to_sql("Show me top 5 best selling categories")
    assert intent == "top_categories"
    assert "v_order_items" in sql
    assert "v_products" in sql
    assert "LIMIT 5" in sql


def test_nl_to_sql_delivery_performance(agent):
    """Verifies delivery performance translation with state filtering."""
    intent, sql = agent.translate_to_sql("Check delivery performance in SP state")
    assert intent == "delivery_performance"
    assert "v_orders" in sql
    assert "v_customers" in sql
    assert "SP" in sql


def test_nl_to_sql_payment_breakdown(agent):
    """Verifies payment method distribution translation."""
    intent, sql = agent.translate_to_sql("How do customers pay for orders?")
    assert intent == "payment_breakdown"
    assert "v_order_payments" in sql
    assert "payment_type" in sql


def test_ask_monthly_revenue_execution(agent, mock_tables):
    """Verifies complete execution and business explanation for revenue query."""
    result = agent.ask("What is our monthly revenue trend?", tables_data=mock_tables)
    assert isinstance(result, SQLDecisionResult)
    assert result.is_valid is True
    assert result.error_message is None
    assert result.row_count > 0
    assert "realized revenue" in result.business_summary.lower()
    assert "total_revenue" in result.key_metrics
    assert result.evidence is not None
    assert result.evidence.provenance_type == ProvenanceType.OBSERVED
    assert result.execution_time_seconds > 0.0

    # Serialization check
    d = result.to_dict()
    assert d["is_valid"] is True
    assert "total_revenue" in d["key_metrics"]


def test_ask_top_categories_execution(agent, mock_tables):
    """Verifies top categories execution and ranking summary."""
    result = agent.ask("Which are the top categories by sales?", tables_data=mock_tables)
    assert result.is_valid is True
    assert "top product category" in result.business_summary.lower()
    assert "top_category" in result.key_metrics
    assert result.row_count >= 1


def test_ask_raw_sql_passthrough(agent, mock_tables):
    """Verifies that direct compliant SQL passes through safely."""
    raw_sql = "SELECT order_status, COUNT(order_id) AS cnt FROM v_orders GROUP BY order_status"
    result = agent.ask(raw_sql, tables_data=mock_tables)
    assert result.is_valid is True
    assert result.row_count > 0


def test_adversarial_injection_blocked(agent, mock_tables):
    """Verifies SQLGuard blocks statement chaining and malicious drop attempts."""
    malicious_query = "SELECT * FROM v_orders; DROP TABLE v_orders; --"
    result = agent.ask(malicious_query, tables_data=mock_tables)
    assert result.is_valid is False
    assert "Security Violation" in result.error_message
    assert result.result_df.empty


def test_adversarial_ddl_blocked(agent, mock_tables):
    """Verifies DDL statements are rejected with security errors."""
    result = agent.ask("CREATE TABLE evil (id INT)", tables_data=mock_tables)
    assert result.is_valid is False
    assert "Security Violation" in result.error_message


def test_unapproved_table_blocked(agent, mock_tables):
    """Verifies queries targeting unapproved tables are rejected."""
    result = agent.ask("SELECT * FROM secret_internal_accounts", tables_data=mock_tables)
    assert result.is_valid is False
    assert "Security Violation" in result.error_message


def test_unapproved_column_blocked(agent, mock_tables):
    """Verifies queries referencing unapproved columns are rejected."""
    result = agent.ask("SELECT customer_ssn FROM v_customers", tables_data=mock_tables)
    assert result.is_valid is False
    assert "Security Violation" in result.error_message


def test_timeout_enforcement(agent, mock_tables):
    """Verifies that queries exceeding timeout are terminated cleanly."""
    # Extremely low timeout to trigger watchdog
    result = agent.ask(
        "SELECT o.order_id FROM v_orders o CROSS JOIN v_order_items oi CROSS JOIN v_order_payments op",
        tables_data=mock_tables,
        timeout_seconds=0.001,
    )
    assert result.is_valid is False
    assert "timed out" in result.error_message.lower() or "security violation" in result.error_message.lower()
