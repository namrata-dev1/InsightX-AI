"""Comprehensive Adversarial Security Test Suite for SQLGuard."""

import multiprocessing
import time
import pandas as pd
import pytest

from insightx.agent.sql_guard import (
    DisallowedOperationError,
    QueryTimeoutError,
    SQLGuard,
    SQLSecurityError,
    UnapprovedEntityError,
)


@pytest.fixture
def sql_guard() -> SQLGuard:
    """Provides a SQLGuard instance loaded with default security policies."""
    return SQLGuard()


@pytest.fixture
def mock_tables() -> dict:
    """Provides lightweight serializable mock tables for sandbox execution tests."""
    return {
        "v_orders": pd.DataFrame({
            "order_id": ["o1", "o2", "o3"],
            "customer_id": ["c1", "c2", "c3"],
            "order_status": ["delivered", "delivered", "shipped"],
            "order_purchase_timestamp": ["2018-01-01", "2018-02-01", "2018-03-01"],
        }),
        "v_order_payments": pd.DataFrame({
            "order_id": ["o1", "o2", "o3"],
            "payment_value": [100.0, 250.0, 75.0],
            "payment_type": ["credit_card", "boleto", "credit_card"],
        }),
    }


def test_valid_select_with_comments_allowed(sql_guard: SQLGuard):
    """Verifies that legitimate developer comments (-- and /* */) are parsed safely."""
    query_line_comment = "-- Executive diagnostic query\nSELECT order_id FROM v_orders"
    res1 = sql_guard.validate_ast(query_line_comment)
    assert "v_orders" in res1.lower()

    query_block_comment = "SELECT /* inline check */ order_id, customer_id FROM v_orders"
    res2 = sql_guard.validate_ast(query_block_comment)
    assert "order_id" in res2.lower()


def test_statement_chaining_rejected(sql_guard: SQLGuard):
    """Verifies that query chaining with semicolons is strictly rejected."""
    query = "SELECT * FROM v_orders; DROP TABLE v_orders;"
    with pytest.raises(DisallowedOperationError, match="Statement chaining"):
        sql_guard.validate_ast(query)

    query_two_selects = "SELECT order_id FROM v_orders; SELECT customer_id FROM v_customers;"
    with pytest.raises(DisallowedOperationError, match="Statement chaining"):
        sql_guard.validate_ast(query_two_selects)


def test_non_select_root_rejected(sql_guard: SQLGuard):
    """Verifies that DDL and DML operations are strictly rejected."""
    queries = [
        "INSERT INTO v_orders VALUES ('o99', 'c99', 'delivered', '2026-01-01')",
        "UPDATE v_orders SET order_status = 'cancelled'",
        "DELETE FROM v_orders WHERE order_id = 'o1'",
        "DROP TABLE v_orders",
        "ALTER TABLE v_orders ADD COLUMN fake_col INT",
        "CREATE TABLE fake_table (id INT)",
    ]
    for q in queries:
        with pytest.raises(DisallowedOperationError, match="Unauthorized root operation"):
            sql_guard.validate_ast(q)


def test_cte_and_subqueries_allowed(sql_guard: SQLGuard):
    """Verifies that CTEs and subqueries referencing allowed views pass validation."""
    query_cte = """
        WITH delivered_orders AS (
            SELECT order_id, customer_id 
            FROM v_orders 
            WHERE order_status = 'delivered'
        )
        SELECT order_id FROM delivered_orders
    """
    res = sql_guard.validate_ast(query_cte)
    assert "v_orders" in res.lower()

    query_subquery = "SELECT order_id FROM (SELECT order_id FROM v_orders) sub"
    res_sub = sql_guard.validate_ast(query_subquery)
    assert "v_orders" in res_sub.lower()


def test_unapproved_table_in_cte_rejected(sql_guard: SQLGuard):
    """Verifies that CTE referencing unapproved/external sources is rejected."""
    query = """
        WITH malicious_cte AS (
            SELECT * FROM read_csv('passwords.csv')
        )
        SELECT * FROM malicious_cte
    """
    with pytest.raises(UnapprovedEntityError):
        sql_guard.validate_ast(query)


def test_schema_qualifiers_rejected(sql_guard: SQLGuard):
    """Verifies that system or external schema qualifiers are rejected."""
    queries = [
        "SELECT * FROM duckdb.duckdb_tables",
        "SELECT * FROM information_schema.columns",
        "SELECT * FROM sqlite_master",
    ]
    for q in queries:
        with pytest.raises(UnapprovedEntityError):
            sql_guard.validate_ast(q)


def test_nested_disallowed_functions_rejected(sql_guard: SQLGuard):
    """Verifies that nested prohibited functions are detected and rejected."""
    query = "SELECT ROUND(read_csv('test.csv')) FROM v_orders"
    with pytest.raises(UnapprovedEntityError, match="Prohibited function 'READ_CSV'"):
        sql_guard.validate_ast(query)


def test_unapproved_column_rejected(sql_guard: SQLGuard):
    """Verifies that querying unapproved/unregistered columns raises UnapprovedEntityError."""
    query = "SELECT secret_salary_column FROM v_orders"
    with pytest.raises(UnapprovedEntityError, match="Column 'secret_salary_column'"):
        sql_guard.validate_ast(query)


def test_row_limit_enforcement(sql_guard: SQLGuard):
    """Verifies that queries without LIMIT get LIMIT 500, and excessive LIMIT is clamped."""
    query_no_limit = "SELECT order_id FROM v_orders"
    res1 = sql_guard.validate_ast(query_no_limit)
    assert "limit 500" in res1.lower()

    query_high_limit = "SELECT order_id FROM v_orders LIMIT 10000"
    res2 = sql_guard.validate_ast(query_high_limit)
    assert "limit 500" in res2.lower()

    query_small_limit = "SELECT order_id FROM v_orders LIMIT 10"
    res3 = sql_guard.validate_ast(query_small_limit)
    assert "limit 10" in res3.lower()


def test_execute_safely_end_to_end(sql_guard: SQLGuard, mock_tables: dict):
    """Verifies end-to-end execution inside isolated child process on synthetic data."""
    query = """
        SELECT o.order_id, p.payment_value 
        FROM v_orders o 
        JOIN v_order_payments p ON o.order_id = p.order_id 
        WHERE o.order_status = 'delivered'
    """
    df = sql_guard.execute_safely(query, mock_tables, timeout_seconds=3.0)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "payment_value" in df.columns


def test_execute_safely_timeout_termination(sql_guard: SQLGuard, mock_tables: dict):
    """Verifies that an execution exceeding timeout is forcibly killed and raises QueryTimeoutError."""
    # We test with an aggressive timeout of 0.2s on an intentional CPU-spinning join
    spinning_query = """
        SELECT o1.order_id 
        FROM v_orders o1 
        CROSS JOIN v_orders o2 
        CROSS JOIN v_orders o3 
        CROSS JOIN v_order_payments p1 
        CROSS JOIN v_order_payments p2
    """
    start = time.perf_counter()
    with pytest.raises(QueryTimeoutError):
        # 0.05s timeout to guarantee timeout trigger and process kill
        sql_guard.execute_safely(spinning_query, mock_tables, timeout_seconds=0.05)
    duration = time.perf_counter() - start
    assert duration < 2.0  # Must terminate promptly


# ---------------------------------------------------------------------------
# Adversarial wildcard security tests
# ---------------------------------------------------------------------------

@pytest.fixture
def poisoned_tables() -> dict:
    """DataFrame for v_orders that includes columns NOT in the allowlist.

    Simulates a data pipeline bug or deliberate injection that adds sensitive
    columns (credit_card_number, secret_salary_column) to the registered view.
    """
    return {
        "v_orders": pd.DataFrame({
            "order_id": ["o1", "o2"],
            "customer_id": ["c1", "c2"],
            "order_status": ["delivered", "shipped"],
            "order_purchase_timestamp": ["2018-01-01", "2018-02-01"],
            # ── Unauthorized columns ──────────────────────────────────────────
            "credit_card_number": ["4111-1111-1111-1111", "5500-0000-0000-0004"],
            "secret_salary_column": [95000, 87000],
        }),
    }


def test_select_star_cannot_expose_unauthorized_columns(
    sql_guard: SQLGuard, poisoned_tables: dict
):
    """SELECT * must return only allowlisted columns; unauthorized columns must be absent."""
    df = sql_guard.execute_safely("SELECT * FROM v_orders", poisoned_tables, timeout_seconds=10.0)
    assert "credit_card_number" not in df.columns, (
        "SECURITY FAILURE: SELECT * exposed 'credit_card_number'"
    )
    assert "secret_salary_column" not in df.columns, (
        "SECURITY FAILURE: SELECT * exposed 'secret_salary_column'"
    )
    # Must still return the approved columns that the DataFrame actually has
    for approved_col in ("order_id", "customer_id", "order_status", "order_purchase_timestamp"):
        assert approved_col in df.columns, f"Approved column '{approved_col}' missing from result"


def test_qualified_wildcard_cannot_expose_unauthorized_columns(
    sql_guard: SQLGuard, poisoned_tables: dict
):
    """SELECT v_orders.* must return only allowlisted columns."""
    df = sql_guard.execute_safely(
        "SELECT v_orders.* FROM v_orders", poisoned_tables, timeout_seconds=10.0
    )
    assert "credit_card_number" not in df.columns, (
        "SECURITY FAILURE: SELECT v_orders.* exposed 'credit_card_number'"
    )
    assert "secret_salary_column" not in df.columns, (
        "SECURITY FAILURE: SELECT v_orders.* exposed 'secret_salary_column'"
    )
    for approved_col in ("order_id", "customer_id"):
        assert approved_col in df.columns


def test_explicit_unauthorized_column_rejected_at_ast(sql_guard: SQLGuard):
    """Explicit column reference to an unauthorized column must be rejected during AST validation."""
    with pytest.raises(UnapprovedEntityError):
        sql_guard.validate_ast("SELECT credit_card_number FROM v_orders")

    with pytest.raises(UnapprovedEntityError):
        sql_guard.validate_ast("SELECT secret_salary_column FROM v_orders")


def test_cte_wildcard_cannot_expose_unauthorized_columns(
    sql_guard: SQLGuard, poisoned_tables: dict
):
    """A CTE that SELECT * from v_orders must not leak unauthorized columns to the outer query."""
    query = """
        WITH safe_orders AS (
            SELECT * FROM v_orders
        )
        SELECT * FROM safe_orders
    """
    df = sql_guard.execute_safely(query, poisoned_tables, timeout_seconds=10.0)
    assert "credit_card_number" not in df.columns, (
        "SECURITY FAILURE: CTE wildcard chain exposed 'credit_card_number'"
    )
    assert "secret_salary_column" not in df.columns, (
        "SECURITY FAILURE: CTE wildcard chain exposed 'secret_salary_column'"
    )


def test_subquery_wildcard_cannot_expose_unauthorized_columns(
    sql_guard: SQLGuard, poisoned_tables: dict
):
    """A subquery that SELECT * from v_orders must not leak unauthorized columns."""
    query = "SELECT * FROM (SELECT * FROM v_orders) sub"
    df = sql_guard.execute_safely(query, poisoned_tables, timeout_seconds=10.0)
    assert "credit_card_number" not in df.columns, (
        "SECURITY FAILURE: subquery wildcard exposed 'credit_card_number'"
    )
    assert "secret_salary_column" not in df.columns, (
        "SECURITY FAILURE: subquery wildcard exposed 'secret_salary_column'"
    )


def test_existing_valid_queries_still_work_with_wildcard_fix(
    sql_guard: SQLGuard, mock_tables: dict
):
    """All previously valid queries must continue to pass after the wildcard security fix."""
    # Explicit column references
    sql_guard.validate_ast("SELECT order_id, customer_id FROM v_orders")
    # Line comments
    sql_guard.validate_ast("-- diagnostic\nSELECT order_id FROM v_orders")
    # Block comment
    sql_guard.validate_ast("SELECT /* count */ order_id FROM v_orders")
    # CTE with explicit columns
    sql_guard.validate_ast("""
        WITH d AS (SELECT order_id FROM v_orders WHERE order_status = 'delivered')
        SELECT order_id FROM d
    """)
    # JOIN with explicit columns — end-to-end execution
    df = sql_guard.execute_safely(
        """
        SELECT o.order_id, p.payment_value
        FROM v_orders o
        JOIN v_order_payments p ON o.order_id = p.order_id
        WHERE o.order_status = 'delivered'
        """,
        mock_tables,
        timeout_seconds=3.0,
    )
    assert len(df) == 2
    assert "payment_value" in df.columns
