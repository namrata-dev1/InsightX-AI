"""Business KPI analysis using SQL over Olist tables.

Executes analytical queries using DuckDB directly against in-memory DataFrames
or CSV files for executive-level reporting.
"""

from typing import Optional
import duckdb
import pandas as pd

from insightx.utils.logger import get_logger

logger = get_logger(__name__)


# Analytical SQL Queries for Business KPIs
QUERY_MONTHLY_REVENUE = """
    SELECT 
        strftime('%Y-%m', CAST(order_purchase_timestamp AS TIMESTAMP)) AS order_month,
        COUNT(DISTINCT order_id) AS total_orders,
        ROUND(SUM(payment_value), 2) AS total_revenue,
        ROUND(AVG(payment_value), 2) AS average_order_value
    FROM orders_df o
    JOIN payments_df p ON o.order_id = p.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY 1
    ORDER BY 1;
"""

QUERY_TOP_CATEGORIES = """
    SELECT 
        COALESCE(t.product_category_name_english, p.product_category_name, 'Other') AS category,
        COUNT(oi.order_item_id) AS items_sold,
        ROUND(SUM(oi.price), 2) AS total_sales,
        ROUND(AVG(oi.price), 2) AS avg_item_price
    FROM order_items_df oi
    JOIN products_df p ON oi.product_id = p.product_id
    LEFT JOIN translations_df t ON p.product_category_name = t.product_category_name
    GROUP BY 1
    ORDER BY total_sales DESC
    LIMIT 10;
"""

QUERY_DELIVERY_PERFORMANCE = """
    SELECT 
        customer_state,
        COUNT(order_id) AS delivered_orders,
        ROUND(AVG(date_diff('day', CAST(order_purchase_timestamp AS TIMESTAMP), CAST(order_delivered_customer_date AS TIMESTAMP))), 1) AS avg_delivery_days,
        ROUND(AVG(date_diff('day', CAST(order_delivered_customer_date AS TIMESTAMP), CAST(order_estimated_delivery_date AS TIMESTAMP))), 1) AS avg_days_ahead_of_estimate
    FROM orders_df o
    JOIN customers_df c ON o.customer_id = c.customer_id
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
    GROUP BY customer_state
    ORDER BY delivered_orders DESC;
"""


class SQLAnalyticsEngine:
    """Executes SQL queries on in-memory DataFrames using DuckDB."""

    def __init__(self):
        self.con = duckdb.connect(database=":memory:")

    def execute_query(self, query: str, **dataframes: pd.DataFrame) -> pd.DataFrame:
        """Registers DataFrames as temporary SQL views and executes an analytical query.

        Args:
            query: ANSI SQL query string.
            **dataframes: Named DataFrames (e.g., orders_df=df).

        Returns:
            pd.DataFrame: Result of the SQL query.
        """
        for table_name, df in dataframes.items():
            self.con.register(table_name, df)

        logger.info("Executing analytical SQL query...")
        return self.con.execute(query).df()
