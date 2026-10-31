"""Hardened SQL Catalog and Semantic View Registry for InsightX AI 3.0.

Provides verified, in-memory DuckDB views over Olist relational tables with
strict engine-level isolation (disabled external access, memory ceiling, thread cap).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import duckdb
import pandas as pd

from insightx.utils.logger import get_logger

logger = get_logger(__name__)

# Default engine-level security limits for DuckDB in-memory session
DEFAULT_DUCKDB_ENGINE_CONFIG: Dict[str, Any] = {
    "enable_external_access": False,
    "max_memory": "256MB",
    "threads": 1,
}

# Standard analytical queries for executive KPIs
QUERY_MONTHLY_REVENUE = """
    SELECT 
        strftime('%Y-%m', CAST(order_purchase_timestamp AS TIMESTAMP)) AS order_month,
        COUNT(DISTINCT o.order_id) AS total_orders,
        ROUND(SUM(payment_value), 2) AS total_revenue,
        ROUND(AVG(payment_value), 2) AS average_order_value
    FROM v_orders o
    JOIN v_order_payments p ON o.order_id = p.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY 1
    ORDER BY 1;
"""

QUERY_TOP_CATEGORIES = """
    SELECT 
        COALESCE(p.product_category_name, 'Other') AS category,
        COUNT(oi.order_item_id) AS items_sold,
        ROUND(SUM(oi.price), 2) AS total_sales,
        ROUND(AVG(oi.price), 2) AS avg_item_price
    FROM v_order_items oi
    JOIN v_products p ON oi.product_id = p.product_id
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
    FROM v_orders o
    JOIN v_customers c ON o.customer_id = c.customer_id
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
    GROUP BY customer_state
    ORDER BY delivered_orders DESC;
"""


class SQLCatalog:
    """Manages verified views and executes read-only analytical queries in DuckDB."""

    def __init__(self, engine_config: Optional[Dict[str, Any]] = None):
        """Initializes catalog with restricted engine configuration."""
        self.engine_config = engine_config or DEFAULT_DUCKDB_ENGINE_CONFIG
        self.con = duckdb.connect(database=":memory:", config=self.engine_config)
        self._registered_tables: Dict[str, pd.DataFrame] = {}
        logger.info(f"SQLCatalog initialized with engine config: {self.engine_config}")

    def register_view(self, view_name: str, df: pd.DataFrame) -> None:
        """Registers a verified DataFrame view in the DuckDB session.

        Args:
            view_name: Standard view name (e.g., 'v_orders').
            df: Pandas DataFrame backing the view.
        """
        self._registered_tables[view_name] = df
        self.con.register(view_name, df)
        logger.info(f"Registered view '{view_name}' with {len(df)} rows.")

    def register_all_views(self, tables: Dict[str, pd.DataFrame]) -> None:
        """Standardizes and registers all Olist tables as 'v_' prefixed views."""
        for name, df in tables.items():
            view_name = name if name.startswith("v_") else f"v_{name}"
            self.register_view(view_name, df)

    def get_registered_tables(self) -> Dict[str, pd.DataFrame]:
        """Returns copies of all registered tables for process serialization."""
        return dict(self._registered_tables)

    def execute_read_only(self, query: str) -> pd.DataFrame:
        """Executes a verified query directly in the current session.

        Note: For untrusted LLM-generated SQL, use SQLGuard.execute_safely().
        """
        return self.con.execute(query).df()

    def get_monthly_revenue_kpis(self) -> pd.DataFrame:
        """Executes executive monthly revenue KPI query."""
        return self.execute_read_only(QUERY_MONTHLY_REVENUE)

    def get_top_categories_kpis(self) -> pd.DataFrame:
        """Executes executive top product categories KPI query."""
        return self.execute_read_only(QUERY_TOP_CATEGORIES)

    def get_delivery_performance_kpis(self) -> pd.DataFrame:
        """Executes delivery performance analysis by customer state."""
        return self.execute_read_only(QUERY_DELIVERY_PERFORMANCE)

    def close(self) -> None:
        """Closes the underlying DuckDB connection."""
        self.con.close()
