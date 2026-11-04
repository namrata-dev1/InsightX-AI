"""Olist Dataset Adapter for InsightX-AI.

Provides schema normalization, type coercion, referential integrity verification,
missing value auditing, and memory-optimized downcasting across raw Olist tables.

Safeguards:
1. Backward compatibility: Works seamlessly with existing DataFrames.
2. Datetime integrity: Enforces timezone-naive UTC timestamps (matching Phase 1-3 conventions).
3. Non-destructive auditing: Never silently drops invalid or orphan records without reporting.
4. Memory efficiency: Optional selective loading and numerical downcasting (float64 -> float32).
5. Referential integrity: Checks key relationships (e.g. order_items -> orders, orders -> customers).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from insightx.utils.logger import get_logger

logger = get_logger(__name__)

# Required columns per core Olist table
CORE_TABLE_REQUIRED_COLUMNS: Dict[str, List[str]] = {
    "orders": ["order_id", "customer_id", "order_status", "order_purchase_timestamp"],
    "customers": ["customer_id", "customer_unique_id"],
    "order_items": ["order_id", "order_item_id", "product_id", "seller_id", "price", "freight_value"],
    "order_payments": ["order_id", "payment_sequential", "payment_type", "payment_installments", "payment_value"],
    "order_reviews": ["review_id", "order_id", "review_score"],
    "products": ["product_id"],
    "sellers": ["seller_id"],
    "category_translation": ["product_category_name", "product_category_name_english"],
}

# Timestamp columns expected per table
TIMESTAMP_COLUMNS_BY_TABLE: Dict[str, List[str]] = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": [
        "shipping_limit_date",
    ],
    "order_reviews": [
        "review_creation_date",
        "review_answer_timestamp",
    ],
}


@dataclass
class QualityReport:
    """Detailed audit report produced by OlistDatasetAdapter."""
    table_name: str
    total_rows: int
    missing_counts: Dict[str, int] = field(default_factory=dict)
    invalid_timestamp_counts: Dict[str, int] = field(default_factory=dict)
    duplicate_key_counts: int = 0
    orphan_record_counts: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    is_valid: bool = True


class OlistDatasetAdapter:
    """Normalizes and validates raw Olist tables for consumption by InsightX-AI modules."""

    def __init__(
        self,
        downcast_floats: bool = True,
        enforce_naive_timestamps: bool = True,
    ):
        """Initializes adapter.

        Args:
            downcast_floats: If True, downcasts float64 to float32 to conserve RAM.
            enforce_naive_timestamps: If True, guarantees all parsed datetimes are timezone-naive.
        """
        self.downcast_floats = downcast_floats
        self.enforce_naive_timestamps = enforce_naive_timestamps
        self.audit_reports: Dict[str, QualityReport] = {}

    def adapt_table(
        self,
        table_key: str,
        df: pd.DataFrame,
        strict: bool = False,
    ) -> Tuple[pd.DataFrame, QualityReport]:
        """Sanitizes, coerces datetimes, audits, and optionally downcasts a single table.

        Args:
            table_key: Normalized short name (e.g. 'orders', 'customers', 'order_items').
            df: Raw input DataFrame.
            strict: If True, raises ValueError upon schema violations or fatal errors.

        Returns:
            Tuple[pd.DataFrame, QualityReport]: Adapted DataFrame and audit report.
        """
        report = QualityReport(table_name=table_key, total_rows=len(df))
        adapted_df = df.copy()

        # 1. Schema check
        required_cols = CORE_TABLE_REQUIRED_COLUMNS.get(table_key, [])
        missing_cols = [c for c in required_cols if c not in adapted_df.columns]
        if missing_cols:
            err_msg = f"Table '{table_key}' missing required columns: {missing_cols}"
            report.errors.append(err_msg)
            report.is_valid = False
            if strict:
                raise ValueError(err_msg)
            else:
                logger.warning(err_msg)

        # 2. Audit missing values
        null_counts = adapted_df.isnull().sum()
        report.missing_counts = {col: int(cnt) for col, cnt in null_counts.items() if cnt > 0}

        # 3. Datetime coercion (timezone-naive)
        expected_ts_cols = TIMESTAMP_COLUMNS_BY_TABLE.get(table_key, [])
        for ts_col in expected_ts_cols:
            if ts_col in adapted_df.columns:
                before_null = int(adapted_df[ts_col].isnull().sum())
                parsed = pd.to_datetime(adapted_df[ts_col], errors="coerce")
                if self.enforce_naive_timestamps:
                    if hasattr(parsed.dt, "tz") and parsed.dt.tz is not None:
                        parsed = parsed.dt.tz_localize(None)

                after_null = int(parsed.isnull().sum())
                new_nulls = after_null - before_null
                if new_nulls > 0:
                    report.invalid_timestamp_counts[ts_col] = new_nulls
                    report.warnings.append(
                        f"{new_nulls} unparseable timestamps coerced to NaT in column '{ts_col}'."
                    )

                adapted_df[ts_col] = parsed

        # 4. Duplicate primary key check
        pk_map = {
            "orders": ["order_id"],
            "customers": ["customer_id"],
            "order_items": ["order_id", "order_item_id"],
            "products": ["product_id"],
            "sellers": ["seller_id"],
        }
        if table_key in pk_map:
            pk_cols = pk_map[table_key]
            if all(col in adapted_df.columns for col in pk_cols):
                dup_mask = adapted_df.duplicated(subset=pk_cols, keep=False)
                dup_cnt = int(dup_mask.sum())
                if dup_cnt > 0:
                    report.duplicate_key_counts = dup_cnt
                    report.warnings.append(
                        f"Found {dup_cnt} duplicate rows on primary key {pk_cols} in '{table_key}'."
                    )

        # 5. Optional downcasting of numeric values
        if self.downcast_floats:
            for col in adapted_df.select_dtypes(include=["float64"]).columns:
                adapted_df[col] = adapted_df[col].astype(np.float32)

        self.audit_reports[table_key] = report
        return adapted_df, report

    def verify_referential_integrity(
        self,
        tables: Dict[str, pd.DataFrame],
    ) -> Dict[str, int]:
        """Audits foreign key linkages across tables without dropping records.

        Checks:
          - order_items.order_id in orders.order_id
          - order_items.product_id in products.product_id
          - orders.customer_id in customers.customer_id
          - order_payments.order_id in orders.order_id
          - order_reviews.order_id in orders.order_id

        Returns:
            Dict[str, int]: Mapping from relation check to orphan row count.
        """
        orphan_summary: Dict[str, int] = {}

        # 1. items -> orders
        if "order_items" in tables and "orders" in tables:
            items_df = tables["order_items"]
            orders_df = tables["orders"]
            if "order_id" in items_df.columns and "order_id" in orders_df.columns:
                known_orders = set(orders_df["order_id"].dropna().unique())
                orphan_items = (~items_df["order_id"].isin(known_orders)).sum()
                orphan_summary["items_without_valid_order"] = int(orphan_items)
                if orphan_items > 0:
                    logger.warning(f"Found {orphan_items} order_items without a matching order_id.")

        # 2. orders -> customers
        if "orders" in tables and "customers" in tables:
            orders_df = tables["orders"]
            cust_df = tables["customers"]
            if "customer_id" in orders_df.columns and "customer_id" in cust_df.columns:
                known_cust = set(cust_df["customer_id"].dropna().unique())
                orphan_orders = (~orders_df["customer_id"].isin(known_cust)).sum()
                orphan_summary["orders_without_valid_customer"] = int(orphan_orders)
                if orphan_orders > 0:
                    logger.warning(f"Found {orphan_orders} orders without a matching customer_id.")

        # 3. items -> products
        if "order_items" in tables and "products" in tables:
            items_df = tables["order_items"]
            prod_df = tables["products"]
            if "product_id" in items_df.columns and "product_id" in prod_df.columns:
                known_prods = set(prod_df["product_id"].dropna().unique())
                orphan_prods = (~items_df["product_id"].isin(known_prods)).sum()
                orphan_summary["items_without_valid_product"] = int(orphan_prods)
                if orphan_prods > 0:
                    logger.warning(f"Found {orphan_prods} order_items with product_id missing from catalog.")

        # 4. items -> sellers
        if "order_items" in tables and "sellers" in tables:
            items_df = tables["order_items"]
            sellers_df = tables["sellers"]
            if "seller_id" in items_df.columns and "seller_id" in sellers_df.columns:
                known_sellers = set(sellers_df["seller_id"].dropna().unique())
                orphan_sellers = (~items_df["seller_id"].isin(known_sellers)).sum()
                orphan_summary["items_without_valid_seller"] = int(orphan_sellers)
                if orphan_sellers > 0:
                    logger.warning(f"Found {orphan_sellers} order_items with seller_id missing from sellers.")

        # 5. payments -> orders
        if "order_payments" in tables and "orders" in tables:
            pay_df = tables["order_payments"]
            orders_df = tables["orders"]
            if "order_id" in pay_df.columns and "order_id" in orders_df.columns:
                known_orders = set(orders_df["order_id"].dropna().unique())
                orphan_pay = (~pay_df["order_id"].isin(known_orders)).sum()
                orphan_summary["payments_without_valid_order"] = int(orphan_pay)
                if orphan_pay > 0:
                    logger.warning(f"Found {orphan_pay} payments without a matching order_id.")

        # 6. reviews -> orders
        if "order_reviews" in tables and "orders" in tables:
            rev_df = tables["order_reviews"]
            orders_df = tables["orders"]
            if "order_id" in rev_df.columns and "order_id" in orders_df.columns:
                known_orders = set(orders_df["order_id"].dropna().unique())
                orphan_rev = (~rev_df["order_id"].isin(known_orders)).sum()
                orphan_summary["reviews_without_valid_order"] = int(orphan_rev)
                if orphan_rev > 0:
                    logger.warning(f"Found {orphan_rev} reviews without a matching order_id.")

        return orphan_summary

    def adapt_all(
        self,
        tables: Dict[str, pd.DataFrame],
        strict: bool = False,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, QualityReport], Dict[str, int]]:
        """Adapts a full collection of loaded Olist tables.

        Returns:
            Tuple[Dict[str, pd.DataFrame], Dict[str, QualityReport], Dict[str, int]]:
            (adapted_tables, per_table_reports, referential_orphan_summary).
        """
        adapted_tables = {}
        reports = {}

        for key, df in tables.items():
            adapted_df, rep = self.adapt_table(key, df, strict=strict)
            adapted_tables[key] = adapted_df
            reports[key] = rep

        orphans = self.verify_referential_integrity(adapted_tables)
        return adapted_tables, reports, orphans
