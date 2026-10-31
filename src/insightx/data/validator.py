"""Temporal Leakage Validator and Empirical Repeat-Purchase Engine.

Provides:
1. Point-in-time feature leakage verification.
2. Chronological partition integrity checks.
3. Empirical dynamic repeat-purchase rate calculation based strictly on delivered orders per customer_unique_id.
"""

from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd

from insightx.utils.logger import get_logger

logger = get_logger(__name__)


class TemporalLeakageError(Exception):
    """Raised when future events or label peek occurs within a historical feature set."""
    pass


class TemporalLeakageValidator:
    """Verifies that feature engineering and evaluation splits uphold strict point-in-time correctness."""

    @staticmethod
    def verify_chronological_splits(
        splits: Dict[str, pd.DataFrame],
        timestamp_col: str = "order_purchase_timestamp",
    ) -> bool:
        """Verifies that train, validation, and test partitions are strictly non-overlapping in time.

        Args:
            splits: Dictionary with 'train', 'validation', and 'test' DataFrames.
            timestamp_col: Column name of order timestamps.

        Returns:
            bool: True if chronological ordering is valid.

        Raises:
            TemporalLeakageError: If future records bleed into an earlier partition.
        """
        train_df = splits["train"]
        val_df = splits["validation"]
        test_df = splits["test"]

        max_train = pd.to_datetime(train_df[timestamp_col]).max()
        min_val = pd.to_datetime(val_df[timestamp_col]).min()
        max_val = pd.to_datetime(val_df[timestamp_col]).max()
        min_test = pd.to_datetime(test_df[timestamp_col]).min()

        if max_train >= min_val:
            raise TemporalLeakageError(
                f"Temporal leak detected: Max train timestamp ({max_train}) >= Min validation timestamp ({min_val})"
            )

        if max_val >= min_test:
            raise TemporalLeakageError(
                f"Temporal leak detected: Max validation timestamp ({max_val}) >= Min test timestamp ({min_test})"
            )

        logger.info("Chronological split integrity verified: Train < Validation < Test.")
        return True

    @staticmethod
    def verify_point_in_time(
        features_df: pd.DataFrame,
        cutoff_date: Union[str, pd.Timestamp],
        timestamp_col: str = "latest_order_timestamp",
    ) -> bool:
        """Verifies that no record in the feature matrix was generated from events past the cutoff date.

        Args:
            features_df: Customer feature matrix containing timestamp_col.
            cutoff_date: Strict point-in-time snapshot timestamp.
            timestamp_col: Column recording the latest event timestamp incorporated in the feature row.

        Returns:
            bool: True if no feature row leaks future data.

        Raises:
            TemporalLeakageError: If any feature row used data after cutoff_date.
        """
        cutoff = pd.to_datetime(cutoff_date)
        timestamps = pd.to_datetime(features_df[timestamp_col])
        future_mask = timestamps > cutoff

        if future_mask.any():
            future_count = int(future_mask.sum())
            max_future = timestamps[future_mask].max()
            raise TemporalLeakageError(
                f"Point-in-time violation: {future_count} feature rows incorporate events past "
                f"cutoff {cutoff}. Latest future event: {max_future}."
            )

        logger.info(f"Point-in-time feature verification passed for cutoff {cutoff}.")
        return True


def calculate_empirical_repeat_rate(
    orders_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Dynamically calculates the empirical repeat-purchase rate over a documented observation window.

    Definition:
        Numerator: Distinct unique customers (customer_unique_id) with >= 2 delivered orders.
        Denominator: Distinct unique customers (customer_unique_id) with >= 1 delivered orders.

    Filter:
        Strictly includes only orders with order_status = 'delivered'.

    Returns:
        Dict[str, Any]: Structured audit metrics. Never hardcodes constants.
    """
    # 1. Merge customer_unique_id
    if "customer_unique_id" not in orders_df.columns:
        df = orders_df.merge(
            customers_df[["customer_id", "customer_unique_id"]],
            on="customer_id",
            how="inner",
        )
    else:
        df = orders_df.copy()

    # 2. Strict delivered status filter
    df = df[df["order_status"] == "delivered"].copy()

    # 3. Apply optional observation window filter
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    if start_date is not None:
        start_ts = pd.to_datetime(start_date)
        df = df[df["order_purchase_timestamp"] >= start_ts]
    if end_date is not None:
        end_ts = pd.to_datetime(end_date)
        df = df[df["order_purchase_timestamp"] <= end_ts]

    if len(df) == 0:
        return {
            "repeat_rate": 0.0,
            "repeat_customers_count": 0,
            "total_customers_count": 0,
            "observation_window_start": start_date or "N/A",
            "observation_window_end": end_date or "N/A",
            "order_status_filter": "delivered",
            "granularity": "customer_unique_id",
        }

    # 4. Count distinct delivered orders per real customer
    order_counts = df.groupby("customer_unique_id")["order_id"].nunique()
    total_customers = int(len(order_counts))
    repeat_customers = int((order_counts > 1).sum())

    repeat_rate = float(repeat_customers) / float(total_customers) if total_customers > 0 else 0.0

    actual_start = str(df["order_purchase_timestamp"].min())
    actual_end = str(df["order_purchase_timestamp"].max())

    logger.info(
        f"Empirical Repeat Rate in window [{actual_start} to {actual_end}]: "
        f"{repeat_rate:.4f} ({repeat_customers}/{total_customers} delivered customers)."
    )

    return {
        "repeat_rate": round(repeat_rate, 4),
        "repeat_customers_count": repeat_customers,
        "total_customers_count": total_customers,
        "observation_window_start": actual_start,
        "observation_window_end": actual_end,
        "order_status_filter": "delivered",
        "granularity": "customer_unique_id",
    }
