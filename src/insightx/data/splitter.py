"""Chronological Temporal Splitter for InsightX AI 3.0.

Partitions datasets along strict chronological boundaries to eliminate future-data leakage
while preserving natural customer continuity for repeat-purchase evaluation.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import pandas as pd

from insightx.utils.logger import get_logger

logger = get_logger(__name__)

# Default chronological boundaries for Olist dataset
DEFAULT_TRAIN_START = "2017-01-01 00:00:00"
DEFAULT_TRAIN_END = "2018-03-01 00:00:00"
DEFAULT_VAL_END = "2018-05-31 23:59:59"
DEFAULT_TEST_END = "2018-08-31 23:59:59"


@dataclass(frozen=True)
class SplitMetadata:
    """Immutable audit metadata for chronological dataset splits."""
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str
    train_rows: int
    val_rows: int
    test_rows: int
    train_unique_customers: int
    val_unique_customers: int
    test_unique_customers: int
    customer_overlap_train_val: int
    customer_overlap_train_test: int


class TemporalSplitter:
    """Splits order events by chronological cutoffs without future leakage."""

    def __init__(
        self,
        train_start: str = DEFAULT_TRAIN_START,
        train_end: str = DEFAULT_TRAIN_END,
        val_end: str = DEFAULT_VAL_END,
        test_end: str = DEFAULT_TEST_END,
    ):
        self.train_start = pd.to_datetime(train_start)
        self.train_end = pd.to_datetime(train_end)
        self.val_start = self.train_end + pd.Timedelta(seconds=1)
        self.val_end = pd.to_datetime(val_end)
        self.test_start = self.val_end + pd.Timedelta(seconds=1)
        self.test_end = pd.to_datetime(test_end)

        if not (self.train_start < self.train_end < self.val_end < self.test_end):
            raise ValueError(
                f"Invalid temporal boundaries ordering: {train_start} < {train_end} < {val_end} < {test_end}"
            )

    def split_orders(
        self,
        orders_df: pd.DataFrame,
        customers_df: Optional[pd.DataFrame] = None,
        timestamp_col: str = "order_purchase_timestamp",
    ) -> Tuple[Dict[str, pd.DataFrame], SplitMetadata]:
        """Chronologically partitions orders DataFrame into train, validation, and test.

        Preserves customer overlap between train and test for valid repeat-buyer evaluation.

        Args:
            orders_df: Orders DataFrame containing timestamp_col.
            customers_df: Optional customers DataFrame for tracking customer_unique_id overlap.
            timestamp_col: Column name containing order timestamp.

        Returns:
            Tuple[Dict[str, pd.DataFrame], SplitMetadata]: Partitions and split audit metadata.
        """
        df = orders_df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # If customers_df is provided, merge customer_unique_id for continuity tracking
        if customers_df is not None and "customer_unique_id" in customers_df.columns:
            if "customer_unique_id" not in df.columns:
                df = df.merge(
                    customers_df[["customer_id", "customer_unique_id"]],
                    on="customer_id",
                    how="left",
                )

        train_mask = (df[timestamp_col] >= self.train_start) & (df[timestamp_col] <= self.train_end)
        val_mask = (df[timestamp_col] >= self.val_start) & (df[timestamp_col] <= self.val_end)
        test_mask = (df[timestamp_col] >= self.test_start) & (df[timestamp_col] <= self.test_end)

        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()

        # Track unique customer continuity if customer_unique_id is available
        cust_col = "customer_unique_id" if "customer_unique_id" in df.columns else "customer_id"
        train_custs = set(train_df[cust_col].dropna().unique())
        val_custs = set(val_df[cust_col].dropna().unique())
        test_custs = set(test_df[cust_col].dropna().unique())

        overlap_train_val = len(train_custs.intersection(val_custs))
        overlap_train_test = len(train_custs.intersection(test_custs))

        metadata = SplitMetadata(
            train_start=str(self.train_start),
            train_end=str(self.train_end),
            val_start=str(self.val_start),
            val_end=str(self.val_end),
            test_start=str(self.test_start),
            test_end=str(self.test_end),
            train_rows=len(train_df),
            val_rows=len(val_df),
            test_rows=len(test_df),
            train_unique_customers=len(train_custs),
            val_unique_customers=len(val_custs),
            test_unique_customers=len(test_custs),
            customer_overlap_train_val=overlap_train_val,
            customer_overlap_train_test=overlap_train_test,
        )

        logger.info(
            f"Temporal Split: Train={len(train_df)} rows, Val={len(val_df)} rows, Test={len(test_df)} rows. "
            f"Customer continuity (train-test overlap)={overlap_train_test} users."
        )

        return {"train": train_df, "validation": val_df, "test": test_df}, metadata
