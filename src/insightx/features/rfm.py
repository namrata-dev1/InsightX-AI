"""RFM (Recency, Frequency, Monetary) Engine for InsightX-AI 3.0.

Key Capabilities:
1. Strict customer_unique_id grouping with point-in-time observation reference cutoff.
2. Adaptive discrete frequency scoring to handle Olist's 96.9% single-purchase sparsity without bin collapse.
3. Quantile rank scoring with robust tie-breaking for Recency and Monetary.
4. Comprehensive business segment taxonomy mapping.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import yaml

from insightx.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_WEIGHTS = {"recency": 0.33, "frequency": 0.33, "monetary": 0.34}


class RFMAnalyzer:
    """Computes mathematically correct customer-level RFM metrics and segment labels."""

    def __init__(
        self,
        reference_date: Optional[Union[str, pd.Timestamp, datetime]] = None,
        config_path: Optional[Union[str, Path]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """Args:
            reference_date: Strict point-in-time cutoff date.
            config_path: Optional path to business_assumptions.yaml or model_config.yaml.
            weights: Optional custom weights for composite RFM score.
        """
        self.reference_date = pd.to_datetime(reference_date) if reference_date is not None else None
        self.weights = weights or DEFAULT_WEIGHTS

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                    rfm_cfg = cfg.get("rfm", cfg.get("rfm_assumptions", {}))
                    if "rfm_weights" in rfm_cfg:
                        self.weights = rfm_cfg["rfm_weights"]
            except Exception as e:
                logger.warning(f"Could not load RFM weights from {config_path}: {e}")

    def calculate_rfm(
        self,
        orders_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        payments_df: Optional[pd.DataFrame] = None,
        order_items_df: Optional[pd.DataFrame] = None,
        reference_date: Optional[Union[str, pd.Timestamp, datetime]] = None,
        delivered_only: bool = True,
    ) -> pd.DataFrame:
        """Calculates Recency, Frequency, and Monetary values per unique customer.

        Args:
            orders_df: Orders table containing order_id, customer_id, order_purchase_timestamp.
            customers_df: Customers table containing customer_id, customer_unique_id.
            payments_df: Optional payments table (order_id, payment_value).
            order_items_df: Optional items table (order_id, price, freight_value).
            reference_date: Point-in-time snapshot date.
            delivered_only: If True, only considers order_status == 'delivered'.

        Returns:
            pd.DataFrame: DataFrame indexed by customer_unique_id with recency, frequency, monetary.
        """
        ref_ts = (
            pd.to_datetime(reference_date)
            if reference_date is not None
            else self.reference_date
        )

        orders = orders_df.copy()
        orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])

        if ref_ts is not None:
            orders = orders[orders["order_purchase_timestamp"] <= ref_ts].copy()
        else:
            ref_ts = orders["order_purchase_timestamp"].max() + pd.Timedelta(seconds=1)
            logger.warning(
                f"No reference_date provided to RFMAnalyzer. Dynamically using {ref_ts}. "
                "For point-in-time reproducibility, always pass an explicit reference_date."
            )

        if len(orders) == 0:
            logger.warning("No orders available prior to reference date.")
            return pd.DataFrame(
                columns=["customer_unique_id", "recency", "frequency", "monetary", "reference_date"]
            )

        if delivered_only and "order_status" in orders.columns:
            orders = orders[orders["order_status"] == "delivered"].copy()

        # Merge customer_unique_id
        if "customer_unique_id" not in orders.columns:
            orders = orders.merge(
                customers_df[["customer_id", "customer_unique_id"]],
                on="customer_id",
                how="inner",
            )

        valid_order_ids = set(orders["order_id"])

        # Determine monetary value per order
        if payments_df is not None and len(payments_df) > 0:
            pmts = payments_df[payments_df["order_id"].isin(valid_order_ids)]
            order_monetary = pmts.groupby("order_id")["payment_value"].sum()
        elif order_items_df is not None and len(order_items_df) > 0:
            items = order_items_df[order_items_df["order_id"].isin(valid_order_ids)]
            items["item_total"] = items["price"] + items.get("freight_value", 0.0)
            order_monetary = items.groupby("order_id")["item_total"].sum()
        elif "payment_value" in orders.columns:
            order_monetary = orders.groupby("order_id")["payment_value"].sum()
        elif "price" in orders.columns:
            order_monetary = orders.groupby("order_id")["price"].sum()
        else:
            order_monetary = pd.Series(0.0, index=orders["order_id"].unique())

        orders["order_monetary"] = orders["order_id"].map(order_monetary).fillna(0.0)

        # Aggregate strictly per customer_unique_id
        grouped = orders.groupby("customer_unique_id")
        recency = (ref_ts - grouped["order_purchase_timestamp"].max()).dt.total_seconds() / 86400.0
        frequency = grouped["order_id"].nunique()
        monetary = grouped["order_monetary"].sum()

        rfm_df = pd.DataFrame({
            "customer_unique_id": recency.index,
            "recency": np.round(recency.values, 2),
            "frequency": frequency.values.astype(int),
            "monetary": np.round(monetary.values, 2),
            "reference_date": str(ref_ts),
        }).reset_index(drop=True)

        logger.info(f"Computed RFM metrics for {len(rfm_df)} unique customers as of {ref_ts}.")
        return rfm_df

    def assign_rfm_scores(self, rfm_df: pd.DataFrame, quantiles: int = 4) -> pd.DataFrame:
        """Assigns R, F, and M scores (1 to quantiles) and segment labels.

        Handling Olist's 96.9% single-purchase sparsity:
        - Recency: Rank-percentile quartiles (1 = oldest, 4 = most recent).
        - Frequency: Discrete rule-based scoring (1 = single order, 2 = 2 orders, 3 = 3 orders, 4 = 4+ orders)
                     to prevent quantile bin-edge collapse.
        - Monetary: Rank-percentile quartiles (1 = lowest spend, 4 = highest spend).

        Args:
            rfm_df: Output from calculate_rfm.
            quantiles: Number of quantile bins for R and M (default: 4).

        Returns:
            pd.DataFrame: RFM DataFrame with r_score, f_score, m_score, rfm_score, and rfm_segment.
        """
        if len(rfm_df) == 0:
            return rfm_df.copy()

        df = rfm_df.copy()

        # 1. Recency Scoring: Lower recency (fewer days ago) -> Higher score (4 = best)
        # Using rank(pct=True) prevents bin-edge collapse on duplicate values
        r_pct = df["recency"].rank(pct=True, method="first", ascending=False)
        df["r_score"] = np.ceil(r_pct * quantiles).astype(int).clip(1, quantiles)

        # 2. Frequency Scoring: Adaptive integer rules for sparse repeat behaviour
        # 96.9% of Olist customers have Frequency = 1. Quantile binning produces degenerate bins.
        def _score_frequency(f: int) -> int:
            if f <= 1:
                return 1
            elif f == 2:
                return 2
            elif f == 3:
                return 3
            else:
                return 4

        df["f_score"] = df["frequency"].apply(_score_frequency)

        # 3. Monetary Scoring: Higher spend -> Higher score (4 = best)
        m_pct = df["monetary"].rank(pct=True, method="first", ascending=True)
        df["m_score"] = np.ceil(m_pct * quantiles).astype(int).clip(1, quantiles)

        # 4. Composite scores
        df["rfm_str"] = (
            df["r_score"].astype(str)
            + df["f_score"].astype(str)
            + df["m_score"].astype(str)
        )
        w_r = self.weights.get("recency", 0.33)
        w_f = self.weights.get("frequency", 0.33)
        w_m = self.weights.get("monetary", 0.34)
        df["rfm_score"] = np.round(
            df["r_score"] * w_r + df["f_score"] * w_f + df["m_score"] * w_m, 2
        )

        # 5. Segment Taxonomy Mapping
        df["rfm_segment"] = df.apply(self._map_segment, axis=1)

        logger.info(f"Assigned RFM scores and segments across {len(df)} customers.")
        return df

    @staticmethod
    def _map_segment(row: pd.Series) -> str:
        """Maps R, F, M integer scores to standard actionable customer segments."""
        r, f, m = row["r_score"], row["f_score"], row["m_score"]

        if r >= 3 and f >= 3 and m >= 3:
            return "Champions"
        elif r >= 2 and f >= 2 and m >= 2:
            return "Loyal Customers"
        elif r == 4 and f == 1 and m >= 3:
            return "Recent High Spenders"
        elif r >= 3 and f == 1 and m == 2:
            return "Potential Loyalists"
        elif r >= 3 and f == 1 and m == 1:
            return "Promising"
        elif r <= 2 and f >= 2:
            return "At Risk"
        elif r == 2 and f == 1:
            return "About to Sleep"
        elif r == 1 and f == 1 and m <= 2:
            return "Lost"
        elif r == 1 and f == 1 and m >= 3:
            return "Hibernating High Spenders"
        else:
            return "Standard Customers"
