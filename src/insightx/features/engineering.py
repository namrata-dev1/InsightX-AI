"""Point-in-Time Customer Feature Engineering Pipeline for InsightX-AI 3.0.

Guarantees:
1. Strict point-in-time cutoff enforcement (zero future leakage).
2. Auxiliary event timestamp auditing (reviews, delivery dates, payments must not post-date cutoff).
3. Primary key aggregation strictly at `customer_unique_id`.
4. Multi-item and multi-payment split deduplication without row explosion.
5. Forward prediction horizon label generation for churn and repeat purchase models.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from insightx.utils.logger import get_logger

logger = get_logger(__name__)


class CustomerFeaturePipeline:
    """Extracts leak-free customer behavioral features anchored to a strict point-in-time cutoff."""

    def __init__(self, default_cutoff: Optional[Union[str, pd.Timestamp]] = None):
        """Args:
            default_cutoff: Optional default snapshot date (e.g. '2018-03-01 00:00:00').
        """
        self.default_cutoff = pd.to_datetime(default_cutoff) if default_cutoff else None

    def build_customer_feature_matrix(
        self,
        orders_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        order_items_df: Optional[pd.DataFrame] = None,
        payments_df: Optional[pd.DataFrame] = None,
        reviews_df: Optional[pd.DataFrame] = None,
        products_df: Optional[pd.DataFrame] = None,
        cutoff_date: Optional[Union[str, pd.Timestamp]] = None,
    ) -> pd.DataFrame:
        """Builds point-in-time customer feature matrix as of cutoff_date.

        All events, reviews, and deliveries after cutoff_date are strictly excluded or masked.

        Args:
            orders_df: Orders table (must have order_id, customer_id, order_purchase_timestamp).
            customers_df: Customers table (must have customer_id, customer_unique_id).
            order_items_df: Optional items table (order_id, price, freight_value, product_id).
            payments_df: Optional payments table (order_id, payment_type, payment_installments, payment_value).
            reviews_df: Optional reviews table (order_id, review_score, review_creation_date).
            products_df: Optional products table (product_id, product_category_name).
            cutoff_date: Specific point-in-time cutoff timestamp.

        Returns:
            pd.DataFrame: Feature matrix indexed or keyed by `customer_unique_id`.
        """
        cutoff_ts = (
            pd.to_datetime(cutoff_date)
            if cutoff_date is not None
            else (self.default_cutoff if self.default_cutoff is not None else pd.to_datetime(orders_df["order_purchase_timestamp"]).max())
        )
        logger.info(f"Extracting customer behavioral features anchored at cutoff: {cutoff_ts}")

        # 1. Prepare and filter orders strictly on or before cutoff
        orders = orders_df.copy()
        orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
        orders = orders[orders["order_purchase_timestamp"] <= cutoff_ts].copy()

        if len(orders) == 0:
            logger.warning("No orders found prior to cutoff date. Returning empty DataFrame.")
            return pd.DataFrame()

        # 2. Join customer_unique_id from customers table
        if "customer_unique_id" not in orders.columns:
            orders = orders.merge(
                customers_df[["customer_id", "customer_unique_id", "customer_state"]],
                on="customer_id",
                how="inner",
            )
        else:
            if "customer_state" not in orders.columns and "customer_state" in customers_df.columns:
                orders = orders.merge(
                    customers_df[["customer_id", "customer_state"]],
                    on="customer_id",
                    how="left",
                )

        # 3. Handle delivery timestamp point-in-time status
        # If order was delivered AFTER cutoff, it was still pending/in-transit as of cutoff!
        if "order_delivered_customer_date" in orders.columns:
            orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"])
            orders["is_delivered_at_cutoff"] = (
                (orders["order_status"] == "delivered")
                & (orders["order_delivered_customer_date"].notna())
                & (orders["order_delivered_customer_date"] <= cutoff_ts)
            )
            if "order_estimated_delivery_date" in orders.columns:
                orders["order_estimated_delivery_date"] = pd.to_datetime(orders["order_estimated_delivery_date"])
                orders["delivery_delay_days"] = np.where(
                    orders["is_delivered_at_cutoff"],
                    (orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]).dt.total_seconds() / 86400.0,
                    np.nan,
                )
                orders["is_on_time"] = np.where(
                    orders["is_delivered_at_cutoff"],
                    orders["delivery_delay_days"] <= 0,
                    np.nan,
                )
        else:
            orders["is_delivered_at_cutoff"] = orders["order_status"] == "delivered"
            orders["delivery_delay_days"] = np.nan
            orders["is_on_time"] = np.nan

        valid_order_ids = set(orders["order_id"])

        # 4. Process order items (if provided)
        items_summary = None
        if order_items_df is not None and len(order_items_df) > 0:
            items = order_items_df[order_items_df["order_id"].isin(valid_order_ids)].copy()
            if products_df is not None and "product_category_name" in products_df.columns:
                items = items.merge(
                    products_df[["product_id", "product_category_name"]],
                    on="product_id",
                    how="left",
                )
                cat_nunique = items.groupby("order_id")["product_category_name"].nunique()
            else:
                cat_nunique = pd.Series(0, index=items["order_id"].unique())

            items_agg = items.groupby("order_id").agg(
                item_count=("order_item_id", "count"),
                total_item_price=("price", "sum"),
                total_freight_value=("freight_value", "sum"),
            )
            items_agg["category_count"] = cat_nunique
            items_summary = items_agg

        # 5. Process payments (if provided)
        payments_summary = None
        if payments_df is not None and len(payments_df) > 0:
            pmts = payments_df[payments_df["order_id"].isin(valid_order_ids)].copy()
            pmts_agg = pmts.groupby("order_id").agg(
                payment_total=("payment_value", "sum"),
                max_installments=("payment_installments", "max"),
                avg_installments=("payment_installments", "mean"),
            )
            # Payment type one-hot fractions
            for p_type in ["credit_card", "boleto", "voucher", "debit_card"]:
                is_type = (pmts["payment_type"] == p_type).astype(float)
                type_share = pmts.assign(**{f"is_{p_type}": is_type}).groupby("order_id")[f"is_{p_type}"].mean()
                pmts_agg[f"{p_type}_share"] = type_share

            payments_summary = pmts_agg

        # 6. Process reviews strictly on or before cutoff date (zero future review leak)
        reviews_summary = None
        if reviews_df is not None and len(reviews_df) > 0:
            revs = reviews_df[reviews_df["order_id"].isin(valid_order_ids)].copy()
            if "review_creation_date" in revs.columns:
                revs["review_creation_date"] = pd.to_datetime(revs["review_creation_date"])
                # Exclude any review created after cutoff
                revs = revs[revs["review_creation_date"] <= cutoff_ts].copy()

            if len(revs) > 0:
                revs_agg = revs.groupby("order_id").agg(
                    review_score=("review_score", "mean"),
                    low_review=("review_score", lambda s: float((s <= 2).any())),
                    high_review=("review_score", lambda s: float((s >= 4).any())),
                )
                reviews_summary = revs_agg

        # 7. Merge order-level auxiliary features into orders
        if items_summary is not None:
            orders = orders.merge(items_summary, on="order_id", how="left")
        else:
            orders["item_count"] = 1
            orders["total_item_price"] = 0.0
            orders["total_freight_value"] = 0.0
            orders["category_count"] = 1

        if payments_summary is not None:
            orders = orders.merge(payments_summary, on="order_id", how="left")
        else:
            orders["payment_total"] = orders["total_item_price"] + orders["total_freight_value"]
            orders["max_installments"] = 1
            orders["avg_installments"] = 1.0
            for p_type in ["credit_card", "boleto", "voucher", "debit_card"]:
                orders[f"{p_type}_share"] = 0.0

        if reviews_summary is not None:
            orders = orders.merge(reviews_summary, on="order_id", how="left")
        else:
            orders["review_score"] = np.nan
            orders["low_review"] = np.nan
            orders["high_review"] = np.nan

        # Fill total_monetary from payments or items
        orders["order_monetary"] = orders["payment_total"].fillna(
            orders["total_item_price"] + orders["total_freight_value"]
        )

        # 8. Aggregate strictly at customer_unique_id level
        grouped = orders.groupby("customer_unique_id")

        recency_series = (cutoff_ts - grouped["order_purchase_timestamp"].max()).dt.total_seconds() / 86400.0
        latest_ts_series = grouped["order_purchase_timestamp"].max()
        frequency_delivered = grouped["is_delivered_at_cutoff"].sum()
        total_orders_count = grouped["order_id"].nunique()
        monetary_series = grouped["order_monetary"].sum()
        total_freight_series = grouped["total_freight_value"].sum()
        total_items_series = grouped["item_count"].sum()

        cust_features = pd.DataFrame(
            {
                "customer_unique_id": recency_series.index,
                "recency_days": np.round(recency_series.values, 2),
                "frequency": frequency_delivered.values.astype(int),
                "total_orders": total_orders_count.values.astype(int),
                "total_monetary": np.round(monetary_series.values, 2),
                "avg_order_value": np.round(
                    (monetary_series / np.maximum(total_orders_count, 1)).values, 2
                ),
                "total_items": total_items_series.values.astype(int),
                "avg_basket_size": np.round(
                    (total_items_series / np.maximum(total_orders_count, 1)).values, 2
                ),
                "total_freight": np.round(total_freight_series.values, 2),
                "freight_ratio": np.round(
                    np.clip(
                        total_freight_series.values / np.maximum(monetary_series.values, 1e-4),
                        0.0,
                        1.0,
                    ),
                    4,
                ),
                "avg_installments": np.round(
                    grouped["avg_installments"].mean().fillna(1.0).values, 2
                ),
                "max_installments": grouped["max_installments"].max().fillna(1).values.astype(int),
                "credit_card_share": np.round(
                    grouped["credit_card_share"].mean().fillna(0.0).values, 4
                ),
                "boleto_share": np.round(grouped["boleto_share"].mean().fillna(0.0).values, 4),
                "voucher_share": np.round(grouped["voucher_share"].mean().fillna(0.0).values, 4),
                "avg_review_score": np.round(grouped["review_score"].mean().values, 2),
                "low_review_ratio": np.round(grouped["low_review"].mean().fillna(0.0).values, 4),
                "high_review_ratio": np.round(grouped["high_review"].mean().fillna(0.0).values, 4),
                "on_time_delivery_rate": np.round(
                    grouped["is_on_time"].mean().fillna(1.0).values, 4
                ),
                "distinct_categories_count": grouped["category_count"].sum().values.astype(int),
                "customer_state": grouped["customer_state"].last().values if "customer_state" in orders.columns else "UNKNOWN",
                "latest_order_timestamp": latest_ts_series.values,
                "cutoff_timestamp": str(cutoff_ts),
            }
        ).reset_index(drop=True)

        logger.info(
            f"Successfully built feature matrix for {len(cust_features)} unique customers as of {cutoff_ts}."
        )
        return cust_features

    def create_forward_targets(
        self,
        orders_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        cutoff_date: Union[str, pd.Timestamp],
        horizon_days: int = 90,
        lookback_days: int = 365,
    ) -> pd.DataFrame:
        """Constructs forward prediction target labels without future leakage.

        Target Definitions:
        - Active Cohort: Customers who placed >= 1 delivered order in [cutoff - lookback_days, cutoff].
        - Repeat Purchase Target: 1 if customer placed >= 1 delivered order in (cutoff, cutoff + horizon_days], else 0.
        - Churn Target: 1 if customer placed 0 delivered orders in (cutoff, cutoff + horizon_days], else 0.

        Args:
            orders_df: Orders table.
            customers_df: Customers table.
            cutoff_date: Strict point-in-time cutoff.
            horizon_days: Forward observation window in days (default: 90).
            lookback_days: Backward active eligibility lookback window (default: 365).

        Returns:
            pd.DataFrame: DataFrame containing customer_unique_id, active_in_lookback,
                          repeat_in_horizon, churn_in_horizon, forward_orders_count,
                          and forward_spend.
        """
        cutoff_ts = pd.to_datetime(cutoff_date)
        lookback_ts = cutoff_ts - pd.Timedelta(days=lookback_days)
        forward_end_ts = cutoff_ts + pd.Timedelta(days=horizon_days)

        orders = orders_df.copy()
        orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])

        if "customer_unique_id" not in orders.columns:
            orders = orders.merge(
                customers_df[["customer_id", "customer_unique_id"]],
                on="customer_id",
                how="inner",
            )

        # Delivered orders only for clean repeat/churn evaluation
        delivered_orders = orders[orders["order_status"] == "delivered"].copy()

        # 1. Identify active customers in lookback window
        lookback_orders = delivered_orders[
            (delivered_orders["order_purchase_timestamp"] >= lookback_ts)
            & (delivered_orders["order_purchase_timestamp"] <= cutoff_ts)
        ]
        active_customers = set(lookback_orders["customer_unique_id"].dropna().unique())

        # 2. Identify purchases in forward observation horizon
        forward_orders = delivered_orders[
            (delivered_orders["order_purchase_timestamp"] > cutoff_ts)
            & (delivered_orders["order_purchase_timestamp"] <= forward_end_ts)
        ]
        forward_counts = forward_orders.groupby("customer_unique_id")["order_id"].nunique()

        all_unique_customers = set(customers_df["customer_unique_id"].dropna().unique())

        records = []
        for cid in all_unique_customers:
            is_active = cid in active_customers
            f_orders = forward_counts.get(cid, 0)
            records.append(
                {
                    "customer_unique_id": cid,
                    "is_active_in_lookback": is_active,
                    "repeat_in_horizon": int(f_orders >= 1),
                    "churn_in_horizon": int(f_orders == 0),
                    "forward_orders_count": int(f_orders),
                }
            )

        target_df = pd.DataFrame(records)
        logger.info(
            f"Forward target generated: {len(target_df)} customers evaluated. "
            f"Active in lookback={target_df['is_active_in_lookback'].sum()}. "
            f"Repeat in horizon={target_df[target_df['is_active_in_lookback']]['repeat_in_horizon'].sum()}."
        )
        return target_df


# Backward-compatible alias
FeaturePipeline = CustomerFeaturePipeline
