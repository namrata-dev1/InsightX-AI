"""Phase 3, Milestone 1: Recommendation Data Foundation.

Builds clean customer-product interaction data and product metadata features
for future content-based and collaborative filtering recommendation models.

Key Design Decisions:
  - Uses customer_unique_id (not customer_id) for all customer aggregations.
  - All interactions are bounded by a strict point-in-time cutoff (no leakage).
  - Only 'delivered', 'shipped', and 'invoiced' orders are included by default.
  - Sparse repeat-purchase reality (~3.1% of Olist customers repeat) is explicitly
    reported, not papered over.
  - Product metadata is enriched with English category translations where available;
    missing translations fall back to 'unknown' without crashing.
  - Provenance for all outputs is tagged with StructuredEvidence (ProvenanceType.OBSERVED).

LIMITATIONS (explicitly documented):
  - ~96.9% of Olist customers have only 1 purchase → extreme sparsity for CF models.
  - No real profit margins available → CLV/revenue-based ranking only.
  - No model training in this module (Milestone 2+).
  - Cold-start problem for new customers/products is acknowledged but not solved here.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid
import warnings

import numpy as np
import pandas as pd

from insightx.data.splitter import TemporalSplitter
from insightx.data.validator import TemporalLeakageValidator
from insightx.provenance.evidence import StructuredEvidence
from insightx.provenance.types import ProvenanceType
from insightx.utils.logger import get_logger

logger = get_logger(__name__)

# Default valid order statuses for interaction data
DEFAULT_VALID_STATUSES = {"delivered", "shipped", "invoiced"}

# Default cutoff aligned with Phase 1/2 observation window
DEFAULT_CUTOFF = "2018-03-01 00:00:00"

# Default price band boundaries in BRL
DEFAULT_PRICE_BANDS = [0.0, 50.0, 150.0, 500.0]
DEFAULT_PRICE_BAND_LABELS = ["budget", "mid", "premium", "luxury"]


class CustomerProductInteractionBuilder:
    """Builds customer-product interaction datasets for recommendation model training.

    All customer aggregations use customer_unique_id to resolve Olist's
    transient customer_id (which changes per order).

    Args:
        cutoff_date: Point-in-time cutoff. Only orders at or before this date
            are included in interaction features. Prevents temporal leakage.
        valid_order_statuses: Set of order_status values treated as confirmed
            purchase interactions.
        min_interactions: Minimum purchase count per (customer_unique_id, product_id)
            pair to include in the interaction matrix.
    """

    def __init__(
        self,
        cutoff_date: str = DEFAULT_CUTOFF,
        valid_order_statuses: Optional[set] = None,
        min_interactions: int = 1,
    ):
        self.cutoff_date = pd.to_datetime(cutoff_date)
        self.valid_order_statuses = (
            valid_order_statuses
            if valid_order_statuses is not None
            else DEFAULT_VALID_STATUSES
        )
        self.min_interactions = max(1, int(min_interactions))
        logger.info(
            f"CustomerProductInteractionBuilder initialized: "
            f"cutoff={self.cutoff_date}, "
            f"valid_statuses={self.valid_order_statuses}, "
            f"min_interactions={self.min_interactions}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_interaction_matrix(
        self,
        orders_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        order_items_df: pd.DataFrame,
        order_reviews_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, List[StructuredEvidence]]:
        """Builds the customer-product interaction matrix.

        Joins orders → customers → order_items and aggregates at
        (customer_unique_id, product_id) granularity. Reviews are optionally
        joined only if their creation date is <= cutoff_date (point-in-time safe).

        Args:
            orders_df: Olist orders table (must include order_purchase_timestamp,
                order_status, customer_id).
            customers_df: Olist customers table (must include customer_id,
                customer_unique_id).
            order_items_df: Olist order items table (must include order_id,
                product_id, price, freight_value).
            order_reviews_df: Optional reviews table. Only review_score values
                with review_creation_date <= cutoff_date are included.

        Returns:
            Tuple of:
                - interaction_df: DataFrame with columns:
                    [customer_unique_id, product_id, purchase_count,
                     total_spend, avg_price, avg_freight_value,
                     avg_review_score (if reviews provided),
                     first_purchase_date, last_purchase_date]
                - evidence: List[StructuredEvidence] with provenance records.

        Raises:
            ValueError: If required columns are missing.
        """
        evidence: List[StructuredEvidence] = []

        # 1. Validate required columns
        self._validate_required_columns(
            orders_df,
            ["order_id", "customer_id", "order_status", "order_purchase_timestamp"],
            "orders_df",
        )
        self._validate_required_columns(
            customers_df,
            ["customer_id", "customer_unique_id"],
            "customers_df",
        )
        self._validate_required_columns(
            order_items_df,
            ["order_id", "product_id", "price", "freight_value"],
            "order_items_df",
        )

        # 2. Parse and apply point-in-time cutoff on orders
        orders = orders_df.copy()
        orders["order_purchase_timestamp"] = pd.to_datetime(
            orders["order_purchase_timestamp"]
        )
        pre_cutoff_mask = orders["order_purchase_timestamp"] <= self.cutoff_date
        status_mask = orders["order_status"].isin(self.valid_order_statuses)
        orders_filtered = orders[pre_cutoff_mask & status_mask].copy()

        n_raw = len(orders_df)
        n_filtered = len(orders_filtered)
        logger.info(
            f"Orders after cutoff+status filter: {n_filtered}/{n_raw} "
            f"(cutoff={self.cutoff_date}, statuses={self.valid_order_statuses})"
        )

        if n_filtered == 0:
            logger.warning(
                "No orders remain after cutoff and status filtering. "
                "Returning empty interaction matrix."
            )
            empty = pd.DataFrame(
                columns=[
                    "customer_unique_id",
                    "product_id",
                    "purchase_count",
                    "total_spend",
                    "avg_price",
                    "avg_freight_value",
                    "first_purchase_date",
                    "last_purchase_date",
                ]
            )
            return empty, evidence

        # 3. Resolve customer_unique_id (not customer_id)
        orders_with_uid = orders_filtered.merge(
            customers_df[["customer_id", "customer_unique_id"]],
            on="customer_id",
            how="left",
        )
        missing_uid = orders_with_uid["customer_unique_id"].isna().sum()
        if missing_uid > 0:
            logger.warning(
                f"{missing_uid} orders could not be resolved to customer_unique_id "
                "and will be excluded."
            )
        orders_with_uid = orders_with_uid.dropna(subset=["customer_unique_id"])

        # 4. Join order items
        interactions = orders_with_uid.merge(
            order_items_df[["order_id", "product_id", "price", "freight_value"]],
            on="order_id",
            how="inner",
        )

        if len(interactions) == 0:
            logger.warning(
                "No interactions after joining order_items. "
                "Returning empty interaction matrix."
            )
            empty = pd.DataFrame(
                columns=[
                    "customer_unique_id",
                    "product_id",
                    "purchase_count",
                    "total_spend",
                    "avg_price",
                    "avg_freight_value",
                    "first_purchase_date",
                    "last_purchase_date",
                ]
            )
            return empty, evidence

        # 5. Optionally join point-in-time-safe review scores
        review_col_available = False
        if order_reviews_df is not None and len(order_reviews_df) > 0:
            reviews = order_reviews_df.copy()
            if "review_creation_date" in reviews.columns:
                reviews["review_creation_date"] = pd.to_datetime(
                    reviews["review_creation_date"]
                )
                # Point-in-time: only reviews created before or at cutoff
                reviews = reviews[
                    reviews["review_creation_date"] <= self.cutoff_date
                ].copy()
                if len(reviews) > 0 and "review_score" in reviews.columns:
                    interactions = interactions.merge(
                        reviews[["order_id", "review_score"]],
                        on="order_id",
                        how="left",
                    )
                    review_col_available = True
                    logger.info(
                        f"Review scores joined (point-in-time safe): "
                        f"{reviews['review_score'].notna().sum()} valid reviews."
                    )

        # 6. Aggregate to (customer_unique_id, product_id) grain
        agg_dict: Dict[str, Any] = {
            "order_id": "nunique",       # purchase_count = distinct orders
            "price": ["sum", "mean"],
            "freight_value": "mean",
            "order_purchase_timestamp": ["min", "max"],
        }
        if review_col_available:
            agg_dict["review_score"] = "mean"

        grouped = interactions.groupby(
            ["customer_unique_id", "product_id"], as_index=False
        ).agg(agg_dict)

        # Flatten MultiIndex columns
        grouped.columns = [
            "_".join(c).strip("_") if isinstance(c, tuple) else c
            for c in grouped.columns
        ]

        rename_map = {
            "order_id_nunique": "purchase_count",
            "price_sum": "total_spend",
            "price_mean": "avg_price",
            "freight_value_mean": "avg_freight_value",
            "order_purchase_timestamp_min": "first_purchase_date",
            "order_purchase_timestamp_max": "last_purchase_date",
        }
        if review_col_available:
            rename_map["review_score_mean"] = "avg_review_score"

        grouped.rename(columns=rename_map, inplace=True)

        # 7. Apply min_interactions filter
        if self.min_interactions > 1:
            before = len(grouped)
            grouped = grouped[grouped["purchase_count"] >= self.min_interactions]
            logger.info(
                f"min_interactions={self.min_interactions} filter: "
                f"{len(grouped)}/{before} customer-product pairs retained."
            )

        # Ensure round-trippable date columns
        grouped["first_purchase_date"] = pd.to_datetime(grouped["first_purchase_date"])
        grouped["last_purchase_date"] = pd.to_datetime(grouped["last_purchase_date"])

        # 8. Build provenance evidence
        n_customers = int(grouped["customer_unique_id"].nunique())
        n_products = int(grouped["product_id"].nunique())
        n_pairs = len(grouped)

        sparsity = self._compute_sparsity(n_customers, n_products, n_pairs)

        evidence.append(
            self._make_evidence(
                claim_id="interaction_matrix_customer_count",
                table_name="interaction_matrix",
                column_name="customer_unique_id",
                key="n_unique_customers",
                value=n_customers,
                unit="count",
                provenance="COUNT(DISTINCT customer_unique_id) from interaction matrix",
                linkage=(
                    f"Interaction matrix contains {n_customers} unique customers "
                    f"(resolved via customer_unique_id, not customer_id)."
                ),
            )
        )
        evidence.append(
            self._make_evidence(
                claim_id="interaction_matrix_product_count",
                table_name="interaction_matrix",
                column_name="product_id",
                key="n_unique_products",
                value=n_products,
                unit="count",
                provenance="COUNT(DISTINCT product_id) from interaction matrix",
                linkage=(
                    f"Interaction matrix contains {n_products} unique products."
                ),
            )
        )
        evidence.append(
            self._make_evidence(
                claim_id="interaction_matrix_sparsity",
                table_name="interaction_matrix",
                column_name="purchase_count",
                key="sparsity_ratio",
                value=round(sparsity, 6),
                unit="ratio",
                provenance=(
                    "1 - (n_observed_pairs / (n_customers * n_products))"
                ),
                linkage=(
                    f"Interaction matrix sparsity={sparsity:.4%}. "
                    f"Observed pairs={n_pairs} of {n_customers * n_products} possible."
                ),
            )
        )

        logger.info(
            f"Interaction matrix built: {n_pairs} customer-product pairs, "
            f"{n_customers} customers, {n_products} products, "
            f"sparsity={sparsity:.4%}"
        )
        return grouped, evidence

    def build_product_metadata(
        self,
        products_df: pd.DataFrame,
        order_items_df: pd.DataFrame,
        category_translation_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, List[StructuredEvidence]]:
        """Builds product metadata feature table for content-based recommendations.

        Args:
            products_df: Olist products table (product_id, product_category_name,
                product_weight_g, product_length_cm, product_height_cm,
                product_width_cm, product_photos_qty, etc.).
            order_items_df: Used to compute per-product median price and
                freight ratio.
            category_translation_df: Optional. Maps product_category_name →
                product_category_name_english. Falls back to 'unknown' if absent.

        Returns:
            Tuple of:
                - metadata_df: DataFrame with columns:
                    [product_id, product_category_name_english, price_band,
                     avg_freight_ratio, product_weight_g, product_length_cm,
                     product_height_cm, product_width_cm, product_photos_qty,
                     has_metadata]
                - evidence: List[StructuredEvidence] with provenance records.
        """
        evidence: List[StructuredEvidence] = []

        self._validate_required_columns(
            products_df, ["product_id"], "products_df"
        )
        self._validate_required_columns(
            order_items_df, ["order_id", "product_id", "price", "freight_value"],
            "order_items_df",
        )

        # 1. Compute per-product price and freight statistics from order_items
        price_stats = (
            order_items_df.groupby("product_id")
            .agg(
                median_price=("price", "median"),
                mean_freight_value=("freight_value", "mean"),
                mean_price_for_freight=("price", "mean"),
            )
            .reset_index()
        )
        # avg_freight_ratio = avg(freight / price) — avoid division by zero
        price_stats["avg_freight_ratio"] = np.where(
            price_stats["mean_price_for_freight"] > 0,
            price_stats["mean_freight_value"] / price_stats["mean_price_for_freight"],
            0.0,
        )

        # 2. Merge with product catalog
        meta = products_df.copy()
        meta = meta.merge(price_stats[["product_id", "median_price", "avg_freight_ratio"]],
                          on="product_id", how="left")

        # 3. Resolve English category name
        if category_translation_df is not None and len(category_translation_df) > 0:
            if (
                "product_category_name" in category_translation_df.columns
                and "product_category_name_english" in category_translation_df.columns
            ):
                meta = meta.merge(
                    category_translation_df[
                        ["product_category_name", "product_category_name_english"]
                    ],
                    on="product_category_name",
                    how="left",
                )
            else:
                logger.warning(
                    "category_translation_df missing expected columns. "
                    "Defaulting all categories to 'unknown'."
                )
                meta["product_category_name_english"] = "unknown"
        else:
            logger.info(
                "No category translation table provided. "
                "Setting product_category_name_english = 'unknown'."
            )
            meta["product_category_name_english"] = "unknown"

        # Fill missing translations
        meta["product_category_name_english"] = (
            meta["product_category_name_english"]
            .fillna("unknown")
            .astype(str)
        )

        # 4. Price band discretization (rank-percentile safe for small catalogs)
        meta = self._assign_price_bands(meta, "median_price")

        # 5. Fill numeric metadata gaps
        numeric_cols = [
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
            "product_photos_qty",
        ]
        for col in numeric_cols:
            if col not in meta.columns:
                meta[col] = 0.0
            else:
                meta[col] = pd.to_numeric(meta[col], errors="coerce").fillna(0.0)

        # 6. Flag products with incomplete metadata
        key_meta_cols = ["product_category_name", "product_weight_g"]
        available_key_cols = [c for c in key_meta_cols if c in products_df.columns]
        if available_key_cols:
            meta["has_metadata"] = ~products_df.set_index("product_id").reindex(
                meta["product_id"]
            )[available_key_cols].isna().any(axis=1).values
        else:
            meta["has_metadata"] = False

        # 7. Select and order final columns
        output_cols = [
            "product_id",
            "product_category_name_english",
            "price_band",
            "median_price",
            "avg_freight_ratio",
        ] + numeric_cols + ["has_metadata"]
        for col in output_cols:
            if col not in meta.columns:
                meta[col] = 0.0 if col not in ("product_id", "product_category_name_english", "price_band") else "unknown"

        metadata_df = meta[output_cols].copy()

        # 8. Provenance
        n_products = len(metadata_df)
        n_with_metadata = int(metadata_df["has_metadata"].sum())
        n_missing_metadata = n_products - n_with_metadata

        evidence.append(
            self._make_evidence(
                claim_id="product_metadata_total_products",
                table_name="product_metadata",
                column_name="product_id",
                key="n_products",
                value=n_products,
                unit="count",
                provenance="COUNT(DISTINCT product_id) from products_df",
                linkage=(
                    f"Product metadata table contains {n_products} products. "
                    f"{n_missing_metadata} products have incomplete metadata "
                    f"(has_metadata=False)."
                ),
            )
        )
        evidence.append(
            self._make_evidence(
                claim_id="product_metadata_coverage",
                table_name="product_metadata",
                column_name="has_metadata",
                key="metadata_coverage_ratio",
                value=round(n_with_metadata / n_products, 4) if n_products > 0 else 0.0,
                unit="ratio",
                provenance="SUM(has_metadata) / COUNT(*)",
                linkage=(
                    f"Metadata coverage: {n_with_metadata}/{n_products} products "
                    f"({100*n_with_metadata/n_products:.1f}% complete) based on "
                    "product_category_name and product_weight_g availability."
                ),
            )
        )

        logger.info(
            f"Product metadata built: {n_products} products, "
            f"{n_with_metadata} with complete metadata, "
            f"{n_missing_metadata} missing metadata."
        )
        return metadata_df, evidence

    def split_interactions_chronologically(
        self,
        interaction_df: pd.DataFrame,
        train_start: str = "2017-01-01",
        train_end: str = "2018-03-01",
        val_end: str = "2018-05-31",
        test_end: str = "2018-08-31",
        timestamp_col: str = "last_purchase_date",
    ) -> Tuple[Dict[str, pd.DataFrame], "SplitMetadata", bool]:
        """Chronologically splits interaction data into train/val/test.

        Uses last_purchase_date as the split key: a customer-product pair
        is assigned to the partition corresponding to their most recent
        purchase within the observation window.

        Args:
            interaction_df: Output of build_interaction_matrix().
            train_start: Inclusive start of training window.
            train_end: Exclusive end of training window.
            val_end: Exclusive end of validation window.
            test_end: Exclusive end of test window.
            timestamp_col: Column used for chronological ordering.

        Returns:
            Tuple of:
                - splits: Dict with 'train', 'validation', 'test' DataFrames.
                - split_metadata: SplitMetadata from TemporalSplitter.
                - leakage_clean: bool, True if no temporal leakage detected.
        """
        if len(interaction_df) == 0:
            logger.warning("Empty interaction_df passed to split. Returning empty splits.")
            from insightx.data.splitter import SplitMetadata
            empty = pd.DataFrame(columns=interaction_df.columns)
            metadata = SplitMetadata(
                train_start=train_start, train_end=train_end,
                val_start=train_end, val_end=val_end,
                test_start=val_end, test_end=test_end,
                train_rows=0, val_rows=0, test_rows=0,
                train_unique_customers=0, val_unique_customers=0,
                test_unique_customers=0,
                customer_overlap_train_val=0, customer_overlap_train_test=0,
            )
            return {"train": empty, "validation": empty, "test": empty}, metadata, True

        if timestamp_col not in interaction_df.columns:
            raise ValueError(
                f"timestamp_col='{timestamp_col}' not found in interaction_df. "
                f"Available columns: {list(interaction_df.columns)}"
            )

        # TemporalSplitter expects 'order_purchase_timestamp'
        df_for_split = interaction_df.rename(
            columns={timestamp_col: "order_purchase_timestamp"}
        ).copy()

        splitter = TemporalSplitter(
            train_start=train_start,
            train_end=train_end,
            val_end=val_end,
            test_end=test_end,
        )

        # Create a synthetic customers-like df for unique customer tracking
        cust_df = (
            interaction_df[["customer_unique_id"]]
            .drop_duplicates()
            .assign(customer_id=lambda x: x["customer_unique_id"])
        )

        splits_raw, metadata = splitter.split_orders(
            df_for_split,
            customers_df=cust_df,
            timestamp_col="order_purchase_timestamp",
        )

        # Rename back
        splits: Dict[str, pd.DataFrame] = {}
        for partition, split_df in splits_raw.items():
            splits[partition] = split_df.rename(
                columns={"order_purchase_timestamp": timestamp_col}
            )

        # Validate leakage
        leakage_clean = True
        try:
            leakage_rename = {
                k: v.rename(columns={timestamp_col: "order_purchase_timestamp"})
                for k, v in splits.items()
                if len(v) > 0
            }
            if (
                len(leakage_rename.get("train", [])) > 0
                and len(leakage_rename.get("validation", [])) > 0
                and len(leakage_rename.get("test", [])) > 0
            ):
                TemporalLeakageValidator.verify_chronological_splits(leakage_rename)
        except Exception as e:
            logger.error(f"Temporal leakage detected in interaction splits: {e}")
            leakage_clean = False

        logger.info(
            f"Interaction splits: "
            f"train={metadata.train_rows}, "
            f"val={metadata.val_rows}, "
            f"test={metadata.test_rows}"
        )
        return splits, metadata, leakage_clean

    def compute_sparsity_report(
        self,
        interaction_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Computes sparsity statistics for the interaction matrix.

        Returns:
            Dict with sparsity, n_customers, n_products, n_pairs,
            pct_customers_with_repeat_product_purchase, and
            cold_start_risk_acknowledged flag.
        """
        if len(interaction_df) == 0:
            return {
                "n_customers": 0,
                "n_products": 0,
                "n_pairs": 0,
                "sparsity": 1.0,
                "pct_customers_repeat_product": 0.0,
                "cold_start_risk_acknowledged": True,
                "limitation": "Olist has ~96.9% one-time buyers; CF models face extreme sparsity.",
            }

        n_customers = int(interaction_df["customer_unique_id"].nunique())
        n_products = int(interaction_df["product_id"].nunique())
        n_pairs = len(interaction_df)
        sparsity = self._compute_sparsity(n_customers, n_products, n_pairs)

        # Customers who purchased same product > once
        repeat_product_mask = interaction_df["purchase_count"] > 1
        n_repeat_product_customers = int(
            interaction_df.loc[repeat_product_mask, "customer_unique_id"].nunique()
        )
        pct_repeat = (
            n_repeat_product_customers / n_customers if n_customers > 0 else 0.0
        )

        return {
            "n_customers": n_customers,
            "n_products": n_products,
            "n_pairs": n_pairs,
            "sparsity": round(sparsity, 6),
            "pct_customers_repeat_product": round(pct_repeat, 4),
            "cold_start_risk_acknowledged": True,
            "limitation": (
                "Olist has ~96.9% one-time buyers. Collaborative filtering "
                "models will face extreme sparsity. Content-based and hybrid "
                "approaches are strongly recommended."
            ),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_required_columns(
        df: pd.DataFrame, required: List[str], df_name: str
    ) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"{df_name} is missing required columns: {missing}. "
                f"Available: {list(df.columns)}"
            )

    @staticmethod
    def _compute_sparsity(
        n_customers: int, n_products: int, n_observed: int
    ) -> float:
        """Returns 1 - (observed / possible) interaction density."""
        possible = n_customers * n_products
        if possible == 0:
            return 1.0
        return 1.0 - (n_observed / possible)

    @staticmethod
    def _assign_price_bands(
        df: pd.DataFrame,
        price_col: str,
        boundaries: Optional[List[float]] = None,
        labels: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Assigns a price_band label based on configurable BRL boundaries.

        Uses pd.cut (absolute boundaries) rather than pd.qcut (rank-based)
        because price distributions may be heavily right-skewed and
        pd.qcut fails on duplicate edges in small datasets.
        """
        if boundaries is None:
            boundaries = DEFAULT_PRICE_BANDS
        if labels is None:
            labels = DEFAULT_PRICE_BAND_LABELS

        df = df.copy()
        if price_col not in df.columns or df[price_col].isna().all():
            df["price_band"] = "unknown"
            return df

        prices = pd.to_numeric(df[price_col], errors="coerce").fillna(0.0)

        # Build cut bins: add -inf and +inf to ensure all values are covered
        bins = [-np.inf] + boundaries + [np.inf]
        all_labels = labels + ["luxury+"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                df["price_band"] = pd.cut(
                    prices,
                    bins=bins,
                    labels=all_labels,
                    right=True,
                    include_lowest=True,
                ).astype(str)
            except Exception:
                df["price_band"] = "unknown"

        df["price_band"] = df["price_band"].fillna("unknown").astype(str)
        return df

    @staticmethod
    def _make_evidence(
        claim_id: str,
        table_name: str,
        column_name: str,
        key: str,
        value: Any,
        unit: str,
        provenance: str,
        linkage: str,
    ) -> StructuredEvidence:
        """Creates a StructuredEvidence record with RECSYS_CORE source_type."""
        # Ensure raw_value is a PrimitiveValue
        if isinstance(value, (np.integer,)):
            value = int(value)
        elif isinstance(value, (np.floating,)):
            value = float(value)

        return StructuredEvidence(
            evidence_id=str(uuid.uuid4()),
            claim_id=claim_id,
            dataset_version_id="olist_phase3_m1",
            query_hash=str(uuid.uuid5(uuid.NAMESPACE_DNS, provenance)),
            execution_id=str(uuid.uuid4()),
            source_type="RECSYS_CORE",
            table_or_view_name=table_name,
            column_name=column_name,
            result_row_or_key=key,
            raw_value=value,
            unit=unit,
            calculation_provenance=provenance,
            provenance_type=ProvenanceType.OBSERVED,
            evidence_to_claim_linkage=linkage,
            created_at=datetime.utcnow(),
        )
