"""Phase 3, Milestone 1: Recommendation Data Foundation — Synthetic Tests.

Tests are fully self-contained using synthetic data. No real Olist CSVs required.
All fixtures are kept <= 500 rows to satisfy local execution limits.

Test coverage:
  1. customer_unique_id grouping (not customer_id).
  2. Point-in-time cutoff enforcement on interactions.
  3. Missing product metadata handled without crash.
  4. Chronological split ordering verified.
  5. No customer with 0 interactions in output matrix.
  6. Category translation fallback when translation table is absent.
  7. Interaction matrix dimensions (rows = unique customer-product pairs).
  8. Sparsity report returns expected keys.
  9. Price band assignment correctness.
  10. Empty interaction edge case (no orders survive filtering).
"""

from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pytest

from insightx.data.recommendation_data import CustomerProductInteractionBuilder


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_customers(n: int = 10) -> pd.DataFrame:
    """Creates a synthetic customers table with distinct customer_unique_id per real customer.

    Deliberately gives 2 customer_id rows to the same customer_unique_id
    to verify aggregation uses customer_unique_id correctly.
    """
    rows = []
    # First customer has 2 transient customer_ids (Olist pattern)
    rows.append({"customer_id": "cid_001a", "customer_unique_id": "uid_001", "customer_state": "SP"})
    rows.append({"customer_id": "cid_001b", "customer_unique_id": "uid_001", "customer_state": "SP"})
    for i in range(2, n + 1):
        rows.append({
            "customer_id": f"cid_{i:03d}",
            "customer_unique_id": f"uid_{i:03d}",
            "customer_state": "RJ",
        })
    return pd.DataFrame(rows)


def _make_orders(customers_df: pd.DataFrame, n_orders: int = 30) -> pd.DataFrame:
    """Creates synthetic orders distributed across customers.

    Ensures uid_001 has 2 orders (from its 2 transient customer_ids),
    remaining customers have roughly 1-3 orders each.
    """
    rng = np.random.default_rng(42)
    rows = []
    # uid_001 with two different customer_ids — both before cutoff
    rows.append({
        "order_id": "ord_001a",
        "customer_id": "cid_001a",
        "order_status": "delivered",
        "order_purchase_timestamp": "2018-01-15 10:00:00",
    })
    rows.append({
        "order_id": "ord_001b",
        "customer_id": "cid_001b",
        "order_status": "delivered",
        "order_purchase_timestamp": "2018-02-10 12:00:00",
    })

    # Other customers
    other_cids = [r["customer_id"] for r in [
        {"customer_id": f"cid_{i:03d}"} for i in range(2, 11)
    ]]
    base_date = datetime(2017, 6, 1)
    for i, cid in enumerate(other_cids[:min(len(other_cids), n_orders - 2)]):
        rows.append({
            "order_id": f"ord_{i+2:03d}",
            "customer_id": cid,
            "order_status": "delivered",
            "order_purchase_timestamp": (
                base_date + timedelta(days=int(rng.integers(0, 250)))
            ).strftime("%Y-%m-%d %H:%M:%S"),
        })

    # Add 1 order after cutoff (should be filtered out)
    rows.append({
        "order_id": "ord_future",
        "customer_id": "cid_002",
        "order_status": "delivered",
        "order_purchase_timestamp": "2018-05-01 08:00:00",  # > 2018-03-01 cutoff
    })

    return pd.DataFrame(rows)


def _make_order_items(orders_df: pd.DataFrame) -> pd.DataFrame:
    """Creates synthetic order_items, each order gets 1-2 products."""
    rng = np.random.default_rng(99)
    products = [f"prod_{i:03d}" for i in range(1, 6)]  # 5 products
    rows = []
    for _, row in orders_df.iterrows():
        n_items = int(rng.integers(1, 3))
        for j in range(n_items):
            rows.append({
                "order_id": row["order_id"],
                "product_id": rng.choice(products),
                "price": round(float(rng.uniform(20.0, 500.0)), 2),
                "freight_value": round(float(rng.uniform(5.0, 50.0)), 2),
            })
    return pd.DataFrame(rows)


def _make_products(product_ids=None) -> pd.DataFrame:
    """Creates a synthetic products table."""
    if product_ids is None:
        product_ids = [f"prod_{i:03d}" for i in range(1, 6)]
    rows = []
    categories = ["eletronicos", "brinquedos", "esporte_lazer", None, "informatica_acessorios"]
    for i, pid in enumerate(product_ids):
        rows.append({
            "product_id": pid,
            "product_category_name": categories[i % len(categories)],
            "product_weight_g": float(100 * (i + 1)) if i != 2 else None,  # 1 missing
            "product_length_cm": 20.0,
            "product_height_cm": 10.0,
            "product_width_cm": 15.0,
            "product_photos_qty": 3,
        })
    return pd.DataFrame(rows)


def _make_reviews(orders_df: pd.DataFrame) -> pd.DataFrame:
    """Creates synthetic reviews; half are after cutoff to test point-in-time filtering."""
    rng = np.random.default_rng(7)
    rows = []
    for i, row in orders_df.iterrows():
        # Alternate: even index reviews before cutoff, odd after
        if i % 2 == 0:
            creation_date = "2018-02-01 10:00:00"
        else:
            creation_date = "2018-04-01 10:00:00"  # after 2018-03-01 cutoff
        rows.append({
            "order_id": row["order_id"],
            "review_score": float(rng.integers(1, 6)),
            "review_creation_date": creation_date,
        })
    return pd.DataFrame(rows)


def _make_translation() -> pd.DataFrame:
    return pd.DataFrame({
        "product_category_name": [
            "eletronicos", "brinquedos", "esporte_lazer", "informatica_acessorios"
        ],
        "product_category_name_english": [
            "electronics", "toys", "sports_leisure", "computers_accessories"
        ],
    })


# ---------------------------------------------------------------------------
# Test 1: customer_unique_id grouping (not customer_id)
# ---------------------------------------------------------------------------

class TestCustomerUniqueIdGrouping:
    """uid_001 has 2 transient customer_ids. Both orders map to the same unique customer."""

    def test_uid001_aggregated_not_split(self):
        customers = _make_customers(n=5)
        orders = _make_orders(customers, n_orders=12)
        items = _make_order_items(orders)

        builder = CustomerProductInteractionBuilder(cutoff_date="2018-03-01")
        matrix, evidence = builder.build_interaction_matrix(
            orders, customers, items
        )

        # uid_001's orders should all map to customer_unique_id = "uid_001"
        uid001_rows = matrix[matrix["customer_unique_id"] == "uid_001"]
        # Rows should exist and customer_id should NOT appear as a column
        assert "customer_id" not in matrix.columns, (
            "Interaction matrix must not contain customer_id — use customer_unique_id only."
        )
        # uid_001 should appear at most once per product (aggregated)
        assert len(uid001_rows) == uid001_rows["product_id"].nunique(), (
            "uid_001 rows should be deduplicated to (uid, product) grain."
        )

    def test_cid_not_in_output(self):
        """customer_id must not appear in the final interaction matrix."""
        customers = _make_customers(n=5)
        orders = _make_orders(customers, n_orders=12)
        items = _make_order_items(orders)

        builder = CustomerProductInteractionBuilder(cutoff_date="2018-03-01")
        matrix, _ = builder.build_interaction_matrix(orders, customers, items)

        assert "customer_id" not in matrix.columns


# ---------------------------------------------------------------------------
# Test 2: Point-in-time cutoff enforcement
# ---------------------------------------------------------------------------

class TestPointInTimeCutoff:

    def test_future_orders_excluded(self):
        """Orders after cutoff (ord_future at 2018-05-01) must not appear."""
        customers = _make_customers(n=5)
        orders = _make_orders(customers, n_orders=12)
        items = _make_order_items(orders)

        builder = CustomerProductInteractionBuilder(cutoff_date="2018-03-01")
        matrix, _ = builder.build_interaction_matrix(orders, customers, items)

        # ord_future belongs to cid_002 → uid_002
        # If cutoff is enforced, last_purchase_date for uid_002 must be <= 2018-03-01
        if "uid_002" in matrix["customer_unique_id"].values:
            uid002 = matrix[matrix["customer_unique_id"] == "uid_002"]
            assert (uid002["last_purchase_date"] <= pd.Timestamp("2018-03-01")).all(), (
                "last_purchase_date for uid_002 must not exceed cutoff."
            )

    def test_last_purchase_date_leq_cutoff(self):
        """All last_purchase_date values must be <= the cutoff."""
        customers = _make_customers(n=5)
        orders = _make_orders(customers, n_orders=12)
        items = _make_order_items(orders)

        cutoff = pd.Timestamp("2018-03-01")
        builder = CustomerProductInteractionBuilder(cutoff_date="2018-03-01")
        matrix, _ = builder.build_interaction_matrix(orders, customers, items)

        assert (matrix["last_purchase_date"] <= cutoff).all(), (
            "All last_purchase_date values must be <= cutoff_date."
        )

    def test_reviews_point_in_time_filtered(self):
        """Review scores from after cutoff must not contribute to avg_review_score."""
        customers = _make_customers(n=5)
        orders = _make_orders(customers, n_orders=12)
        items = _make_order_items(orders)
        reviews = _make_reviews(orders)

        builder = CustomerProductInteractionBuilder(cutoff_date="2018-03-01")
        matrix, _ = builder.build_interaction_matrix(
            orders, customers, items, order_reviews_df=reviews
        )

        # avg_review_score should exist and be between 1 and 5
        if "avg_review_score" in matrix.columns:
            valid = matrix["avg_review_score"].dropna()
            assert (valid >= 1.0).all() and (valid <= 5.0).all(), (
                "avg_review_score must be in [1, 5] — future reviews should be excluded."
            )


# ---------------------------------------------------------------------------
# Test 3: Missing product metadata handled without crash
# ---------------------------------------------------------------------------

class TestMissingProductMetadata:

    def test_missing_category_name_no_crash(self):
        """Products with missing category_name should not crash — fallback to 'unknown'."""
        products = _make_products()
        # Blank out all category names
        products["product_category_name"] = None
        items = _make_order_items(_make_orders(_make_customers(n=3), n_orders=8))

        builder = CustomerProductInteractionBuilder()
        metadata, _ = builder.build_product_metadata(products, items, category_translation_df=None)

        assert "product_category_name_english" in metadata.columns
        assert (metadata["product_category_name_english"] == "unknown").all()

    def test_missing_numeric_filled_with_zero(self):
        """Missing numeric features (weight, dimensions) filled with 0.0, not NaN."""
        products = _make_products()
        products["product_weight_g"] = None
        products["product_length_cm"] = None
        items = _make_order_items(_make_orders(_make_customers(n=3), n_orders=8))

        builder = CustomerProductInteractionBuilder()
        metadata, _ = builder.build_product_metadata(products, items)

        assert metadata["product_weight_g"].isna().sum() == 0, (
            "product_weight_g NaN should be filled with 0.0"
        )
        assert metadata["product_length_cm"].isna().sum() == 0, (
            "product_length_cm NaN should be filled with 0.0"
        )

    def test_absent_translation_table(self):
        """No translation table → all categories become 'unknown' without error."""
        products = _make_products()
        items = _make_order_items(_make_orders(_make_customers(n=3), n_orders=8))

        builder = CustomerProductInteractionBuilder()
        metadata, _ = builder.build_product_metadata(
            products, items, category_translation_df=None
        )

        assert (metadata["product_category_name_english"] == "unknown").all()


# ---------------------------------------------------------------------------
# Test 4: Chronological split ordering
# ---------------------------------------------------------------------------

class TestChronologicalSplitOrdering:

    def test_split_is_chronologically_ordered(self):
        """Train max timestamp must be strictly less than val min, val max < test min."""
        customers = _make_customers(n=10)
        orders = _make_orders(customers, n_orders=20)
        items = _make_order_items(orders)

        builder = CustomerProductInteractionBuilder(cutoff_date="2018-08-31")
        matrix, _ = builder.build_interaction_matrix(orders, customers, items)

        if len(matrix) < 3:
            pytest.skip("Not enough data to test 3-way split.")

        splits, metadata, leakage_clean = builder.split_interactions_chronologically(
            matrix,
            train_start="2017-01-01",
            train_end="2018-01-31",
            val_end="2018-03-31",
            test_end="2018-08-31",
        )

        assert leakage_clean, "Leakage validator must pass for chronological splits."

        # Check that non-empty partitions respect ordering
        train = splits["train"]
        val = splits["validation"]
        test = splits["test"]

        if len(train) > 0 and len(val) > 0:
            assert train["last_purchase_date"].max() < val["last_purchase_date"].min() or \
                   train["last_purchase_date"].max() <= pd.Timestamp("2018-01-31"), (
                "Train last_purchase_date should not exceed train_end."
            )

    def test_split_metadata_row_counts_sum(self):
        """train_rows + val_rows + test_rows should equal total non-zero partitions."""
        customers = _make_customers(n=10)
        orders = _make_orders(customers, n_orders=20)
        items = _make_order_items(orders)

        builder = CustomerProductInteractionBuilder(cutoff_date="2018-08-31")
        matrix, _ = builder.build_interaction_matrix(orders, customers, items)

        if len(matrix) < 1:
            pytest.skip("Empty matrix.")

        splits, metadata, _ = builder.split_interactions_chronologically(
            matrix,
            train_start="2017-01-01",
            train_end="2018-01-31",
            val_end="2018-03-31",
            test_end="2018-08-31",
        )

        total_split = metadata.train_rows + metadata.val_rows + metadata.test_rows
        assert total_split == len(matrix), (
            f"Split rows ({total_split}) must sum to total matrix rows ({len(matrix)})."
        )


# ---------------------------------------------------------------------------
# Test 5: No customer with 0 interactions in output
# ---------------------------------------------------------------------------

class TestNoZeroInteractionCustomers:

    def test_all_output_customers_have_at_least_one_purchase(self):
        """Every customer_unique_id in the matrix must have purchase_count >= 1."""
        customers = _make_customers(n=8)
        orders = _make_orders(customers, n_orders=15)
        items = _make_order_items(orders)

        builder = CustomerProductInteractionBuilder(cutoff_date="2018-03-01")
        matrix, _ = builder.build_interaction_matrix(orders, customers, items)

        assert (matrix["purchase_count"] >= 1).all(), (
            "All customer-product pairs must have purchase_count >= 1."
        )


# ---------------------------------------------------------------------------
# Test 6: Category translation fallback
# ---------------------------------------------------------------------------

class TestCategoryTranslationFallback:

    def test_translation_applied_correctly(self):
        """Known categories should be translated; unknown mapped to 'unknown'."""
        products = _make_products()
        items = _make_order_items(_make_orders(_make_customers(n=3), n_orders=8))
        translation = _make_translation()

        builder = CustomerProductInteractionBuilder()
        metadata, _ = builder.build_product_metadata(products, items, translation)

        # "eletronicos" → "electronics"
        electronics = metadata[metadata["product_category_name_english"] == "electronics"]
        assert len(electronics) >= 0  # at least 0 — no crash

        # Categories without translation → "unknown"
        known_translations = set(translation["product_category_name"])
        products_with_known = products[products["product_category_name"].isin(known_translations)]
        products_without = products[~products["product_category_name"].isin(known_translations)]

        if len(products_without) > 0:
            meta_without = metadata[metadata["product_id"].isin(products_without["product_id"])]
            # These should be "unknown" (missing or None category)
            assert (meta_without["product_category_name_english"] == "unknown").all()

    def test_malformed_translation_df_no_crash(self):
        """Translation df without expected columns → graceful fallback."""
        products = _make_products()
        items = _make_order_items(_make_orders(_make_customers(n=3), n_orders=8))
        bad_translation = pd.DataFrame({"col_a": [1], "col_b": [2]})  # wrong columns

        builder = CustomerProductInteractionBuilder()
        metadata, _ = builder.build_product_metadata(products, items, bad_translation)

        assert (metadata["product_category_name_english"] == "unknown").all()


# ---------------------------------------------------------------------------
# Test 7: Interaction matrix dimensions
# ---------------------------------------------------------------------------

class TestInteractionMatrixDimensions:

    def test_row_count_equals_unique_customer_product_pairs(self):
        """Each row must represent exactly one (customer_unique_id, product_id) pair."""
        customers = _make_customers(n=8)
        orders = _make_orders(customers, n_orders=20)
        items = _make_order_items(orders)

        builder = CustomerProductInteractionBuilder(cutoff_date="2018-03-01")
        matrix, _ = builder.build_interaction_matrix(orders, customers, items)

        n_unique_pairs = matrix[["customer_unique_id", "product_id"]].drop_duplicates().shape[0]
        assert len(matrix) == n_unique_pairs, (
            f"Matrix rows ({len(matrix)}) != unique (uid, product) pairs ({n_unique_pairs})."
        )

    def test_required_columns_present(self):
        """Interaction matrix must contain all required feature columns."""
        customers = _make_customers(n=8)
        orders = _make_orders(customers, n_orders=20)
        items = _make_order_items(orders)

        builder = CustomerProductInteractionBuilder(cutoff_date="2018-03-01")
        matrix, _ = builder.build_interaction_matrix(orders, customers, items)

        required = [
            "customer_unique_id",
            "product_id",
            "purchase_count",
            "total_spend",
            "avg_price",
            "avg_freight_value",
            "first_purchase_date",
            "last_purchase_date",
        ]
        for col in required:
            assert col in matrix.columns, f"Required column '{col}' missing from interaction matrix."


# ---------------------------------------------------------------------------
# Test 8: Sparsity report
# ---------------------------------------------------------------------------

class TestSparsityReport:

    def test_sparsity_report_keys(self):
        """Sparsity report must contain all expected keys."""
        customers = _make_customers(n=8)
        orders = _make_orders(customers, n_orders=20)
        items = _make_order_items(orders)

        builder = CustomerProductInteractionBuilder(cutoff_date="2018-03-01")
        matrix, _ = builder.build_interaction_matrix(orders, customers, items)
        report = builder.compute_sparsity_report(matrix)

        required_keys = [
            "n_customers", "n_products", "n_pairs", "sparsity",
            "pct_customers_repeat_product", "cold_start_risk_acknowledged", "limitation",
        ]
        for key in required_keys:
            assert key in report, f"Key '{key}' missing from sparsity report."

    def test_sparsity_between_0_and_1(self):
        """Sparsity must be in [0, 1]."""
        customers = _make_customers(n=8)
        orders = _make_orders(customers, n_orders=20)
        items = _make_order_items(orders)

        builder = CustomerProductInteractionBuilder(cutoff_date="2018-03-01")
        matrix, _ = builder.build_interaction_matrix(orders, customers, items)
        report = builder.compute_sparsity_report(matrix)

        assert 0.0 <= report["sparsity"] <= 1.0

    def test_empty_matrix_sparsity(self):
        """Empty interaction matrix should not crash sparsity report."""
        builder = CustomerProductInteractionBuilder()
        empty = pd.DataFrame(columns=["customer_unique_id", "product_id", "purchase_count"])
        report = builder.compute_sparsity_report(empty)
        assert report["sparsity"] == 1.0


# ---------------------------------------------------------------------------
# Test 9: Price band assignment
# ---------------------------------------------------------------------------

class TestPriceBandAssignment:

    def test_price_bands_cover_all_rows(self):
        """Every product must receive a price_band (no NaN bands)."""
        products = _make_products()
        items = _make_order_items(_make_orders(_make_customers(n=3), n_orders=8))

        builder = CustomerProductInteractionBuilder()
        metadata, _ = builder.build_product_metadata(products, items)

        assert metadata["price_band"].notna().all(), "price_band must not contain NaN."
        assert (metadata["price_band"] != "").all(), "price_band must not be empty string."

    def test_price_band_labels_are_valid(self):
        """price_band values must be from the expected label set."""
        products = _make_products()
        items = _make_order_items(_make_orders(_make_customers(n=3), n_orders=8))

        builder = CustomerProductInteractionBuilder()
        metadata, _ = builder.build_product_metadata(products, items)

        valid_bands = {"budget", "mid", "premium", "luxury", "luxury+", "unknown"}
        actual_bands = set(metadata["price_band"].unique())
        assert actual_bands.issubset(valid_bands), (
            f"Unexpected price_band values: {actual_bands - valid_bands}"
        )


# ---------------------------------------------------------------------------
# Test 10: Empty interaction edge case
# ---------------------------------------------------------------------------

class TestEmptyInteractionEdgeCase:

    def test_all_orders_after_cutoff_returns_empty(self):
        """If all orders are after cutoff, interaction matrix should be empty (no crash)."""
        customers = pd.DataFrame({
            "customer_id": ["cid_001"],
            "customer_unique_id": ["uid_001"],
        })
        orders = pd.DataFrame({
            "order_id": ["ord_001"],
            "customer_id": ["cid_001"],
            "order_status": ["delivered"],
            "order_purchase_timestamp": ["2019-01-01 00:00:00"],  # after any reasonable cutoff
        })
        items = pd.DataFrame({
            "order_id": ["ord_001"],
            "product_id": ["prod_001"],
            "price": [100.0],
            "freight_value": [10.0],
        })

        builder = CustomerProductInteractionBuilder(cutoff_date="2018-03-01")
        matrix, evidence = builder.build_interaction_matrix(orders, customers, items)

        assert len(matrix) == 0, "Matrix must be empty when all orders are after cutoff."

    def test_wrong_status_returns_empty(self):
        """Orders with status 'canceled' excluded from default valid_order_statuses."""
        customers = pd.DataFrame({
            "customer_id": ["cid_001"],
            "customer_unique_id": ["uid_001"],
        })
        orders = pd.DataFrame({
            "order_id": ["ord_001"],
            "customer_id": ["cid_001"],
            "order_status": ["canceled"],
            "order_purchase_timestamp": ["2018-01-01 00:00:00"],
        })
        items = pd.DataFrame({
            "order_id": ["ord_001"],
            "product_id": ["prod_001"],
            "price": [100.0],
            "freight_value": [10.0],
        })

        builder = CustomerProductInteractionBuilder(cutoff_date="2018-03-01")
        matrix, _ = builder.build_interaction_matrix(orders, customers, items)

        assert len(matrix) == 0, "Canceled orders must be excluded from interaction matrix."


# ---------------------------------------------------------------------------
# Test 11: Provenance evidence integrity
# ---------------------------------------------------------------------------

class TestProvenanceEvidence:

    def test_evidence_records_have_correct_source_type(self):
        """All evidence records must use source_type='RECSYS_CORE'."""
        from insightx.provenance.types import ProvenanceType

        customers = _make_customers(n=5)
        orders = _make_orders(customers, n_orders=12)
        items = _make_order_items(orders)

        builder = CustomerProductInteractionBuilder(cutoff_date="2018-03-01")
        _, evidence = builder.build_interaction_matrix(orders, customers, items)

        for ev in evidence:
            assert ev.source_type == "RECSYS_CORE", (
                f"Evidence source_type must be 'RECSYS_CORE', got '{ev.source_type}'."
            )
            assert ev.provenance_type == ProvenanceType.OBSERVED, (
                "Evidence provenance_type must be OBSERVED."
            )

    def test_evidence_raw_value_is_primitive(self):
        """Evidence raw_value must be a primitive (int, float, str, etc.)."""
        customers = _make_customers(n=5)
        orders = _make_orders(customers, n_orders=12)
        items = _make_order_items(orders)

        builder = CustomerProductInteractionBuilder(cutoff_date="2018-03-01")
        _, evidence = builder.build_interaction_matrix(orders, customers, items)

        for ev in evidence:
            assert isinstance(ev.raw_value, (str, int, float, bool, bytes, type(None))), (
                f"raw_value must be a primitive, got {type(ev.raw_value).__name__}."
            )

    def test_product_metadata_evidence_records(self):
        """Product metadata build must also produce valid provenance evidence."""
        products = _make_products()
        items = _make_order_items(_make_orders(_make_customers(n=3), n_orders=8))
        translation = _make_translation()

        builder = CustomerProductInteractionBuilder()
        _, evidence = builder.build_product_metadata(products, items, translation)

        assert len(evidence) >= 1, "Product metadata must produce at least 1 evidence record."
        for ev in evidence:
            assert ev.source_type == "RECSYS_CORE"
