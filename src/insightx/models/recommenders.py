"""Phase 3, Milestone 2: Recommendation Baselines.

Implements three non-parametric recommendation strategies built exclusively
from training-period interaction data:

  1. PopularityRecommender  — global purchase-count ranking.
  2. CategoryRecommender    — category-affinity ranking per customer history.
  3. CoPurchaseRecommender  — item-item co-purchase counts (Jaccard similarity).

Design invariants:
  - All recommendation statistics are derived from training interactions only
    (the ``train`` split from Phase 3 M1 / TemporalSplitter).
  - Already-purchased products can be optionally excluded per customer.
  - New customers / new products / sparse histories are handled without crash.
  - Top-K is configurable; ties are broken deterministically (product_id lexicographic).
  - Scores are purchase-count derived — not predicted probabilities; they must not
    be interpreted as causal purchase likelihoods.
  - All outputs are tagged with StructuredEvidence (ProvenanceType.OBSERVED for
    training statistics; ProvenanceType.SIMULATION_ASSUMPTION for derived scores).

LIMITATIONS (explicitly documented):
  - Popularity bias: popular products dominate; niche items are under-recommended.
  - Category recommender requires at least 1 known product category per customer.
    Customers with 100% 'unknown' category history fall back to global popularity.
  - Co-purchase recommender requires at least 2 distinct products per customer order
    basket. With ~96.9% one-time Olist buyers, most customers have single-product
    histories, yielding very sparse co-purchase graphs.
  - No personalisation beyond category and co-purchase signals.
  - Scores are not calibrated probabilities.
"""

from datetime import datetime
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple
import uuid
import warnings

import numpy as np
import pandas as pd

from insightx.provenance.evidence import StructuredEvidence
from insightx.provenance.types import ProvenanceType
from insightx.utils.logger import get_logger

logger = get_logger(__name__)

# Sentinel value when a score cannot be computed
_SCORE_UNAVAILABLE = 0.0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_evidence(
    claim_id: str,
    table_name: str,
    column_name: str,
    key: str,
    value: Any,
    unit: str,
    provenance: str,
    linkage: str,
    provenance_type: ProvenanceType = ProvenanceType.OBSERVED,
    dataset_version_id: str = "olist_phase3_m2",
) -> StructuredEvidence:
    """Creates a StructuredEvidence record for a recommendation statistic."""
    if isinstance(value, (np.integer,)):
        value = int(value)
    elif isinstance(value, (np.floating,)):
        value = float(value)
    return StructuredEvidence(
        evidence_id=str(uuid.uuid4()),
        claim_id=claim_id,
        dataset_version_id=dataset_version_id,
        query_hash=str(uuid.uuid5(uuid.NAMESPACE_DNS, provenance)),
        execution_id=str(uuid.uuid4()),
        source_type="RECSYS_CORE",
        table_or_view_name=table_name,
        column_name=column_name,
        result_row_or_key=key,
        raw_value=value,
        unit=unit,
        calculation_provenance=provenance,
        provenance_type=provenance_type,
        evidence_to_claim_linkage=linkage,
        created_at=datetime.utcnow(),
    )


def _deterministic_topk(
    scored: pd.DataFrame,
    score_col: str,
    product_col: str,
    k: int,
) -> pd.DataFrame:
    """Returns top-K rows sorted by score desc, product_id asc (for tie-breaking).

    Deterministic: given the same input, always returns the same K products.
    """
    return (
        scored
        .sort_values([score_col, product_col], ascending=[False, True])
        .head(k)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 1. Popularity-Based Recommender
# ---------------------------------------------------------------------------

class PopularityRecommender:
    """Ranks products by total purchase count in the training period.

    Training source: interaction_train_df (output of
    CustomerProductInteractionBuilder.build_interaction_matrix() on train split).

    Score definition:
        popularity_score(product) = SUM(purchase_count) across all customers
                                    who bought that product in the training period.

    Tie-breaking: ascending product_id (lexicographic, deterministic).

    Args:
        top_k: Default number of recommendations to return per customer.
        exclude_purchased: If True, remove products the customer already bought
            from the training data before returning recommendations.
    """

    def __init__(self, top_k: int = 10, exclude_purchased: bool = True):
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self.top_k = top_k
        self.exclude_purchased = exclude_purchased

        # Fitted state
        self._popularity_df: Optional[pd.DataFrame] = None   # [product_id, popularity_score]
        self._customer_history: Optional[Dict[str, Set[str]]] = None
        self._n_train_pairs: int = 0
        self._n_train_customers: int = 0
        self._n_catalog: int = 0
        self._evidence: List[StructuredEvidence] = []
        self._fitted = False

    def fit(self, interaction_train_df: pd.DataFrame) -> "PopularityRecommender":
        """Computes global popularity scores from training interactions.

        Args:
            interaction_train_df: DataFrame with columns
                [customer_unique_id, product_id, purchase_count, ...].

        Returns:
            self (for chaining).
        """
        self._fitted = False
        self._evidence = []

        required = ["customer_unique_id", "product_id", "purchase_count"]
        missing = [c for c in required if c not in interaction_train_df.columns]
        if missing:
            raise ValueError(f"interaction_train_df missing columns: {missing}")

        df = interaction_train_df.copy()
        df["purchase_count"] = pd.to_numeric(df["purchase_count"], errors="coerce").fillna(0)

        # Aggregate purchase counts per product across all customers
        popularity = (
            df.groupby("product_id", as_index=False)["purchase_count"]
            .sum()
            .rename(columns={"purchase_count": "popularity_score"})
        )
        # Deterministic tie-breaking is applied at predict time
        self._popularity_df = popularity.copy()

        # Build customer history lookup for exclusion
        self._customer_history = (
            df.groupby("customer_unique_id")["product_id"]
            .apply(set)
            .to_dict()
        )

        self._n_train_pairs = len(df)
        self._n_train_customers = int(df["customer_unique_id"].nunique())
        self._n_catalog = int(popularity["product_id"].nunique())
        self._fitted = True

        # Provenance
        top_product = popularity.sort_values(
            ["popularity_score", "product_id"], ascending=[False, True]
        ).iloc[0]["product_id"] if len(popularity) > 0 else "N/A"

        self._evidence.append(_make_evidence(
            claim_id="popularity_catalog_size",
            table_name="popularity_scores",
            column_name="product_id",
            key="n_catalog_products",
            value=self._n_catalog,
            unit="count",
            provenance="COUNT(DISTINCT product_id) from training interaction matrix",
            linkage=f"Popularity catalog contains {self._n_catalog} products from training data.",
        ))
        self._evidence.append(_make_evidence(
            claim_id="popularity_top1_product",
            table_name="popularity_scores",
            column_name="product_id",
            key="top1_product_id",
            value=str(top_product),
            unit="product_id",
            provenance="argmax(SUM(purchase_count)) over training interactions",
            linkage=f"Most popular product in training period: {top_product}.",
        ))

        logger.info(
            f"PopularityRecommender fitted: {self._n_catalog} products, "
            f"{self._n_train_customers} training customers."
        )
        return self

    def recommend(
        self,
        customer_id: Optional[str] = None,
        k: Optional[int] = None,
        exclude_purchased: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Returns top-K popular products for a customer.

        Args:
            customer_id: customer_unique_id. If None, returns global top-K.
            k: Number of recommendations. Defaults to self.top_k.
            exclude_purchased: Override self.exclude_purchased for this call.

        Returns:
            DataFrame with columns [product_id, popularity_score, rank].
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before recommend().")

        k = k if k is not None else self.top_k
        exc = exclude_purchased if exclude_purchased is not None else self.exclude_purchased

        pool = self._popularity_df.copy()

        # Exclude already-purchased products for this customer
        if exc and customer_id is not None and customer_id in self._customer_history:
            purchased = self._customer_history[customer_id]
            pool = pool[~pool["product_id"].isin(purchased)]

        if len(pool) == 0:
            return pd.DataFrame(columns=["product_id", "popularity_score", "rank"])

        result = _deterministic_topk(pool, "popularity_score", "product_id", k)
        result["rank"] = range(1, len(result) + 1)
        return result[["product_id", "popularity_score", "rank"]]

    def recommend_batch(
        self,
        customer_ids: List[Optional[str]],
        k: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Returns recommendations for a list of customers.

        New customers (not in training data) receive global top-K (cold-start fallback).

        Returns:
            Dict mapping customer_unique_id → DataFrame[product_id, popularity_score, rank].
        """
        return {
            cid: self.recommend(customer_id=cid, k=k)
            for cid in customer_ids
        }

    @property
    def evidence(self) -> List[StructuredEvidence]:
        return list(self._evidence)

    @property
    def catalog_size(self) -> int:
        return self._n_catalog


# ---------------------------------------------------------------------------
# 2. Category-Based Recommender
# ---------------------------------------------------------------------------

class CategoryRecommender:
    """Recommends popular products from categories relevant to the customer's history.

    Strategy:
        1. For each customer, identify the product categories in their purchase history.
        2. Rank candidate products by (category_relevance × product_popularity) within
           those categories, using training-period purchase counts.
        3. category_relevance = fraction of customer's total purchases in that category.
        4. Ties broken by product_id (lexicographic, deterministic).
        5. Falls back to global popularity if no category history is available.

    Score definition (per candidate product p, customer c):
        category_score(c, p) = category_weight(c, cat(p))
                                × global_popularity(p)

        where category_weight(c, cat) = (# times c bought from cat in training)
                                        / (total purchases by c in training)

    Note: Scores are computed from purchase counts, not predicted probabilities.

    Args:
        top_k: Default top-K to return.
        exclude_purchased: Exclude already-purchased products.
        fallback_to_popularity: If True, customers with no category history
            receive global top-K as fallback.
    """

    def __init__(
        self,
        top_k: int = 10,
        exclude_purchased: bool = True,
        fallback_to_popularity: bool = True,
    ):
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self.top_k = top_k
        self.exclude_purchased = exclude_purchased
        self.fallback_to_popularity = fallback_to_popularity

        self._product_category: Optional[pd.Series] = None   # product_id → category
        self._product_popularity: Optional[pd.Series] = None  # product_id → count
        self._customer_category_weights: Optional[Dict[str, Dict[str, float]]] = None
        self._customer_history: Optional[Dict[str, Set[str]]] = None
        self._popularity_fallback: Optional[PopularityRecommender] = None
        self._n_catalog: int = 0
        self._evidence: List[StructuredEvidence] = []
        self._fitted = False

    def fit(
        self,
        interaction_train_df: pd.DataFrame,
        product_metadata_df: pd.DataFrame,
    ) -> "CategoryRecommender":
        """Builds category-popularity statistics from training data.

        Args:
            interaction_train_df: Training interaction matrix
                [customer_unique_id, product_id, purchase_count, ...].
            product_metadata_df: Product metadata from Phase 3 M1
                [product_id, product_category_name_english, ...].

        Returns:
            self.
        """
        self._fitted = False
        self._evidence = []

        for name, df, cols in [
            ("interaction_train_df", interaction_train_df,
             ["customer_unique_id", "product_id", "purchase_count"]),
            ("product_metadata_df", product_metadata_df,
             ["product_id", "product_category_name_english"]),
        ]:
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(f"{name} missing columns: {missing}")

        df = interaction_train_df.copy()
        df["purchase_count"] = pd.to_numeric(df["purchase_count"], errors="coerce").fillna(0)

        # Merge product category onto interactions
        cat_map = product_metadata_df.set_index("product_id")["product_category_name_english"]
        df["category"] = df["product_id"].map(cat_map).fillna("unknown")

        # Global product popularity (for scoring within categories)
        self._product_popularity = (
            df.groupby("product_id")["purchase_count"].sum()
        )

        # Product → category mapping (from metadata; one product → one category)
        self._product_category = cat_map.fillna("unknown")
        self._n_catalog = int(self._product_popularity.index.nunique())

        # Customer category weights
        # weight(c, cat) = SUM(purchase_count where category=cat) / SUM(purchase_count total)
        cust_cat = (
            df.groupby(["customer_unique_id", "category"])["purchase_count"]
            .sum()
            .reset_index()
        )
        cust_total = cust_cat.groupby("customer_unique_id")["purchase_count"].sum()
        cust_cat["weight"] = (
            cust_cat["purchase_count"]
            / cust_cat["customer_unique_id"].map(cust_total)
        )
        self._customer_category_weights = (
            cust_cat.groupby("customer_unique_id")
            .apply(lambda g: dict(zip(g["category"], g["weight"])), include_groups=False)
            .to_dict()
        )

        # Customer history for exclusion
        self._customer_history = (
            df.groupby("customer_unique_id")["product_id"]
            .apply(set)
            .to_dict()
        )

        # Fit popularity fallback
        self._popularity_fallback = PopularityRecommender(
            top_k=self.top_k, exclude_purchased=self.exclude_purchased
        )
        self._popularity_fallback.fit(interaction_train_df)

        self._fitted = True

        # Provenance
        n_categories = int(cat_map.fillna("unknown").nunique())
        self._evidence.append(_make_evidence(
            claim_id="category_recommender_n_categories",
            table_name="category_recommender_stats",
            column_name="category",
            key="n_distinct_categories",
            value=n_categories,
            unit="count",
            provenance="COUNT(DISTINCT product_category_name_english) from product_metadata_df",
            linkage=f"Category recommender trained on {n_categories} distinct product categories.",
        ))
        self._evidence.append(_make_evidence(
            claim_id="category_recommender_n_customers",
            table_name="category_recommender_stats",
            column_name="customer_unique_id",
            key="n_training_customers",
            value=int(df["customer_unique_id"].nunique()),
            unit="count",
            provenance="COUNT(DISTINCT customer_unique_id) from training interaction matrix",
            linkage=(
                f"Category weights computed for {int(df['customer_unique_id'].nunique())} "
                "training customers."
            ),
        ))

        logger.info(
            f"CategoryRecommender fitted: {self._n_catalog} products, "
            f"{n_categories} categories."
        )
        return self

    def recommend(
        self,
        customer_id: Optional[str] = None,
        k: Optional[int] = None,
        exclude_purchased: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Returns category-weighted top-K products for a customer.

        Args:
            customer_id: customer_unique_id. New customers get popularity fallback.
            k: Number of recommendations.
            exclude_purchased: Override for this call.

        Returns:
            DataFrame[product_id, category_score, category, rank].
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before recommend().")

        k = k if k is not None else self.top_k
        exc = exclude_purchased if exclude_purchased is not None else self.exclude_purchased

        cat_weights = (
            self._customer_category_weights.get(customer_id, {})
            if customer_id is not None
            else {}
        )

        # Fallback: no category history → global popularity
        if not cat_weights or all(c == "unknown" for c in cat_weights):
            if self.fallback_to_popularity and self._popularity_fallback is not None:
                pop = self._popularity_fallback.recommend(customer_id=customer_id, k=k,
                                                          exclude_purchased=exc)
                if len(pop) > 0:
                    pop = pop.rename(columns={"popularity_score": "category_score"})
                    pop["category"] = "unknown"
                    return pop[["product_id", "category_score", "category", "rank"]]
            return pd.DataFrame(
                columns=["product_id", "category_score", "category", "rank"]
            )

        # Score all catalog products: category_weight × global_popularity
        products = self._product_popularity.reset_index()
        products.columns = ["product_id", "global_pop"]
        products["category"] = products["product_id"].map(
            self._product_category
        ).fillna("unknown")
        products["cat_weight"] = products["category"].map(cat_weights).fillna(0.0)
        products["category_score"] = products["cat_weight"] * products["global_pop"]

        # Only include products from relevant categories (weight > 0)
        relevant = products[products["cat_weight"] > 0].copy()
        if len(relevant) == 0:
            # Fallback
            if self.fallback_to_popularity and self._popularity_fallback is not None:
                pop = self._popularity_fallback.recommend(customer_id=customer_id, k=k,
                                                          exclude_purchased=exc)
                if len(pop) > 0:
                    pop = pop.rename(columns={"popularity_score": "category_score"})
                    pop["category"] = "unknown"
                    return pop[["product_id", "category_score", "category", "rank"]]
            return pd.DataFrame(
                columns=["product_id", "category_score", "category", "rank"]
            )

        # Exclude already-purchased products
        if exc and customer_id is not None and customer_id in self._customer_history:
            purchased = self._customer_history[customer_id]
            relevant = relevant[~relevant["product_id"].isin(purchased)]

        if len(relevant) == 0:
            return pd.DataFrame(
                columns=["product_id", "category_score", "category", "rank"]
            )

        result = _deterministic_topk(relevant, "category_score", "product_id", k)
        result["rank"] = range(1, len(result) + 1)
        return result[["product_id", "category_score", "category", "rank"]]

    def recommend_batch(
        self,
        customer_ids: List[Optional[str]],
        k: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Batch recommendations for a list of customers."""
        return {cid: self.recommend(customer_id=cid, k=k) for cid in customer_ids}

    @property
    def evidence(self) -> List[StructuredEvidence]:
        return list(self._evidence)


# ---------------------------------------------------------------------------
# 3. Item-Item Co-Purchase Recommender
# ---------------------------------------------------------------------------

class CoPurchaseRecommender:
    """Recommends products frequently purchased together with a customer's history.

    Co-purchase graph construction (training data only):
        For each customer order basket (set of distinct products per customer):
            For each product pair (A, B) in the basket:
                co_purchase_count[A][B] += 1
                co_purchase_count[B][A] += 1  (symmetric)

    Similarity metric — Jaccard coefficient:
        jaccard(A, B) = co_purchase_count(A, B)
                        / (purchase_count(A) + purchase_count(B) - co_purchase_count(A, B))

    Jaccard is chosen over raw co-count to avoid bias toward universally popular items.

    Recommendation for customer c:
        1. Collect c's purchased products (seed set S).
        2. For each candidate product p not in S:
               score(p) = MAX over s in S of jaccard(s, p)
           (max-aggregation: recommend based on the best matching seed item).
        3. Return top-K by score, tie-break by product_id.

    LIMITATIONS:
        - Requires basket size >= 2. Single-product baskets contribute no co-purchase signal.
        - On Olist (~96.9% one-time buyers with typically 1 product per order),
          the co-purchase graph will be very sparse; this is expected and reported.
        - New customers receive global popularity fallback.

    Args:
        top_k: Default top-K to return.
        exclude_purchased: Exclude already-purchased products.
        min_co_count: Minimum co-purchase count for a pair to be indexed.
        fallback_to_popularity: Fall back to global popularity for cold-start customers.
    """

    def __init__(
        self,
        top_k: int = 10,
        exclude_purchased: bool = True,
        min_co_count: int = 1,
        fallback_to_popularity: bool = True,
    ):
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if min_co_count < 1:
            raise ValueError(f"min_co_count must be >= 1, got {min_co_count}")

        self.top_k = top_k
        self.exclude_purchased = exclude_purchased
        self.min_co_count = min_co_count
        self.fallback_to_popularity = fallback_to_popularity

        # Fitted state
        self._co_counts: Optional[pd.DataFrame] = None     # [product_a, product_b, co_count]
        self._product_counts: Optional[pd.Series] = None   # product_id → total purchases
        self._jaccard_df: Optional[pd.DataFrame] = None    # [product_a, product_b, jaccard]
        self._customer_history: Optional[Dict[str, Set[str]]] = None
        self._popularity_fallback: Optional[PopularityRecommender] = None
        self._n_co_pairs: int = 0
        self._n_catalog: int = 0
        self._evidence: List[StructuredEvidence] = []
        self._fitted = False

    def fit(self, interaction_train_df: pd.DataFrame) -> "CoPurchaseRecommender":
        """Builds co-purchase graph and Jaccard similarity index from training data.

        The basket for each customer is the set of distinct products they purchased
        (across all training orders). This is consistent with customer_unique_id
        aggregation from Phase 3 M1.

        Args:
            interaction_train_df: Training interaction matrix
                [customer_unique_id, product_id, purchase_count, ...].

        Returns:
            self.
        """
        self._fitted = False
        self._evidence = []

        required = ["customer_unique_id", "product_id", "purchase_count"]
        missing = [c for c in required if c not in interaction_train_df.columns]
        if missing:
            raise ValueError(f"interaction_train_df missing columns: {missing}")

        df = interaction_train_df.copy()
        df["purchase_count"] = pd.to_numeric(df["purchase_count"], errors="coerce").fillna(0)

        # Customer history lookup
        self._customer_history = (
            df.groupby("customer_unique_id")["product_id"]
            .apply(set)
            .to_dict()
        )

        # Product global purchase counts
        self._product_counts = df.groupby("product_id")["purchase_count"].sum()
        self._n_catalog = int(self._product_counts.index.nunique())

        # Build co-purchase graph from customer baskets
        # Each customer's basket = all distinct products they bought in training
        co_count_map: Dict[Tuple[str, str], int] = {}
        n_baskets_with_pairs = 0

        for cid, products in self._customer_history.items():
            product_list = sorted(products)  # deterministic ordering
            if len(product_list) < 2:
                continue
            n_baskets_with_pairs += 1
            for i, p_a in enumerate(product_list):
                for p_b in product_list[i + 1:]:
                    key = (p_a, p_b)
                    co_count_map[key] = co_count_map.get(key, 0) + 1

        # Build symmetric co-count DataFrame
        if co_count_map:
            rows = [
                {"product_a": a, "product_b": b, "co_count": c}
                for (a, b), c in co_count_map.items()
                if c >= self.min_co_count
            ]
            if rows:
                half = pd.DataFrame(rows)
                # Make symmetric
                mirror = half.rename(
                    columns={"product_a": "product_b", "product_b": "product_a"}
                )
                self._co_counts = pd.concat([half, mirror], ignore_index=True)
            else:
                self._co_counts = pd.DataFrame(
                    columns=["product_a", "product_b", "co_count"]
                )
        else:
            self._co_counts = pd.DataFrame(
                columns=["product_a", "product_b", "co_count"]
            )

        self._n_co_pairs = len(co_count_map)  # unique unordered pairs

        # Compute Jaccard similarity
        if len(self._co_counts) > 0:
            jdf = self._co_counts.copy()
            jdf["count_a"] = jdf["product_a"].map(self._product_counts).fillna(0)
            jdf["count_b"] = jdf["product_b"].map(self._product_counts).fillna(0)
            denom = jdf["count_a"] + jdf["count_b"] - jdf["co_count"]
            denom = denom.replace(0, np.nan)
            jdf["jaccard"] = jdf["co_count"] / denom
            jdf["jaccard"] = jdf["jaccard"].fillna(0.0).clip(0.0, 1.0)
            self._jaccard_df = jdf[["product_a", "product_b", "jaccard"]].copy()
        else:
            self._jaccard_df = pd.DataFrame(
                columns=["product_a", "product_b", "jaccard"]
            )

        # Popularity fallback
        self._popularity_fallback = PopularityRecommender(
            top_k=self.top_k, exclude_purchased=self.exclude_purchased
        )
        self._popularity_fallback.fit(interaction_train_df)

        self._fitted = True

        # Provenance
        self._evidence.append(_make_evidence(
            claim_id="copurchase_n_baskets_with_pairs",
            table_name="copurchase_graph",
            column_name="customer_unique_id",
            key="n_baskets_with_pairs",
            value=n_baskets_with_pairs,
            unit="count",
            provenance="COUNT(customer_unique_id WHERE basket_size >= 2) from training",
            linkage=(
                f"{n_baskets_with_pairs} customers had >= 2 distinct products "
                "and contributed to the co-purchase graph."
            ),
        ))
        self._evidence.append(_make_evidence(
            claim_id="copurchase_n_unique_pairs",
            table_name="copurchase_graph",
            column_name="co_count",
            key="n_unique_product_pairs",
            value=self._n_co_pairs,
            unit="count",
            provenance="COUNT(DISTINCT (product_a, product_b)) unordered pairs",
            linkage=(
                f"Co-purchase graph contains {self._n_co_pairs} unique product pairs "
                f"(min_co_count={self.min_co_count})."
            ),
        ))
        self._evidence.append(_make_evidence(
            claim_id="copurchase_jaccard_similarity",
            table_name="copurchase_graph",
            column_name="jaccard",
            key="similarity_metric",
            value="jaccard",
            unit="ratio",
            provenance=(
                "jaccard(A,B) = co_count(A,B) / "
                "(purchase_count(A) + purchase_count(B) - co_count(A,B))"
            ),
            linkage=(
                "Jaccard similarity used to normalize co-purchase counts, "
                "reducing bias toward universally popular products."
            ),
            provenance_type=ProvenanceType.SIMULATION_ASSUMPTION,
        ))

        logger.info(
            f"CoPurchaseRecommender fitted: {self._n_catalog} products, "
            f"{self._n_co_pairs} unique co-purchase pairs, "
            f"{n_baskets_with_pairs} multi-product customer baskets."
        )
        return self

    def recommend(
        self,
        customer_id: Optional[str] = None,
        k: Optional[int] = None,
        exclude_purchased: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Returns co-purchase-based top-K products for a customer.

        Args:
            customer_id: customer_unique_id. New/cold-start customers get fallback.
            k: Number of recommendations.
            exclude_purchased: Override for this call.

        Returns:
            DataFrame[product_id, copurchase_score, rank].
            copurchase_score = max Jaccard similarity to any seed product.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before recommend().")

        k = k if k is not None else self.top_k
        exc = exclude_purchased if exclude_purchased is not None else self.exclude_purchased

        seed_products = (
            self._customer_history.get(customer_id, set())
            if customer_id is not None
            else set()
        )

        # No seed products → fallback (new customer / unknown customer)
        if not seed_products:
            return self._fallback_recommendations(customer_id, k, exc, "no_seed")

        # No co-purchase graph → fallback
        if self._jaccard_df is None or len(self._jaccard_df) == 0:
            return self._fallback_recommendations(customer_id, k, exc, "empty_graph")

        # Find all candidate products with co-purchase signal from seed set
        seed_graph = self._jaccard_df[
            self._jaccard_df["product_a"].isin(seed_products)
        ].copy()

        if len(seed_graph) == 0:
            return self._fallback_recommendations(customer_id, k, exc, "no_graph_signal")

        # For each candidate product_b, take max Jaccard over all seed products
        candidate_scores = (
            seed_graph.groupby("product_b")["jaccard"]
            .max()
            .reset_index()
            .rename(columns={"product_b": "product_id", "jaccard": "copurchase_score"})
        )

        # Exclude already-purchased
        if exc and customer_id is not None:
            candidate_scores = candidate_scores[
                ~candidate_scores["product_id"].isin(seed_products)
            ]

        if len(candidate_scores) == 0:
            return self._fallback_recommendations(customer_id, k, exc, "all_excluded")

        result = _deterministic_topk(candidate_scores, "copurchase_score", "product_id", k)
        result["rank"] = range(1, len(result) + 1)
        return result[["product_id", "copurchase_score", "rank"]]

    def _fallback_recommendations(
        self,
        customer_id: Optional[str],
        k: int,
        exc: bool,
        reason: str,
    ) -> pd.DataFrame:
        """Returns popularity-based fallback with copurchase_score = 0."""
        if self.fallback_to_popularity and self._popularity_fallback is not None:
            pop = self._popularity_fallback.recommend(
                customer_id=customer_id, k=k, exclude_purchased=exc
            )
            if len(pop) > 0:
                logger.debug(
                    f"CoPurchaseRecommender fallback ({reason}) for customer={customer_id}."
                )
                pop = pop.rename(columns={"popularity_score": "copurchase_score"})
                pop["copurchase_score"] = _SCORE_UNAVAILABLE
                return pop[["product_id", "copurchase_score", "rank"]]
        return pd.DataFrame(columns=["product_id", "copurchase_score", "rank"])

    def recommend_batch(
        self,
        customer_ids: List[Optional[str]],
        k: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Batch recommendations for a list of customers."""
        return {cid: self.recommend(customer_id=cid, k=k) for cid in customer_ids}

    @property
    def evidence(self) -> List[StructuredEvidence]:
        return list(self._evidence)

    @property
    def n_co_pairs(self) -> int:
        return self._n_co_pairs
