"""Recommendation Evaluator (Phase 3, Milestones 2 & 3).

Computes offline ranking metrics for recommendation strategies using
chronologically held-out validation or test data.

Metrics implemented:
  - Precision@K  = |recommended ∩ relevant| / K
  - Recall@K     = |recommended ∩ relevant| / |relevant|
  - F1@K         = 2 * P@K * R@K / (P@K + R@K)
  - NDCG@K       = DCG@K / IDCG@K  (binary relevance, log2 discounting)
  - Catalog Coverage@K = |distinct recommended products| / |total catalog products|

Metric definitions:
  - "Relevant" for a customer in the eval split = any product they actually purchased
    in that split (using customer_unique_id, not customer_id).
  - Recommendations are generated from training data only (no leakage).
  - Customers with no relevant items in the eval split are excluded from P@K and R@K
    (they cannot contribute a signal), but are counted in coverage.

IMPORTANT:
  - These are correlation metrics (observed co-occurrence), NOT causal effects.
  - Olist's extreme one-time-buyer sparsity means most customers will have 0 relevant
    items in the validation split → effective evaluation cohort will be small.
    This is reported explicitly.
  - Do not fabricate or extrapolate metric values beyond the evaluated cohort.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from insightx.utils.logger import get_logger

logger = get_logger(__name__)


def _precision_at_k(recommended: List[str], relevant: set, k: int) -> float:
    """Precision@K = hits in top-K / K."""
    if k == 0:
        return 0.0
    top_k = recommended[:k]
    hits = len(set(top_k) & relevant)
    return hits / k


def _recall_at_k(recommended: List[str], relevant: set, k: int) -> float:
    """Recall@K = hits in top-K / |relevant|."""
    if len(relevant) == 0:
        return 0.0
    top_k = recommended[:k]
    hits = len(set(top_k) & relevant)
    return hits / len(relevant)


def _ndcg_at_k(recommended: List[str], relevant: set, k: int) -> float:
    """NDCG@K with binary relevance and log2 positional discount.

    NDCG@K = DCG@K / IDCG@K
    DCG@K  = sum_{i=1}^{K} rel_i / log2(i+1)   (rel_i in {0,1})
    IDCG@K = sum_{i=1}^{min(K,|relevant|)} 1 / log2(i+1)
    """
    if k == 0 or len(relevant) == 0:
        return 0.0
    top_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(top_k)
        if item in relevant
    )
    n_ideal = min(k, len(relevant))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _f1_at_k(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class RecommendationEvaluator:
    """Evaluates recommendation strategies on held-out validation / test interactions.

    Usage:
        evaluator = RecommendationEvaluator(k=10)
        results = evaluator.evaluate(
            recommender=popularity_rec,
            eval_interaction_df=val_df,        # held-out split
            train_interaction_df=train_df,     # used only to identify training customers
            recommender_name="popularity",
        )

    Args:
        k: Top-K for all metrics.
    """

    def __init__(self, k: int = 10):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k

    def evaluate(
        self,
        recommender: Any,
        eval_interaction_df: pd.DataFrame,
        train_interaction_df: pd.DataFrame,
        recommender_name: str = "baseline",
        product_metadata_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Evaluates a fitted recommender on held-out interaction data.

        Args:
            recommender: A fitted recommender with a recommend(customer_id, k) method.
            eval_interaction_df: Held-out interactions (validation or test split).
                Must have [customer_unique_id, product_id] columns.
            train_interaction_df: Training interactions. Used to get total catalog size
                for coverage computation.
            recommender_name: Label for results dict.
            product_metadata_df: Optional. If provided, catalog size is taken from
                product metadata rather than training interactions.

        Returns:
            Dict with:
                - recommender_name: str
                - k: int
                - n_eval_customers_total: total unique customers in eval split
                - n_eval_customers_with_relevant: customers who have >= 1 relevant item
                - precision_at_k: mean Precision@K (over customers with relevant items)
                - recall_at_k: mean Recall@K
                - f1_at_k: mean F1@K
                - catalog_coverage_at_k: fraction of catalog recommended to >= 1 customer
                - n_catalog_products: total catalog size
                - limitation: str warning about sparsity and cohort size
        """
        for name, df, cols in [
            ("eval_interaction_df", eval_interaction_df,
             ["customer_unique_id", "product_id"]),
            ("train_interaction_df", train_interaction_df,
             ["customer_unique_id", "product_id"]),
        ]:
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(f"{name} missing columns: {missing}")

        # Build relevant items per customer from eval split
        relevant_items: Dict[str, set] = (
            eval_interaction_df
            .groupby("customer_unique_id")["product_id"]
            .apply(set)
            .to_dict()
        )

        n_eval_total = len(relevant_items)

        if n_eval_total == 0:
            logger.warning(
                f"[{recommender_name}] No customers in eval split. "
                "Returning zero metrics."
            )
            return self._zero_result(
                recommender_name, n_eval_total,
                train_interaction_df, product_metadata_df,
            )

        # Catalog size
        if product_metadata_df is not None and "product_id" in product_metadata_df.columns:
            catalog_products = set(product_metadata_df["product_id"].unique())
        else:
            catalog_products = set(train_interaction_df["product_id"].unique())
        n_catalog = len(catalog_products)

        # Evaluate per customer
        precision_vals: List[float] = []
        recall_vals: List[float] = []
        ndcg_vals: List[float] = []
        all_recommended_products: set = set()
        n_with_relevant = 0

        train_cids = (
            set(train_interaction_df["customer_unique_id"].unique())
            if "customer_unique_id" in train_interaction_df.columns
            else set()
        )
        cold_rec_cache: Optional[List[str]] = None
        has_cold_cache = False

        for cid, relevant in relevant_items.items():
            if len(relevant) == 0:
                continue

            # Check if customer is cold-start (unseen in training)
            if cid not in train_cids:
                if not has_cold_cache:
                    try:
                        rec_df = recommender.recommend(customer_id=cid, k=self.k)
                        cold_rec_cache = rec_df["product_id"].tolist() if len(rec_df) > 0 else []
                    except Exception as e:
                        logger.warning(
                            f"[{recommender_name}] recommend() failed for cold customer {cid}: {e}"
                        )
                        cold_rec_cache = []
                    has_cold_cache = True
                recommended_products = cold_rec_cache
            else:
                # Warm customer (seen in training data)
                try:
                    rec_df = recommender.recommend(customer_id=cid, k=self.k)
                    recommended_products = rec_df["product_id"].tolist() if len(rec_df) > 0 else []
                except Exception as e:
                    logger.warning(
                        f"[{recommender_name}] recommend() failed for customer {cid}: {e}"
                    )
                    recommended_products = []

            all_recommended_products.update(recommended_products)

            p = _precision_at_k(recommended_products, relevant, self.k)
            r = _recall_at_k(recommended_products, relevant, self.k)
            n = _ndcg_at_k(recommended_products, relevant, self.k)

            # Only count customers who have relevant items toward metric averages
            n_with_relevant += 1
            precision_vals.append(p)
            recall_vals.append(r)
            ndcg_vals.append(n)

        if n_with_relevant == 0:
            logger.warning(
                f"[{recommender_name}] No eval customers had relevant items in the eval split. "
                "Metrics are 0.0. This is expected with Olist's extreme purchase sparsity."
            )
            mean_precision = 0.0
            mean_recall = 0.0
            mean_ndcg = 0.0
        else:
            mean_precision = float(np.mean(precision_vals))
            mean_recall = float(np.mean(recall_vals))
            mean_ndcg = float(np.mean(ndcg_vals))

        mean_f1 = _f1_at_k(mean_precision, mean_recall)
        coverage = (
            len(all_recommended_products & catalog_products) / n_catalog
            if n_catalog > 0
            else 0.0
        )

        limitation = (
            f"Evaluation cohort: {n_with_relevant}/{n_eval_total} customers had >= 1 relevant item "
            f"in the eval split. Olist's ~96.9% one-time-buyer rate severely limits "
            f"the effective cohort size for P@K and R@K. "
            f"Coverage is computed over {n_catalog} training-period catalog products."
        )

        logger.info(
            f"[{recommender_name}] @K={self.k}: "
            f"P@K={mean_precision:.4f}, R@K={mean_recall:.4f}, "
            f"NDCG@K={mean_ndcg:.4f}, Coverage={coverage:.4f}, "
            f"Cohort={n_with_relevant}/{n_eval_total}"
        )

        return {
            "recommender_name": recommender_name,
            "k": self.k,
            "n_eval_customers_total": n_eval_total,
            "n_eval_customers_with_relevant": n_with_relevant,
            "precision_at_k": round(mean_precision, 6),
            "recall_at_k": round(mean_recall, 6),
            "f1_at_k": round(mean_f1, 6),
            "ndcg_at_k": round(mean_ndcg, 6),
            "catalog_coverage_at_k": round(coverage, 6),
            "n_catalog_products": n_catalog,
            "limitation": limitation,
        }

    def compare(
        self,
        results: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Formats a list of evaluate() result dicts into a comparison DataFrame.

        Args:
            results: List of dicts from evaluate().

        Returns:
            DataFrame with one row per recommender and metric columns.
        """
        rows = []
        for r in results:
            rows.append({
                "recommender": r.get("recommender_name", "?"),
                "K": r.get("k", "?"),
                "Precision@K": r.get("precision_at_k", 0.0),
                "Recall@K": r.get("recall_at_k", 0.0),
                "F1@K": r.get("f1_at_k", 0.0),
                "NDCG@K": r.get("ndcg_at_k", 0.0),
                "Coverage@K": r.get("catalog_coverage_at_k", 0.0),
                "Cohort (w/ relevant)": r.get("n_eval_customers_with_relevant", 0),
                "Total eval customers": r.get("n_eval_customers_total", 0),
            })
        return pd.DataFrame(rows)

    def _zero_result(
        self,
        recommender_name: str,
        n_eval_total: int,
        train_df: pd.DataFrame,
        meta_df: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        n_catalog = (
            int(meta_df["product_id"].nunique())
            if meta_df is not None and "product_id" in meta_df.columns
            else int(train_df["product_id"].nunique())
        )
        return {
            "recommender_name": recommender_name,
            "k": self.k,
            "n_eval_customers_total": n_eval_total,
            "n_eval_customers_with_relevant": 0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "f1_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "catalog_coverage_at_k": 0.0,
            "n_catalog_products": n_catalog,
            "limitation": (
                "No customers in eval split. Cannot compute ranking metrics. "
                "This is expected when validation data is empty."
            ),
        }
