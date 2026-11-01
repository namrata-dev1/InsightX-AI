"""Phase 3, Milestone 3: Hybrid Recommender.

Combines scores from multiple fitted recommenders using configurable weights.
Handles missing scores (cold-start, unsupported model) via zero-fill.

Score normalisation:
  Each sub-recommender's scores are min-max normalised to [0, 1] independently
  before weighted combination. This prevents any single recommender from dominating
  due to scale differences (e.g., CF scores in [0, 5] vs cosine similarity in [0, 1]).

  normalised_score(x) = (x - min(x)) / (max(x) - min(x) + epsilon)

  Epsilon (1e-9) prevents division by zero when all scores are equal.

Combined score:
  hybrid_score(product) = w1*norm_CF + w2*norm_CBF + w3*norm_Popularity
                          (weights need not sum to 1; they are relative weights)

Cold-start strategy:
  If a sub-recommender returns 0 products for a customer (unknown user, sparse
  history), its normalised score for all products defaults to 0.0. Other
  recommenders still contribute. If ALL recommenders return empty, the hybrid
  falls back to global popularity.

LIMITATIONS:
  - Weights are configurable but not learned from data (no meta-learning).
  - Score normalisation is done per-customer, so scores are not comparable
    across customers.
  - The hybrid cannot outperform its best component on average; it reduces
    variance at the cost of peak performance.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

import numpy as np
import pandas as pd

from insightx.models.recommenders import PopularityRecommender
from insightx.provenance.evidence import StructuredEvidence
from insightx.provenance.types import ProvenanceType
from insightx.utils.logger import get_logger

logger = get_logger(__name__)

_EPS = 1e-9


def _minmax_normalize(scores: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1]; constant series → all 0.0."""
    lo, hi = scores.min(), scores.max()
    if hi - lo < _EPS:
        return pd.Series(np.zeros(len(scores), dtype=float), index=scores.index)
    return (scores - lo) / (hi - lo + _EPS)


def _make_evidence(
    claim_id: str, table_name: str, column_name: str, key: str,
    value: Any, unit: str, provenance: str, linkage: str,
    provenance_type: ProvenanceType = ProvenanceType.SIMULATION_ASSUMPTION,
) -> StructuredEvidence:
    if isinstance(value, (np.integer,)):
        value = int(value)
    elif isinstance(value, (np.floating,)):
        value = float(value)
    return StructuredEvidence(
        evidence_id=str(uuid.uuid4()),
        claim_id=claim_id,
        dataset_version_id="olist_phase3_m3",
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


class HybridRecommender:
    """Weighted ensemble of multiple fitted recommendation models.

    Usage:
        hybrid = HybridRecommender(
            components=[
                (cf_model,   0.4, "cf_score"),
                (cbf_model,  0.3, "cbf_score"),
                (pop_model,  0.3, "popularity_score"),
            ],
            top_k=10,
            exclude_purchased=True,
        )
        hybrid.fit(interaction_train_df)   # builds fallback popularity + catalog
        recs = hybrid.recommend("uid_001", k=10)

    All component recommenders must be pre-fitted before passing to HybridRecommender.

    Args:
        components: List of (recommender, weight, score_col_name) tuples.
            recommender: Any fitted object with recommend(customer_id, k) method.
            weight: Non-negative float weight for this recommender.
            score_col_name: Name of the score column in recommender's output.
        top_k: Default top-K.
        exclude_purchased: Exclude training-period purchases from output.
        fallback_to_popularity: If all components return empty, use global popularity.
    """

    def __init__(
        self,
        components: List[Tuple[Any, float, str]],
        top_k: int = 10,
        exclude_purchased: bool = True,
        fallback_to_popularity: bool = True,
    ):
        if not components:
            raise ValueError("components list must be non-empty.")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        for i, (rec, w, col) in enumerate(components):
            if w < 0:
                raise ValueError(f"Component {i} weight must be >= 0, got {w}.")

        self.components = components
        self.top_k = top_k
        self.exclude_purchased = exclude_purchased
        self.fallback_to_popularity = fallback_to_popularity

        # Fitted state
        self._catalog_products: Optional[List[str]] = None
        self._customer_history: Optional[Dict[str, Set[str]]] = None
        self._fallback_recommender: Optional[PopularityRecommender] = None
        self._evidence: List[StructuredEvidence] = []
        self._fitted = False

    def fit(self, interaction_train_df: pd.DataFrame) -> "HybridRecommender":
        """Initialises catalog list, customer history, and fallback recommender.

        Args:
            interaction_train_df: Training interactions
                [customer_unique_id, product_id, purchase_count].

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
        self._catalog_products = sorted(df["product_id"].unique().tolist())
        self._customer_history = (
            df.groupby("customer_unique_id")["product_id"].apply(set).to_dict()
        )

        # Fallback popularity recommender (pre-fitted)
        self._fallback_recommender = PopularityRecommender(
            top_k=self.top_k, exclude_purchased=self.exclude_purchased
        ).fit(df)

        self._fitted = True

        # Provenance
        n_components = len(self.components)
        weights = [w for _, w, _ in self.components]
        total_w = sum(weights)
        for i, (_, w, col) in enumerate(self.components):
            self._evidence.append(_make_evidence(
                claim_id=f"hybrid_component_{i}_weight",
                table_name="hybrid_recommender",
                column_name="component_weight",
                key=col,
                value=round(float(w) / total_w, 6) if total_w > 0 else 0.0,
                unit="ratio",
                provenance=f"Relative weight for component '{col}': {w}/{total_w}",
                linkage=(
                    f"Hybrid component '{col}' contributes "
                    f"{100*w/total_w:.1f}% of the combined score." if total_w > 0
                    else "All weights are zero."
                ),
            ))

        logger.info(
            f"HybridRecommender fitted: {n_components} components, "
            f"{len(self._catalog_products)} catalog products."
        )
        return self

    def recommend(
        self,
        customer_id: Optional[str] = None,
        k: Optional[int] = None,
        exclude_purchased: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Returns hybrid top-K for a customer.

        Args:
            customer_id: customer_unique_id.
            k: Number of recommendations.
            exclude_purchased: Override for this call.

        Returns:
            DataFrame[product_id, hybrid_score, rank].
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before recommend().")

        k = k if k is not None else self.top_k
        exc = exclude_purchased if exclude_purchased is not None else self.exclude_purchased

        # Collect all products across all component outputs
        # Use the full catalog as the candidate pool
        all_products = pd.DataFrame({"product_id": self._catalog_products})
        hybrid_scores = pd.Series(0.0, index=all_products["product_id"])

        any_signal = False
        total_weight = sum(w for _, w, _ in self.components)
        if total_weight == 0:
            total_weight = 1.0  # avoid div-by-zero

        for recommender, weight, score_col in self.components:
            if weight == 0:
                continue
            try:
                # Get full-catalog scores by requesting all k
                rec_df = recommender.recommend(
                    customer_id=customer_id,
                    k=len(self._catalog_products),
                    exclude_purchased=False,  # exclusion handled at hybrid level
                )
            except Exception as e:
                logger.debug(f"HybridRecommender: component recommend() failed: {e}")
                rec_df = pd.DataFrame(columns=["product_id", score_col])

            if len(rec_df) == 0 or score_col not in rec_df.columns:
                # No signal from this component; contribute 0
                continue

            any_signal = True
            comp_series = rec_df.set_index("product_id")[score_col].reindex(
                self._catalog_products
            ).fillna(0.0)

            normalised = _minmax_normalize(comp_series)
            hybrid_scores = hybrid_scores + (weight / total_weight) * normalised

        # If no component returned any recommendations → fallback
        if not any_signal and self.fallback_to_popularity and self._fallback_recommender:
            logger.debug(
                f"HybridRecommender: all components empty for {customer_id}; "
                "using popularity fallback."
            )
            pop = self._fallback_recommender.recommend(
                customer_id=customer_id, k=k, exclude_purchased=exc
            )
            if len(pop) > 0:
                pop = pop.rename(columns={"popularity_score": "hybrid_score"})
                pop["rank"] = range(1, len(pop) + 1)
                return pop[["product_id", "hybrid_score", "rank"]]
            return pd.DataFrame(columns=["product_id", "hybrid_score", "rank"])

        result = pd.DataFrame({
            "product_id": hybrid_scores.index,
            "hybrid_score": hybrid_scores.values,
        })

        # Exclude already-purchased
        if exc and customer_id is not None and customer_id in self._customer_history:
            purchased = self._customer_history[customer_id]
            result = result[~result["product_id"].isin(purchased)]

        if len(result) == 0:
            return pd.DataFrame(columns=["product_id", "hybrid_score", "rank"])

        result = (
            result.sort_values(["hybrid_score", "product_id"], ascending=[False, True])
            .head(k)
            .reset_index(drop=True)
        )
        result["rank"] = range(1, len(result) + 1)
        return result[["product_id", "hybrid_score", "rank"]]

    def recommend_batch(
        self,
        customer_ids: List[Optional[str]],
        k: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        return {cid: self.recommend(customer_id=cid, k=k) for cid in customer_ids}

    @property
    def evidence(self) -> List[StructuredEvidence]:
        return list(self._evidence)
