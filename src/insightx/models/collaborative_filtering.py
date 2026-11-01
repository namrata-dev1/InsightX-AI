"""Phase 3, Milestone 3: Collaborative Filtering via Truncated SVD.

Uses sklearn's TruncatedSVD on a log1p-transformed sparse user-item interaction
matrix. This is the lightweight local implementation.

Production recommendation (Google Colab):
  Use the `implicit` library's ALS (Alternating Least Squares) for better
  quality on highly sparse data (96.9% Olist one-time buyers). ALS is optimised
  for implicit feedback (purchase counts) and handles sparsity far better than
  SVD reconstruction. See notebooks/04_advanced_recsys_colab.ipynb.

Local SVD design:
  - n_components controlled via constructor (default 10; capped at min(n_users, n_items)-1).
  - Interaction values: log1p(purchase_count) to compress outliers.
  - Scores: reconstructed user row via U @ Sigma @ Vt, clipped to ≥0.
  - Cold-start (user not in training): returns empty DataFrame; caller should
    use popularity or category fallback.
  - New products (not in training): cannot be scored; excluded from output.

LIMITATIONS:
  - SVD does not model implicit feedback optimally; it minimises MSE on observed
    entries rather than treating zeroes as weak negative signals.
  - On Olist, most users have 1 purchase → their SVD embeddings are dominated
    by the single product's factor. Recommendations will correlate strongly with
    popularity for sparse users.
  - For production quality on Olist, use ALS (Colab notebook).
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from insightx.provenance.evidence import StructuredEvidence
from insightx.provenance.types import ProvenanceType
from insightx.utils.logger import get_logger

logger = get_logger(__name__)


def _make_evidence(
    claim_id: str, table_name: str, column_name: str, key: str,
    value: Any, unit: str, provenance: str, linkage: str,
    provenance_type: ProvenanceType = ProvenanceType.OBSERVED,
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


class SVDCollaborativeFilter:
    """Lightweight SVD-based collaborative filter for local use.

    Fits TruncatedSVD on the log1p(purchase_count) user-item matrix and
    reconstructs user preference scores for all catalog items.

    Args:
        n_components: Number of latent factors. Default 10. Automatically
            capped at min(n_users, n_items) - 1 to satisfy TruncatedSVD.
        top_k: Default top-K to return per customer.
        exclude_purchased: Exclude training-period purchased products.
        random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 10,
        top_k: int = 10,
        exclude_purchased: bool = True,
        random_state: int = 42,
    ):
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self.n_components = n_components
        self.top_k = top_k
        self.exclude_purchased = exclude_purchased
        self.random_state = random_state

        # Fitted state
        self._user_index: Dict[str, int] = {}     # uid → row idx
        self._item_index: Dict[str, int] = {}     # product_id → col idx
        self._item_ids: List[str] = []            # col idx → product_id
        self._user_factors: Optional[np.ndarray] = None   # shape (n_users, k)
        self._item_factors: Optional[np.ndarray] = None   # shape (n_items, k)
        self._customer_history: Optional[Dict[str, Set[str]]] = None
        self._n_components_actual: int = 0
        self._evidence: List[StructuredEvidence] = []
        self._fitted = False

    def fit(self, interaction_train_df: pd.DataFrame) -> "SVDCollaborativeFilter":
        """Builds the SVD decomposition from training interactions.

        Args:
            interaction_train_df: Training interaction matrix with columns
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
        df["purchase_count"] = pd.to_numeric(df["purchase_count"], errors="coerce").fillna(0)

        # Deduplicate: aggregate to (uid, product) grain
        df = (
            df.groupby(["customer_unique_id", "product_id"], as_index=False)["purchase_count"]
            .sum()
        )

        # Build index maps
        users = sorted(df["customer_unique_id"].unique())
        items = sorted(df["product_id"].unique())
        self._user_index = {u: i for i, u in enumerate(users)}
        self._item_index = {p: j for j, p in enumerate(items)}
        self._item_ids = items
        n_users = len(users)
        n_items = len(items)

        # Customer history for exclusion
        self._customer_history = (
            df.groupby("customer_unique_id")["product_id"].apply(set).to_dict()
        )

        # Cannot fit SVD with fewer than 2 users or 2 items
        if n_users < 2 or n_items < 2:
            logger.warning(
                f"SVDCollaborativeFilter: too few users ({n_users}) or items ({n_items}). "
                "Model will return empty recommendations."
            )
            self._user_factors = np.zeros((n_users, 1))
            self._item_factors = np.zeros((n_items, 1))
            self._n_components_actual = 0
            self._fitted = True
            self._build_evidence(n_users, n_items, 0)
            return self

        # Build sparse user-item matrix: values = log1p(purchase_count)
        rows = df["customer_unique_id"].map(self._user_index).values
        cols = df["product_id"].map(self._item_index).values
        vals = np.log1p(df["purchase_count"].values).astype(np.float32)
        sparse_matrix = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))

        # Cap n_components
        max_components = min(n_users, n_items) - 1
        k = max(1, min(self.n_components, max_components))
        self._n_components_actual = k

        svd = TruncatedSVD(n_components=k, random_state=self.random_state)
        # U_reduced: (n_users, k), Sigma embedded in item_factors
        user_factors_raw = svd.fit_transform(sparse_matrix)  # shape (n_users, k)
        # Recover: V^T is (k, n_items), singular values in svd.singular_values_
        sigma = np.diag(svd.singular_values_)                # (k, k)
        item_factors_raw = svd.components_.T                  # (n_items, k)

        # Scale user factors: U @ Sigma for score = U_scaled @ V^T
        self._user_factors = user_factors_raw                 # (n_users, k)
        self._item_factors = item_factors_raw @ sigma         # (n_items, k), pre-scaled

        self._fitted = True
        self._build_evidence(n_users, n_items, k)

        variance_explained = float(svd.explained_variance_ratio_.sum())
        logger.info(
            f"SVDCollaborativeFilter fitted: {n_users} users, {n_items} items, "
            f"k={k} components, explained variance={variance_explained:.3f}."
        )
        return self

    def recommend(
        self,
        customer_id: Optional[str] = None,
        k: Optional[int] = None,
        exclude_purchased: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Returns top-K SVD-scored products for a customer.

        Args:
            customer_id: customer_unique_id. If not in training data, returns empty.
            k: Number of recommendations.
            exclude_purchased: Override for this call.

        Returns:
            DataFrame[product_id, cf_score, rank].
            cf_score = reconstructed preference score (not a probability).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before recommend().")

        k = k if k is not None else self.top_k
        exc = exclude_purchased if exclude_purchased is not None else self.exclude_purchased

        # Unknown/new customer → cold start (caller handles fallback)
        if customer_id is None or customer_id not in self._user_index:
            return pd.DataFrame(columns=["product_id", "cf_score", "rank"])

        if self._n_components_actual == 0:
            return pd.DataFrame(columns=["product_id", "cf_score", "rank"])

        user_idx = self._user_index[customer_id]
        user_vec = self._user_factors[user_idx]  # shape (k,)

        # Scores: dot product with item factors (pre-scaled by Sigma)
        scores = self._item_factors @ user_vec   # shape (n_items,)
        scores = np.clip(scores, 0.0, None)      # non-negative (CF scores can be negative)

        # Build result DataFrame
        result = pd.DataFrame({
            "product_id": self._item_ids,
            "cf_score": scores,
        })

        # Exclude already-purchased
        if exc and customer_id in self._customer_history:
            purchased = self._customer_history[customer_id]
            result = result[~result["product_id"].isin(purchased)]

        if len(result) == 0:
            return pd.DataFrame(columns=["product_id", "cf_score", "rank"])

        result = (
            result.sort_values(["cf_score", "product_id"], ascending=[False, True])
            .head(k)
            .reset_index(drop=True)
        )
        result["rank"] = range(1, len(result) + 1)
        return result[["product_id", "cf_score", "rank"]]

    def recommend_batch(
        self,
        customer_ids: List[Optional[str]],
        k: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        return {cid: self.recommend(customer_id=cid, k=k) for cid in customer_ids}

    def _build_evidence(self, n_users: int, n_items: int, k: int) -> None:
        self._evidence.append(_make_evidence(
            claim_id="svd_cf_n_components",
            table_name="svd_cf_model",
            column_name="n_components",
            key="k_actual",
            value=k,
            unit="count",
            provenance=f"TruncatedSVD(n_components={k}) on log1p(purchase_count) matrix",
            linkage=f"SVD fitted with k={k} latent factors on {n_users}x{n_items} matrix.",
            provenance_type=ProvenanceType.SIMULATION_ASSUMPTION,
        ))
        self._evidence.append(_make_evidence(
            claim_id="svd_cf_matrix_size",
            table_name="svd_cf_model",
            column_name="n_users",
            key="n_training_users",
            value=n_users,
            unit="count",
            provenance="COUNT(DISTINCT customer_unique_id) in training interaction matrix",
            linkage=f"User-item matrix has {n_users} rows (training customers).",
        ))
        self._evidence.append(_make_evidence(
            claim_id="svd_cf_catalog_size",
            table_name="svd_cf_model",
            column_name="n_items",
            key="n_training_items",
            value=n_items,
            unit="count",
            provenance="COUNT(DISTINCT product_id) in training interaction matrix",
            linkage=f"User-item matrix has {n_items} columns (training products).",
        ))

    @property
    def evidence(self) -> List[StructuredEvidence]:
        return list(self._evidence)

    @property
    def n_components_actual(self) -> int:
        return self._n_components_actual
