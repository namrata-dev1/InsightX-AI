"""Phase 3, Milestone 3: Content-Based Recommender.

Computes product-product similarity from metadata features and builds
customer profiles as weighted averages of purchased product vectors.

Feature encoding (all fitted on training data only):
  - product_category_name_english: OneHotEncoder (training categories only).
  - price_band: OneHotEncoder (training price bands only).
  - Numeric: product_weight_g, product_length_cm, product_height_cm,
             product_width_cm, product_photos_qty — StandardScaler (training stats).

Similarity metric: Cosine similarity on the concatenated feature vector.

Customer profile: purchase-count weighted average of purchased product vectors.

Score for candidate product p given customer c:
    cbf_score(c, p) = cosine_similarity(profile(c), feature_vec(p))

LIMITATIONS:
  - Requires product metadata. Products with all-zero features (missing metadata)
    may cluster artificially.
  - Category is the dominant feature; numeric features add small signal.
  - For large catalogs (> ~10k products), computing full cosine similarity matrix
    is memory-intensive. Local implementation is fine for Olist (~32k products),
    but use approximate nearest neighbours in production.
  - Customers with no product metadata in their history get empty recommendations.
  - Content-based cannot discover cross-category preferences.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import uuid
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


class ContentBasedRecommender:
    """Recommends products similar to a customer's purchase history using metadata.

    Args:
        top_k: Default number of recommendations.
        exclude_purchased: Exclude products the customer already bought.
        category_col: Column name for product category.
        price_band_col: Column name for price band.
        numeric_cols: List of numeric feature column names.
    """

    _DEFAULT_NUMERIC = [
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
        "product_photos_qty",
    ]
    _DEFAULT_CAT_COL = "product_category_name_english"
    _DEFAULT_PRICE_COL = "price_band"

    def __init__(
        self,
        top_k: int = 10,
        exclude_purchased: bool = True,
        category_col: str = _DEFAULT_CAT_COL,
        price_band_col: str = _DEFAULT_PRICE_COL,
        numeric_cols: Optional[List[str]] = None,
    ):
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self.top_k = top_k
        self.exclude_purchased = exclude_purchased
        self.category_col = category_col
        self.price_band_col = price_band_col
        self.numeric_cols = numeric_cols if numeric_cols is not None else self._DEFAULT_NUMERIC

        # Fitted state
        self._product_ids: Optional[List[str]] = None      # ordered list
        self._feature_matrix: Optional[np.ndarray] = None  # (n_products, n_features)
        self._cat_encoder: Optional[OneHotEncoder] = None
        self._price_encoder: Optional[OneHotEncoder] = None
        self._numeric_scaler: Optional[StandardScaler] = None
        self._customer_history: Optional[Dict[str, Set[str]]] = None
        self._customer_profiles: Optional[Dict[str, np.ndarray]] = None
        self._n_products: int = 0
        self._n_features: int = 0
        self._evidence: List[StructuredEvidence] = []
        self._fitted = False

    def fit(
        self,
        product_metadata_df: pd.DataFrame,
        interaction_train_df: pd.DataFrame,
    ) -> "ContentBasedRecommender":
        """Builds product feature matrix and customer profiles from training data.

        All encoders/scalers are fitted on training-period products only.

        Args:
            product_metadata_df: Product metadata from Phase 3 M1
                [product_id, product_category_name_english, price_band,
                 product_weight_g, ...].
            interaction_train_df: Training interaction matrix
                [customer_unique_id, product_id, purchase_count].

        Returns:
            self.
        """
        self._fitted = False
        self._evidence = []

        # Validate inputs
        if "product_id" not in product_metadata_df.columns:
            raise ValueError("product_metadata_df missing 'product_id' column.")
        required_int = ["customer_unique_id", "product_id", "purchase_count"]
        missing = [c for c in required_int if c not in interaction_train_df.columns]
        if missing:
            raise ValueError(f"interaction_train_df missing columns: {missing}")

        meta = product_metadata_df.copy()
        meta["product_id"] = meta["product_id"].astype(str)

        # 1. Encode categorical features (fit on training products only)
        cat_cols_present = [
            c for c in [self.category_col, self.price_band_col]
            if c in meta.columns
        ]
        numeric_cols_present = [c for c in self.numeric_cols if c in meta.columns]

        feature_parts = []

        if cat_cols_present:
            # Fill missing category values
            for col in cat_cols_present:
                meta[col] = meta[col].fillna("unknown").astype(str)

            self._cat_encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
                dtype=np.float32,
            )
            cat_encoded = self._cat_encoder.fit_transform(meta[cat_cols_present])
            feature_parts.append(cat_encoded)

        if numeric_cols_present:
            num_vals = meta[numeric_cols_present].copy()
            for col in numeric_cols_present:
                num_vals[col] = pd.to_numeric(num_vals[col], errors="coerce").fillna(0.0)
            self._numeric_scaler = StandardScaler()
            num_scaled = self._numeric_scaler.fit_transform(num_vals.values.astype(np.float32))
            feature_parts.append(num_scaled)

        if not feature_parts:
            logger.warning(
                "ContentBasedRecommender: no usable feature columns found. "
                "Returning empty recommendations."
            )
            self._feature_matrix = np.zeros((len(meta), 1), dtype=np.float32)
            self._product_ids = meta["product_id"].tolist()
            self._n_products = len(meta)
            self._n_features = 0
            self._fitted = True
            return self

        self._feature_matrix = np.hstack(feature_parts).astype(np.float32)
        self._product_ids = meta["product_id"].tolist()
        self._n_products = len(meta)
        self._n_features = self._feature_matrix.shape[1]

        # 2. Build customer purchase history (purchase_count weighted)
        train_df = interaction_train_df.copy()
        train_df["purchase_count"] = (
            pd.to_numeric(train_df["purchase_count"], errors="coerce").fillna(0)
        )
        self._customer_history = (
            train_df.groupby("customer_unique_id")["product_id"].apply(set).to_dict()
        )

        # 3. Build customer profiles: weighted average of purchased product vectors
        product_idx_map = {pid: i for i, pid in enumerate(self._product_ids)}
        self._customer_profiles = {}
        for uid, group in train_df.groupby("customer_unique_id"):
            vecs = []
            weights = []
            for _, row in group.iterrows():
                pid = str(row["product_id"])
                if pid in product_idx_map:
                    vecs.append(self._feature_matrix[product_idx_map[pid]])
                    weights.append(float(row["purchase_count"]) + 1.0)  # +1 to handle 0-count
            if vecs:
                wt = np.array(weights)
                profile = np.average(np.vstack(vecs), weights=wt, axis=0)
                self._customer_profiles[uid] = profile.astype(np.float32)

        self._fitted = True

        # Provenance
        self._evidence.append(_make_evidence(
            claim_id="cbf_n_products",
            table_name="content_based_features",
            column_name="product_id",
            key="n_products_in_catalog",
            value=self._n_products,
            unit="count",
            provenance="COUNT(product_id) from product_metadata_df",
            linkage=f"CBF feature matrix has {self._n_products} products x {self._n_features} features.",
        ))
        self._evidence.append(_make_evidence(
            claim_id="cbf_feature_dim",
            table_name="content_based_features",
            column_name="feature_vector",
            key="n_feature_dimensions",
            value=self._n_features,
            unit="count",
            provenance=(
                f"OneHotEncoder({cat_cols_present}) + StandardScaler({numeric_cols_present})"
            ),
            linkage=f"Product feature vector dimensionality: {self._n_features}.",
            provenance_type=ProvenanceType.SIMULATION_ASSUMPTION,
        ))
        n_profiles = len(self._customer_profiles)
        self._evidence.append(_make_evidence(
            claim_id="cbf_n_customer_profiles",
            table_name="content_based_features",
            column_name="customer_unique_id",
            key="n_customer_profiles",
            value=n_profiles,
            unit="count",
            provenance="Weighted average of purchased product feature vectors per customer",
            linkage=f"CBF customer profiles built for {n_profiles} training customers.",
        ))

        logger.info(
            f"ContentBasedRecommender fitted: {self._n_products} products, "
            f"{self._n_features} features, {n_profiles} customer profiles."
        )
        return self

    def recommend(
        self,
        customer_id: Optional[str] = None,
        k: Optional[int] = None,
        exclude_purchased: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Returns top-K content-similar products for a customer.

        Args:
            customer_id: customer_unique_id. If not in training, returns empty.
            k: Number of recommendations.
            exclude_purchased: Override for this call.

        Returns:
            DataFrame[product_id, cbf_score, rank].
            cbf_score = cosine similarity ∈ [-1, 1]; higher = more similar.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before recommend().")

        k = k if k is not None else self.top_k
        exc = exclude_purchased if exclude_purchased is not None else self.exclude_purchased

        if self._n_features == 0:
            return pd.DataFrame(columns=["product_id", "cbf_score", "rank"])

        if customer_id is None or customer_id not in self._customer_profiles:
            return pd.DataFrame(columns=["product_id", "cbf_score", "rank"])

        profile = self._customer_profiles[customer_id].reshape(1, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cosine_similarity(profile, self._feature_matrix)[0]  # (n_products,)

        result = pd.DataFrame({
            "product_id": self._product_ids,
            "cbf_score": scores.astype(float),
        })

        # Exclude purchased
        if exc and customer_id in self._customer_history:
            purchased = self._customer_history[customer_id]
            result = result[~result["product_id"].isin(purchased)]

        if len(result) == 0:
            return pd.DataFrame(columns=["product_id", "cbf_score", "rank"])

        result = (
            result.sort_values(["cbf_score", "product_id"], ascending=[False, True])
            .head(k)
            .reset_index(drop=True)
        )
        result["rank"] = range(1, len(result) + 1)
        return result[["product_id", "cbf_score", "rank"]]

    def recommend_batch(
        self,
        customer_ids: List[Optional[str]],
        k: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        return {cid: self.recommend(customer_id=cid, k=k) for cid in customer_ids}

    @property
    def evidence(self) -> List[StructuredEvidence]:
        return list(self._evidence)

    @property
    def n_products(self) -> int:
        return self._n_products
