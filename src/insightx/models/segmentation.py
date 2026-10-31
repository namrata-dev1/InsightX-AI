"""Customer Segmentation Engine using Scaled Log-Transformed K-Means with Stability Auditing."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler

from insightx.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SKEWED_FEATURES = ["recency", "monetary", "recency_days", "total_monetary", "total_freight", "avg_installments"]


class CustomerSegmentation:
    """Trains, optimizes, and evaluates stable K-Means customer segmentation models."""

    def __init__(
        self,
        n_clusters: int = 4,
        random_state: int = 42,
        scaler_type: str = "robust",
        skewed_features: Optional[List[str]] = None,
    ):
        """Args:
            n_clusters: Number of clusters (k).
            random_state: Base random seed for reproducibility.
            scaler_type: 'robust' (RobustScaler) or 'standard' (StandardScaler).
            skewed_features: List of feature names to log1p transform.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler_type = scaler_type.lower()
        self.skewed_features = skewed_features or DEFAULT_SKEWED_FEATURES

        self.scaler = RobustScaler() if self.scaler_type == "robust" else StandardScaler()
        self.model: Optional[KMeans] = None
        self.features_used: List[str] = []
        self.is_fitted: bool = False

    def transform_features(self, X: pd.DataFrame, fit_scaler: bool = True) -> np.ndarray:
        """Applies log1p transformation to skewed features followed by robust/standard scaling.

        Args:
            X: Input customer feature DataFrame.
            fit_scaler: If True, fits the scaler on X; otherwise uses pre-fitted scaler.

        Returns:
            np.ndarray: Scaled 2D feature matrix ready for distance-based clustering.
        """
        df = X.copy()
        self.features_used = list(df.columns)

        # 1. Log1p transformation on heavily skewed monetary/recency features
        for col in self.features_used:
            if col in self.skewed_features and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = np.log1p(np.maximum(df[col].fillna(0.0), 0.0))
            else:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(0.0)

        numeric_vals = df[self.features_used].select_dtypes(include=[np.number]).values

        # 2. Scaling
        if fit_scaler:
            scaled_vals = self.scaler.fit_transform(numeric_vals)
        else:
            scaled_vals = self.scaler.transform(numeric_vals)

        return scaled_vals

    def evaluate_k_range(
        self,
        X: pd.DataFrame,
        k_range: Tuple[int, int] = (2, 8),
        sample_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Calculates inertia (Elbow) and silhouette scores across cluster counts.

        Args:
            X: Customer feature DataFrame.
            k_range: (min_k, max_k) inclusive.
            sample_size: Optional subsample size for rapid silhouette scoring.

        Returns:
            Dict containing 'k_values', 'inertia', 'silhouette', and 'optimal_k'.
        """
        scaled_X = self.transform_features(X, fit_scaler=True)
        k_values = list(range(k_range[0], k_range[1] + 1))
        inertia_scores: List[float] = []
        silhouette_scores: List[float] = []

        logger.info(f"Evaluating K-Means across k in {k_values}...")

        for k in k_values:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(scaled_X)
            inertia_scores.append(round(float(km.inertia_), 2))

            # Silhouette computation (subsample if dataset is large)
            if sample_size and len(scaled_X) > sample_size:
                sil = silhouette_score(
                    scaled_X, labels, sample_size=sample_size, random_state=self.random_state
                )
            else:
                sil = silhouette_score(scaled_X, labels)
            silhouette_scores.append(round(float(sil), 4))

        # Select optimal k: highest silhouette score
        best_idx = int(np.argmax(silhouette_scores))
        optimal_k = k_values[best_idx]

        logger.info(
            f"K-range evaluation complete. Optimal k={optimal_k} (Silhouette={silhouette_scores[best_idx]:.4f})."
        )

        return {
            "k_values": k_values,
            "inertia": inertia_scores,
            "silhouette": silhouette_scores,
            "optimal_k": optimal_k,
        }

    def fit_predict(
        self,
        X: pd.DataFrame,
        n_clusters: Optional[int] = None,
    ) -> pd.Series:
        """Scales input features and fits K-Means, returning cluster labels.

        Args:
            X: Customer feature DataFrame.
            n_clusters: Optional override for cluster count.

        Returns:
            pd.Series: Integer cluster labels aligned with input index.
        """
        k = n_clusters or self.n_clusters
        scaled_X = self.transform_features(X, fit_scaler=True)

        logger.info(f"Fitting K-Means with k={k} clusters (random_state={self.random_state})...")
        self.model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        labels = self.model.fit_predict(scaled_X)
        self.is_fitted = True

        return pd.Series(labels, index=X.index, name="cluster")

    def evaluate_stability(
        self,
        X: pd.DataFrame,
        n_clusters: Optional[int] = None,
        seeds: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Evaluates cluster stability using Adjusted Rand Index (ARI) across random seeds.

        Ensures segmentation is not an artifact of random centroid initialization.

        Args:
            X: Customer feature DataFrame.
            n_clusters: Number of clusters (defaults to self.n_clusters).
            seeds: List of random seeds to evaluate (default: [42, 43, 44, 45, 46]).

        Returns:
            Dict containing 'mean_ari', 'pairwise_ari', 'is_stable' (True if mean_ari >= 0.75).
        """
        k = n_clusters or self.n_clusters
        eval_seeds = seeds or [42, 43, 44, 45, 46]
        scaled_X = self.transform_features(X, fit_scaler=True)

        cluster_runs: List[np.ndarray] = []
        for seed in eval_seeds:
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)
            cluster_runs.append(km.fit_predict(scaled_X))

        # Compute all pairwise ARI scores
        pairwise_ari: List[float] = []
        n_runs = len(cluster_runs)
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                ari = adjusted_rand_score(cluster_runs[i], cluster_runs[j])
                pairwise_ari.append(round(float(ari), 4))

        mean_ari = round(float(np.mean(pairwise_ari)), 4) if pairwise_ari else 1.0
        is_stable = bool(mean_ari >= 0.75)

        logger.info(
            f"Cluster stability evaluation across {len(eval_seeds)} seeds: "
            f"Mean ARI = {mean_ari:.4f} (Stable={is_stable}, Target >= 0.75)."
        )

        return {
            "mean_ari": mean_ari,
            "min_ari": min(pairwise_ari) if pairwise_ari else 1.0,
            "max_ari": max(pairwise_ari) if pairwise_ari else 1.0,
            "pairwise_ari": pairwise_ari,
            "is_stable": is_stable,
            "seeds_tested": eval_seeds,
            "n_clusters": k,
        }

    def profile_clusters(
        self,
        X_original: pd.DataFrame,
        cluster_labels: pd.Series,
    ) -> pd.DataFrame:
        """Profiles raw (non-transformed) business metrics per cluster.

        Args:
            X_original: Original, un-scaled feature DataFrame.
            cluster_labels: Series of cluster assignments.

        Returns:
            pd.DataFrame: Mean/median profiles per cluster with assigned persona labels.
        """
        df = X_original.copy()
        df["cluster"] = cluster_labels.values

        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "cluster"]

        profile = df.groupby("cluster")[numeric_cols].agg(["mean", "median", "count"])

        # Create simplified centroid profile table for executive reporting
        centroids = df.groupby("cluster")[numeric_cols].mean().round(2)
        centroids["customer_count"] = df.groupby("cluster").size()
        centroids["customer_share_pct"] = np.round(
            (centroids["customer_count"] / len(df)) * 100.0, 1
        )

        # Synthesize persona label based on recency and monetary centroids
        personas = []
        for cluster_id, row in centroids.iterrows():
            rec = row.get("recency", row.get("recency_days", 100))
            mon = row.get("monetary", row.get("total_monetary", 100))
            freq = row.get("frequency", 1)

            if mon > centroids["monetary" if "monetary" in centroids else "total_monetary"].median() * 1.5:
                persona = "VIP High-Spenders"
            elif freq >= 2:
                persona = "Loyal Repeat Buyers"
            elif rec < 60:
                persona = "Recent Active Customers"
            else:
                persona = "Low-Engagement Customers"
            personas.append(persona)

        centroids["persona_name"] = personas
        return centroids
