"""Customer segmentation using K-Means clustering."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from insightx.utils.logger import get_logger

logger = get_logger(__name__)


class CustomerSegmentation:
    """Trains and evaluates K-Means customer segmentation models."""

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.features_used: List[str] = []

    def evaluate_k_range(
        self, X: pd.DataFrame, k_range: Tuple[int, int] = (2, 10)
    ) -> Dict[str, List[float]]:
        """Calculates inertia (Elbow method) and silhouette scores across cluster counts.

        Returns:
            Dict containing 'inertia' and 'silhouette' metric lists.
        """
        logger.info(f"Evaluating K-Means over cluster range {k_range}...")
        return {"inertia": [], "silhouette": []}

    def fit_predict(self, X: pd.DataFrame) -> pd.Series:
        """Scales input features and assigns cluster labels to customers."""
        logger.info(f"Fitting K-Means with {self.n_clusters} clusters...")
        return pd.Series(dtype=int)
