"""Comprehensive Unit Tests for Scaled Log-Transformed K-Means Customer Segmentation."""

import numpy as np
import pandas as pd
import pytest

from insightx.models.segmentation import CustomerSegmentation


@pytest.fixture
def synthetic_customer_clusters():
    """Generates synthetic customer behavioral data with 3 distinct clusters."""
    np.random.seed(42)
    n_per_cluster = 60

    # Cluster 1: VIP High-Spenders (low recency, high frequency, very high monetary)
    c1 = pd.DataFrame({
        "recency": np.random.uniform(5, 30, n_per_cluster),
        "frequency": np.random.choice([3, 4, 5], n_per_cluster),
        "monetary": np.random.uniform(800, 3000, n_per_cluster),
        "avg_installments": np.random.uniform(4, 10, n_per_cluster),
        "total_freight": np.random.uniform(50, 150, n_per_cluster),
    })

    # Cluster 2: Budget Regulars (low recency, single/repeat order, low monetary)
    c2 = pd.DataFrame({
        "recency": np.random.uniform(10, 45, n_per_cluster),
        "frequency": np.random.choice([1, 2], n_per_cluster),
        "monetary": np.random.uniform(30, 150, n_per_cluster),
        "avg_installments": np.random.uniform(1, 2, n_per_cluster),
        "total_freight": np.random.uniform(10, 25, n_per_cluster),
    })

    # Cluster 3: Churned / Lost (high recency, single order, medium monetary)
    c3 = pd.DataFrame({
        "recency": np.random.uniform(200, 400, n_per_cluster),
        "frequency": np.ones(n_per_cluster, dtype=int),
        "monetary": np.random.uniform(50, 250, n_per_cluster),
        "avg_installments": np.random.uniform(1, 3, n_per_cluster),
        "total_freight": np.random.uniform(15, 35, n_per_cluster),
    })

    return pd.concat([c1, c2, c3], ignore_index=True)


def test_log_transform_and_scaling(synthetic_customer_clusters):
    """Verifies that skewed monetary/recency features are log1p transformed and scaled."""
    seg = CustomerSegmentation(scaler_type="robust")
    scaled_matrix = seg.transform_features(synthetic_customer_clusters)

    assert scaled_matrix.shape == synthetic_customer_clusters.shape
    assert not np.isnan(scaled_matrix).any()
    assert not np.isinf(scaled_matrix).any()
    # Scaled values should have reasonable distribution bounds (no 10000x outliers)
    assert np.all(scaled_matrix < 20.0)


def test_evaluate_k_range(synthetic_customer_clusters):
    """Verifies that inertia decreases monotonically and silhouette scores are computed."""
    seg = CustomerSegmentation(random_state=42)
    results = seg.evaluate_k_range(synthetic_customer_clusters, k_range=(2, 5))

    assert "k_values" in results
    assert results["k_values"] == [2, 3, 4, 5]

    inertias = results["inertia"]
    # Inertia must decrease monotonically as k increases
    for i in range(len(inertias) - 1):
        assert inertias[i] >= inertias[i + 1]

    silhouettes = results["silhouette"]
    assert len(silhouettes) == 4
    for sil in silhouettes:
        assert -1.0 <= sil <= 1.0

    assert results["optimal_k"] in [2, 3, 4, 5]


def test_cluster_stability_ari(synthetic_customer_clusters):
    """Verifies cluster stability across 5 random seeds achieves mean ARI >= 0.75."""
    seg = CustomerSegmentation(n_clusters=3)
    stability = seg.evaluate_stability(
        synthetic_customer_clusters,
        n_clusters=3,
        seeds=[42, 43, 44, 45, 46],
    )

    assert "mean_ari" in stability
    assert "is_stable" in stability
    assert stability["mean_ari"] >= 0.75, (
        f"Cluster stability ARI ({stability['mean_ari']}) below 0.75 target."
    )
    assert stability["is_stable"] is True
    assert len(stability["pairwise_ari"]) == 10  # 5 choose 2 = 10 pairwise comparisons


def test_fit_predict_reproducibility(synthetic_customer_clusters):
    """Verifies that fit_predict produces deterministic cluster labels for the same random_state."""
    seg1 = CustomerSegmentation(n_clusters=3, random_state=42)
    labels1 = seg1.fit_predict(synthetic_customer_clusters)

    seg2 = CustomerSegmentation(n_clusters=3, random_state=42)
    labels2 = seg2.fit_predict(synthetic_customer_clusters)

    pd.testing.assert_series_equal(labels1, labels2)
    assert set(labels1.unique()) == {0, 1, 2}


def test_cluster_profiling(synthetic_customer_clusters):
    """Verifies cluster profiling returns centroids, counts, shares, and persona names."""
    seg = CustomerSegmentation(n_clusters=3, random_state=42)
    labels = seg.fit_predict(synthetic_customer_clusters)

    profile = seg.profile_clusters(synthetic_customer_clusters, labels)
    assert len(profile) == 3
    assert "customer_count" in profile.columns
    assert "customer_share_pct" in profile.columns
    assert "persona_name" in profile.columns
    assert profile["customer_count"].sum() == len(synthetic_customer_clusters)
