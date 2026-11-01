"""Phase 3 M4 Execution Profiling, Hybrid Tuning & Segment-Wise Evaluation.

Runs:
1. Model training across 6 recommenders on training data.
2. Weight tuning on chronological validation split.
3. Comparative evaluation of all 6 models including Diversity, ARP, and Gini.
4. Segment-wise breakdown (repeat, sparse, cold-start).
5. Resource profiling report.
"""
import sys
sys.path.insert(0, "c:/Users/khushi/OneDrive/Desktop/InsightX-AI/src")

import numpy as np
import pandas as pd
from datetime import datetime

from insightx.data.recommendation_data import CustomerProductInteractionBuilder
from insightx.models.recommenders import (
    PopularityRecommender,
    CategoryRecommender,
    CoPurchaseRecommender,
)
from insightx.models.collaborative_filtering import SVDCollaborativeFilter
from insightx.models.content_based import ContentBasedRecommender
from insightx.models.hybrid_recommender import HybridRecommender
from insightx.models.comprehensive_eval import (
    ComprehensiveRecommendationEvaluator,
    tune_hybrid_weights,
)
from insightx.utils.profiling import profile_execution

rng = np.random.default_rng(42)
N_CUST = 150
N_ORDERS = 350
N_PROD = 30
CATEGORIES = ["electronics", "toys", "sports", "home_appliances", "fashion"]

customers = pd.DataFrame({
    "customer_id": [f"cid_{i:04d}" for i in range(N_CUST)],
    "customer_unique_id": [f"uid_{i:04d}" for i in range(N_CUST)],
})

train_dates = pd.date_range("2017-03-01", periods=N_ORDERS - 50, freq="1d")
val_dates   = pd.date_range("2018-04-01", periods=50, freq="1d")
all_dates   = list(train_dates) + list(val_dates)

orders = pd.DataFrame({
    "order_id": [f"ord_{i:04d}" for i in range(N_ORDERS)],
    "customer_id": [f"cid_{int(rng.integers(0, N_CUST)):04d}" for _ in range(N_ORDERS)],
    "order_status": rng.choice(
        ["delivered", "shipped"], size=N_ORDERS, p=[0.85, 0.15]
    ).tolist(),
    "order_purchase_timestamp": [d.strftime("%Y-%m-%d %H:%M:%S") for d in all_dates],
})

items = pd.DataFrame({
    "order_id": rng.choice(orders["order_id"].values, size=500).tolist(),
    "product_id": [f"prod_{int(rng.integers(0, N_PROD)):03d}" for _ in range(500)],
    "price": rng.uniform(20, 500, 500).round(2).tolist(),
    "freight_value": rng.uniform(5, 50, 500).round(2).tolist(),
})

meta = pd.DataFrame({
    "product_id": [f"prod_{i:03d}" for i in range(N_PROD)],
    "product_category_name_english": rng.choice(CATEGORIES, size=N_PROD).tolist(),
    "price_band": rng.choice(["budget", "mid", "premium", "luxury"], size=N_PROD).tolist(),
    "product_weight_g": rng.uniform(100, 5000, N_PROD).round(1).tolist(),
    "product_length_cm": rng.uniform(10, 60, N_PROD).round(1).tolist(),
    "product_height_cm": rng.uniform(5, 40, N_PROD).round(1).tolist(),
    "product_width_cm": rng.uniform(10, 50, N_PROD).round(1).tolist(),
    "product_photos_qty": rng.integers(1, 8, N_PROD).tolist(),
})

TRAIN_CUTOFF = "2018-03-01"
VAL_CUTOFF   = "2018-08-31"

builder_train = CustomerProductInteractionBuilder(cutoff_date=TRAIN_CUTOFF)
train_matrix, _ = builder_train.build_interaction_matrix(orders, customers, items)

val_orders = orders[
    (pd.to_datetime(orders["order_purchase_timestamp"]) > pd.Timestamp(TRAIN_CUTOFF))
    & (pd.to_datetime(orders["order_purchase_timestamp"]) <= pd.Timestamp(VAL_CUTOFF))
].copy()
val_with_uid = val_orders.merge(
    customers[["customer_id", "customer_unique_id"]], on="customer_id", how="left"
)
val_interactions = val_with_uid.merge(
    items[["order_id", "product_id"]], on="order_id", how="inner"
)[["customer_unique_id", "product_id"]].drop_duplicates()

K = 10
evaluator = ComprehensiveRecommendationEvaluator(k=K)

with profile_execution("Phase3_M4_Execution") as prof:
    # 1. Fit base models on training partition only
    pop = PopularityRecommender(top_k=K, exclude_purchased=True).fit(train_matrix)
    cat = CategoryRecommender(top_k=K, exclude_purchased=True).fit(train_matrix, meta)
    cop = CoPurchaseRecommender(top_k=K, exclude_purchased=True).fit(train_matrix)
    cf  = SVDCollaborativeFilter(n_components=5, top_k=K, exclude_purchased=True, random_state=42).fit(train_matrix)
    cbf = ContentBasedRecommender(top_k=K, exclude_purchased=True).fit(meta, train_matrix)

    sub_models = {
        "popularity": (pop, "popularity_score"),
        "category": (cat, "category_score"),
        "copurchase": (cop, "copurchase_score"),
        "svd_cf": (cf, "cf_score"),
        "content_based": (cbf, "cbf_score"),
    }

    # 2. Baseline untuned hybrid (uniform equal weights)
    untuned_components = [
        (cf, 0.25, "cf_score"),
        (cbf, 0.25, "cbf_score"),
        (cop, 0.25, "copurchase_score"),
        (pop, 0.25, "popularity_score"),
    ]
    untuned_hybrid = HybridRecommender(
        components=untuned_components,
        top_k=K,
        exclude_purchased=True,
        normalisation_mode="minmax",
    ).fit(train_matrix)

    # 3. Tune hybrid weights strictly on validation split
    grid_candidates = [
        {"svd_cf": 0.25, "content_based": 0.25, "copurchase": 0.25, "popularity": 0.25},
        {"svd_cf": 0.10, "content_based": 0.20, "copurchase": 0.40, "popularity": 0.30},
        {"svd_cf": 0.05, "content_based": 0.15, "copurchase": 0.50, "popularity": 0.30},
        {"svd_cf": 0.00, "content_based": 0.20, "copurchase": 0.50, "popularity": 0.30},
        {"svd_cf": 0.10, "content_based": 0.10, "copurchase": 0.60, "popularity": 0.20},
    ]
    best_weights, best_ndcg, tuning_trials = tune_hybrid_weights(
        train_interaction_df=train_matrix,
        val_interaction_df=val_interactions,
        product_metadata_df=meta,
        sub_models=sub_models,
        grid_weights=grid_candidates,
        k=K,
        metric="ndcg_at_k",
        normalisation_mode="minmax",
    )

    tuned_components = [
        (sub_models[name][0], w, sub_models[name][1])
        for name, w in best_weights.items()
    ]
    tuned_hybrid = HybridRecommender(
        components=tuned_components,
        top_k=K,
        exclude_purchased=True,
        normalisation_mode="minmax",
    ).fit(train_matrix)

    # 4. Evaluate all recommenders
    eval_models = [
        ("popularity", pop),
        ("category", cat),
        ("copurchase", cop),
        ("svd_cf", cf),
        ("content_based", cbf),
        ("hybrid_untuned", untuned_hybrid),
        ("hybrid_tuned", tuned_hybrid),
    ]

    results = []
    for name, model in eval_models:
        r = evaluator.evaluate(model, val_interactions, train_matrix, name, meta)
        results.append(r)

    # 5. Segment-wise breakdown for best hybrid and baselines
    segment_evals = {}
    for name in ["popularity", "copurchase", "hybrid_tuned"]:
        model = dict(eval_models)[name]
        segment_evals[name] = evaluator.evaluate_segments(
            model, val_interactions, train_matrix, name, meta
        )

rpt = prof["report"]

print("\n" + "=" * 90)
print(f"MILESTONE 4 COMPREHENSIVE BENCHMARK (K={K})")
print("=" * 90)
print(evaluator.compare(results).to_string(index=False))

print("\n" + "=" * 90)
print("HYBRID WEIGHT TUNING (Strictly Chronological Validation Data)")
print("=" * 90)
print(f"Best Weights: {best_weights}")
print(f"Best Val NDCG@10: {best_ndcg:.6f}")
print("\nTrial History:")
print(tuning_trials[["trial_idx", "ndcg_at_k", "recall_at_k", "precision_at_k", "coverage_at_k", "diversity", "gini"]].to_string(index=False))

print("\n" + "=" * 90)
print("SEGMENT-WISE PERFORMANCE BREAKDOWN (Repeat vs Sparse vs Cold-Start)")
print("=" * 90)
seg_rows = []
for model_name, seg_dict in segment_evals.items():
    for seg_name, res in seg_dict.items():
        seg_rows.append({
            "model": model_name,
            "segment": seg_name,
            "P@10": res.get("precision_at_k", 0.0),
            "R@10": res.get("recall_at_k", 0.0),
            "NDCG@10": res.get("ndcg_at_k", 0.0),
            "Coverage": res.get("catalog_coverage_at_k", 0.0),
            "Cohort": f"{res.get('n_eval_customers_with_relevant', 0)}/{res.get('n_eval_customers_total', 0)}",
        })
print(pd.DataFrame(seg_rows).to_string(index=False))

print("\n" + "=" * 90)
print("PROFILING METRICS")
print("=" * 90)
print(f"Duration:         {rpt.duration_seconds:.4f} s")
print(f"Python Heap Peak: {rpt.python_heap_peak_mb:.3f} MB")
print(f"Process Peak RSS: {rpt.process_peak_rss_mb:.3f} MB")
limits_ok = rpt.duration_seconds < 15 and rpt.process_peak_rss_mb < 250
print(f"Limits satisfied: {'YES' if limits_ok else 'NO -- EXCEEDED'}")
