"""Milestone 5 End-to-End Pipeline Profiling & Stress Verification."""
import sys
sys.path.insert(0, "c:/Users/khushi/OneDrive/Desktop/InsightX-AI/src")

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from insightx.models.recommendation_pipeline import RecommendationIntelligencePipeline
from insightx.utils.profiling import profile_execution

rng = np.random.default_rng(101)
n_customers = 100
n_products = 50
n_orders = 250

customers = pd.DataFrame({
    "customer_id": [f"c_{i:04d}" for i in range(n_customers)],
    "customer_unique_id": [f"u_{i:04d}" for i in range(n_customers)],
})

base_date = datetime(2017, 3, 1)
orders = pd.DataFrame({
    "order_id": [f"o_{i:04d}" for i in range(n_orders)],
    "customer_id": [f"c_{int(rng.integers(0, n_customers)):04d}" for _ in range(n_orders)],
    "order_status": rng.choice(["delivered", "shipped"], size=n_orders, p=[0.9, 0.1]).tolist(),
    "order_purchase_timestamp": [
        (base_date + timedelta(days=int(rng.integers(0, 400)))).strftime("%Y-%m-%d %H:%M:%S")
        for _ in range(n_orders)
    ],
})

items = pd.DataFrame({
    "order_id": rng.choice(orders["order_id"].values, size=400).tolist(),
    "product_id": [f"p_{int(rng.integers(0, n_products)):03d}" for _ in range(400)],
    "price": rng.uniform(15, 600, 400).round(2).tolist(),
    "freight_value": rng.uniform(5, 50, 400).round(2).tolist(),
})

products = pd.DataFrame({
    "product_id": [f"p_{i:03d}" for i in range(n_products)],
    "product_category_name": rng.choice(["eletronicos", "brinquedos", "beleza_saude"], size=n_products).tolist(),
    "product_weight_g": rng.uniform(100, 5000, n_products).round(1).tolist(),
    "product_length_cm": rng.uniform(10, 60, n_products).round(1).tolist(),
    "product_height_cm": rng.uniform(5, 40, n_products).round(1).tolist(),
    "product_width_cm": rng.uniform(10, 50, n_products).round(1).tolist(),
    "product_photos_qty": rng.integers(1, 8, n_products).tolist(),
})

translations = pd.DataFrame({
    "product_category_name": ["eletronicos", "brinquedos", "beleza_saude"],
    "product_category_name_english": ["electronics", "toys", "health_beauty"],
})

with profile_execution("E2E_Recommendation_Intelligence_Pipeline") as prof:
    pipeline = RecommendationIntelligencePipeline(cutoff_date="2018-03-01", top_k=10)
    pipeline.fit(
        orders_df=orders,
        customers_df=customers,
        order_items_df=items,
        products_df=products,
        category_translation_df=translations,
    )

    # Generate recommendations for multiple customers across segments
    for cid in ["u_0001", "u_0010", "u_9999_cold"]:
        recs = pipeline.recommend_for_customer(cid, k=10)

    # Split for validation
    val_orders = orders[pd.to_datetime(orders["order_purchase_timestamp"]) > pd.Timestamp("2018-03-01")]
    val_with_uid = val_orders.merge(customers[["customer_id", "customer_unique_id"]], on="customer_id")
    val_interactions = val_with_uid.merge(
        items[["order_id", "product_id"]], on="order_id"
    )[["customer_unique_id", "product_id"]].drop_duplicates()

    if len(val_interactions) > 0:
        eval_metrics = pipeline.evaluate_offline(val_interactions, k=10)

rpt = prof["report"]
print(f"Duration:         {rpt.duration_seconds:.4f} s")
print(f"Python Heap Peak: {rpt.python_heap_peak_mb:.3f} MB")
print(f"Process Peak RSS: {rpt.process_peak_rss_mb:.3f} MB")
limits_ok = rpt.duration_seconds < 15 and rpt.process_peak_rss_mb < 250
print(f"Limits satisfied: {'YES' if limits_ok else 'NO -- EXCEEDED'}")
print(f"Evidence trail:   {len(pipeline.evidence_trail)} records preserved")
