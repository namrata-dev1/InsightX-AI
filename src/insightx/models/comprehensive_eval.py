"""Phase 3, Milestone 4: Comprehensive Recommendation Evaluator & Ranker.

Enhances Phase 3 evaluation with:
  1. Intra-List Diversity (ILD) based on product category pairwise distance.
  2. Popularity Bias:
     - Average Recommendation Popularity (ARP)
     - Gini index of recommended item frequency distribution
  3. Segment-wise evaluation:
     - Repeat buyers (>= 2 purchases in training)
     - Sparse buyers (1 purchase in training)
     - Cold-start customers (0 purchases in training)
  4. Chronological validation weight tuning for HybridRecommender (grid search).

Preserves all prior evaluator signatures and methods for backward compatibility.
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import itertools
import numpy as np
import pandas as pd

from insightx.models.recommendation_eval import (
    RecommendationEvaluator as BaseRecommendationEvaluator,
    _precision_at_k,
    _recall_at_k,
    _f1_at_k,
    _ndcg_at_k,
)
from insightx.utils.logger import get_logger

logger = get_logger(__name__)


def _intra_list_diversity(
    recommended: List[str],
    category_map: Dict[str, str],
) -> float:
    """Computes category-based pairwise distance intra-list diversity (ILD).

    ILD = (2 / (K * (K - 1))) * sum_{i < j} dist(p_i, p_j)
    where dist(p_i, p_j) = 1.0 if categories differ, 0.0 if categories match.
    Returns 0.0 if len(recommended) < 2.
    """
    n = len(recommended)
    if n < 2:
        return 0.0
    diff_count = 0
    total_pairs = 0
    for p1, p2 in itertools.combinations(recommended, 2):
        c1 = category_map.get(str(p1), "unknown")
        c2 = category_map.get(str(p2), "unknown")
        if c1 != c2:
            diff_count += 1
        total_pairs += 1
    return diff_count / total_pairs if total_pairs > 0 else 0.0


def _average_recommendation_popularity(
    all_recommended: List[List[str]],
    product_popularity_map: Dict[str, float],
) -> float:
    """Average recommendation popularity (ARP).

    ARP = (1 / |U|) * sum_{u} (1 / |L_u|) * sum_{p in L_u} pop(p)
    """
    if not all_recommended:
        return 0.0
    user_pops = []
    for rec_list in all_recommended:
        if not rec_list:
            continue
        list_pops = [product_popularity_map.get(str(p), 0.0) for p in rec_list]
        user_pops.append(float(np.mean(list_pops)))
    return float(np.mean(user_pops)) if user_pops else 0.0


def _gini_index(frequencies: np.ndarray) -> float:
    """Calculates Gini coefficient of item recommendation distribution.

    Gini = 0 represents perfect equality across recommended catalog items;
    Gini -> 1 represents extreme concentration on a handful of top items.
    """
    if len(frequencies) == 0:
        return 0.0
    sorted_f = np.sort(frequencies)
    n = len(sorted_f)
    if np.sum(sorted_f) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * sorted_f)) / (n * np.sum(sorted_f)) - (n + 1) / n)


class ComprehensiveRecommendationEvaluator(BaseRecommendationEvaluator):
    """Extended Recommendation Evaluator providing diversity, popularity bias, and customer segmentation.

    Fully backwards-compatible with RecommendationEvaluator.
    """

    def evaluate(
        self,
        recommender: Any,
        eval_interaction_df: pd.DataFrame,
        train_interaction_df: pd.DataFrame,
        recommender_name: str = "baseline",
        product_metadata_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Evaluates recommender with standard ranking, diversity, and popularity bias metrics."""
        # 1. Run base metrics
        base_result = super().evaluate(
            recommender=recommender,
            eval_interaction_df=eval_interaction_df,
            train_interaction_df=train_interaction_df,
            recommender_name=recommender_name,
            product_metadata_df=product_metadata_df,
        )

        # 2. Extract catalog and popularity maps
        if "purchase_count" in train_interaction_df.columns:
            pop_map = (
                train_interaction_df.groupby("product_id")["purchase_count"]
                .sum()
                .to_dict()
            )
        else:
            pop_map = (
                train_interaction_df.groupby("product_id")
                .size()
                .to_dict()
            )

        cat_map: Dict[str, str] = {}
        if product_metadata_df is not None:
            cat_col = (
                "product_category_name_english"
                if "product_category_name_english" in product_metadata_df.columns
                else "product_category_name"
            )
            if cat_col in product_metadata_df.columns:
                cat_map = product_metadata_df.set_index("product_id")[cat_col].fillna("unknown").to_dict()

        # 3. Collect per-customer recommendations in eval set
        eval_cids = eval_interaction_df["customer_unique_id"].unique()
        all_rec_lists: List[List[str]] = []
        diversity_scores: List[float] = []
        rec_frequency: Dict[str, int] = {}

        train_cids = (
            set(train_interaction_df["customer_unique_id"].unique())
            if "customer_unique_id" in train_interaction_df.columns
            else set()
        )
        cold_rec_cache_str: Optional[List[str]] = None
        has_cold_cache_str = False

        for cid in eval_cids:
            if cid not in train_cids:
                if not has_cold_cache_str:
                    try:
                        rec_df = recommender.recommend(customer_id=cid, k=self.k)
                        cold_rec_cache_str = rec_df["product_id"].astype(str).tolist() if len(rec_df) > 0 else []
                    except Exception:
                        cold_rec_cache_str = []
                    has_cold_cache_str = True
                recs = cold_rec_cache_str
            else:
                try:
                    rec_df = recommender.recommend(customer_id=cid, k=self.k)
                    recs = rec_df["product_id"].astype(str).tolist() if len(rec_df) > 0 else []
                except Exception:
                    recs = []

            all_rec_lists.append(recs)
            if len(recs) >= 2 and cat_map:
                diversity_scores.append(_intra_list_diversity(recs, cat_map))
            for p in recs:
                rec_frequency[p] = rec_frequency.get(p, 0) + 1

        arp = _average_recommendation_popularity(all_rec_lists, pop_map)
        mean_diversity = float(np.mean(diversity_scores)) if diversity_scores else 0.0

        catalog_prods = list(pop_map.keys())
        freq_array = np.array([rec_frequency.get(str(p), 0) for p in catalog_prods], dtype=float)
        gini = _gini_index(freq_array)

        # Augment result
        base_result["intra_list_diversity"] = round(mean_diversity, 6)
        base_result["avg_recommendation_popularity"] = round(arp, 6)
        base_result["popularity_gini"] = round(gini, 6)
        return base_result

    def evaluate_segments(
        self,
        recommender: Any,
        eval_interaction_df: pd.DataFrame,
        train_interaction_df: pd.DataFrame,
        recommender_name: str = "baseline",
        product_metadata_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluates customer performance partitioned by training activity.

        Segments:
          - repeat_customers: >= 2 purchases in train
          - sparse_customers: exactly 1 purchase in train
          - cold_start_customers: 0 purchases in train (unseen before eval)
        """
        train_counts = (
            train_interaction_df.groupby("customer_unique_id")["purchase_count"].sum()
            if "purchase_count" in train_interaction_df.columns
            else train_interaction_df.groupby("customer_unique_id").size()
        ).to_dict()

        eval_uids = eval_interaction_df["customer_unique_id"].unique()
        repeat_uids = {u for u in eval_uids if train_counts.get(u, 0) >= 2}
        sparse_uids = {u for u in eval_uids if train_counts.get(u, 0) == 1}
        cold_uids = {u for u in eval_uids if train_counts.get(u, 0) == 0}

        segments = {
            "repeat": eval_interaction_df[eval_interaction_df["customer_unique_id"].isin(repeat_uids)],
            "sparse": eval_interaction_df[eval_interaction_df["customer_unique_id"].isin(sparse_uids)],
            "cold_start": eval_interaction_df[eval_interaction_df["customer_unique_id"].isin(cold_uids)],
        }

        segment_results = {}
        for seg_name, seg_df in segments.items():
            if len(seg_df) == 0:
                segment_results[seg_name] = self._zero_result(
                    recommender_name=f"{recommender_name}_{seg_name}",
                    n_eval_total=0,
                    train_df=train_interaction_df,
                    meta_df=product_metadata_df,
                )
                segment_results[seg_name]["intra_list_diversity"] = 0.0
                segment_results[seg_name]["avg_recommendation_popularity"] = 0.0
                segment_results[seg_name]["popularity_gini"] = 0.0
            else:
                segment_results[seg_name] = self.evaluate(
                    recommender=recommender,
                    eval_interaction_df=seg_df,
                    train_interaction_df=train_interaction_df,
                    recommender_name=f"{recommender_name}_{seg_name}",
                    product_metadata_df=product_metadata_df,
                )
        return segment_results

    def compare(
        self,
        results: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Extended comparison DataFrame including Diversity, ARP, and Gini."""
        rows = []
        for r in results:
            rows.append({
                "recommender": r.get("recommender_name", "?"),
                "K": r.get("k", self.k),
                "Precision@K": r.get("precision_at_k", 0.0),
                "Recall@K": r.get("recall_at_k", 0.0),
                "F1@K": r.get("f1_at_k", 0.0),
                "NDCG@K": r.get("ndcg_at_k", 0.0),
                "Coverage@K": r.get("catalog_coverage_at_k", 0.0),
                "Diversity": r.get("intra_list_diversity", 0.0),
                "ARP": r.get("avg_recommendation_popularity", 0.0),
                "Gini": r.get("popularity_gini", 0.0),
                "Cohort": f"{r.get('n_eval_customers_with_relevant', 0)}/{r.get('n_eval_customers_total', 0)}",
            })
        return pd.DataFrame(rows)


def tune_hybrid_weights(
    train_interaction_df: pd.DataFrame,
    val_interaction_df: pd.DataFrame,
    product_metadata_df: pd.DataFrame,
    sub_models: Dict[str, Tuple[Any, str]],
    grid_weights: Optional[List[Dict[str, float]]] = None,
    k: int = 10,
    metric: str = "ndcg_at_k",
    normalisation_mode: str = "minmax",
) -> Tuple[Dict[str, float], float, pd.DataFrame]:
    """Tuning utility for HybridRecommender on chronological validation data only.

    Never accesses test data. Evaluates permutations of weights and selects the
    configuration that maximizes the chosen metric (default: NDCG@K).

    Args:
        train_interaction_df: Train partition.
        val_interaction_df: Validation partition strictly following train cutoff.
        product_metadata_df: Metadata catalog.
        sub_models: Mapping model_name -> (fitted_recommender_instance, score_column_name).
        grid_weights: List of candidate weight dictionaries.
        k: Top-K ranking depth.
        metric: Optimization metric (e.g. 'ndcg_at_k', 'recall_at_k', 'f1_at_k').
        normalisation_mode: Normalization mode passed to HybridRecommender ('minmax', 'zscore', 'rank').

    Returns:
        (best_weights_dict, best_metric_value, trial_history_df)
    """
    from insightx.models.hybrid_recommender import HybridRecommender

    evaluator = ComprehensiveRecommendationEvaluator(k=k)

    if grid_weights is None:
        # Balanced defaults exploring collaborative, content, co-occurrence, and popularity
        grid_weights = [
            {"svd_cf": 0.25, "content_based": 0.25, "copurchase": 0.25, "popularity": 0.25},
            {"svd_cf": 0.10, "content_based": 0.20, "copurchase": 0.40, "popularity": 0.30},
            {"svd_cf": 0.05, "content_based": 0.15, "copurchase": 0.50, "popularity": 0.30},
            {"svd_cf": 0.40, "content_based": 0.30, "copurchase": 0.20, "popularity": 0.10},
            {"svd_cf": 0.00, "content_based": 0.20, "copurchase": 0.50, "popularity": 0.30},
        ]

    trials = []
    best_val = -1.0
    best_w = grid_weights[0]

    for trial_idx, weights in enumerate(grid_weights):
        components = []
        for name, w in weights.items():
            if name in sub_models:
                model, score_col = sub_models[name]
                components.append((model, float(w), score_col))

        hybrid = HybridRecommender(
            components=components,
            top_k=k,
            exclude_purchased=True,
            normalisation_mode=normalisation_mode,
        ).fit(train_interaction_df)

        res = evaluator.evaluate(
            recommender=hybrid,
            eval_interaction_df=val_interaction_df,
            train_interaction_df=train_interaction_df,
            recommender_name=f"hybrid_trial_{trial_idx}",
            product_metadata_df=product_metadata_df,
        )

        score = float(res.get(metric, 0.0))
        trials.append({
            "trial_idx": trial_idx,
            "weights": weights,
            metric: score,
            "precision_at_k": res.get("precision_at_k", 0.0),
            "recall_at_k": res.get("recall_at_k", 0.0),
            "ndcg_at_k": res.get("ndcg_at_k", 0.0),
            "coverage_at_k": res.get("catalog_coverage_at_k", 0.0),
            "diversity": res.get("intra_list_diversity", 0.0),
            "gini": res.get("popularity_gini", 0.0),
        })

        if score > best_val:
            best_val = score
            best_w = weights

    return best_w, best_val, pd.DataFrame(trials)
