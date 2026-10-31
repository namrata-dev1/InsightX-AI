"""Churn Risk Prediction Engine for InsightX-AI 3.0.

Provides:
1. Heuristic inactivity baseline predictor.
2. Statistical Logistic Regression baseline with balanced class weighting.
3. Cost-sensitive Random Forest and Gradient Boosting architectures.
4. Comprehensive imbalanced-class evaluation: PR-AUC, ROC-AUC, Brier score, Precision@Top-K%.
5. Probability calibration and feature attribution.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from insightx.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_FEATURE_COLS = [
    "recency_days",
    "frequency",
    "total_orders",
    "total_monetary",
    "avg_order_value",
    "avg_basket_size",
    "freight_ratio",
    "avg_installments",
    "credit_card_share",
    "boleto_share",
    "voucher_share",
    "avg_review_score",
    "low_review_ratio",
    "on_time_delivery_rate",
    "distinct_categories_count",
]


class HeuristicChurnBaseline:
    """Simple rule-based inactivity baseline predicting churn based solely on recency cutoff."""

    def __init__(self, inactivity_threshold_days: float = 180.0):
        self.inactivity_threshold_days = inactivity_threshold_days

    def predict_proba(self, X: pd.DataFrame, recency_col: str = "recency_days") -> np.ndarray:
        """Computes rule-based probability of churn."""
        rec = X[recency_col].fillna(self.inactivity_threshold_days).values
        # Sigmoid decay around threshold
        prob_churn = 1.0 / (1.0 + np.exp(-(rec - self.inactivity_threshold_days) / 30.0))
        return np.column_stack([1.0 - prob_churn, prob_churn])

    def predict(self, X: pd.DataFrame, recency_col: str = "recency_days") -> np.ndarray:
        """Predicts binary churn (1 = churn, 0 = active)."""
        rec = X[recency_col].fillna(self.inactivity_threshold_days).values
        return (rec >= self.inactivity_threshold_days).astype(int)


class ChurnRiskPredictor:
    """Supervised classification engine for non-contractual customer churn prediction."""

    def __init__(
        self,
        model_type: str = "logistic_regression",
        random_state: int = 42,
        calibrate_probabilities: bool = True,
        feature_cols: Optional[List[str]] = None,
    ):
        """Args:
            model_type: 'heuristic', 'logistic_regression', or 'random_forest'.
            random_state: Reproducibility seed.
            calibrate_probabilities: If True, wraps model in CalibratedClassifierCV.
            feature_cols: List of numerical feature columns.
        """
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.calibrate_probabilities = calibrate_probabilities
        self.feature_cols = feature_cols or DEFAULT_FEATURE_COLS

        self.scaler = StandardScaler()
        self.model: Optional[Any] = None
        self.is_fitted: bool = False
        self.features_fitted_: List[str] = []

    def _init_base_model(self) -> Any:
        if self.model_type == "heuristic":
            return HeuristicChurnBaseline(inactivity_threshold_days=180.0)
        elif self.model_type == "logistic_regression":
            return LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=self.random_state,
                C=1.0,
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=self.random_state,
                max_depth=6,
                min_samples_split=5,
            )
        else:
            logger.warning(f"Unknown model_type '{self.model_type}'. Defaulting to logistic_regression.")
            return LogisticRegression(class_weight="balanced", random_state=self.random_state)

    def prepare_features(self, df: pd.DataFrame, fit_scaler: bool = True) -> np.ndarray:
        """Selects feature columns, fills missing values, and applies scaling."""
        available_cols = [c for c in self.feature_cols if c in df.columns]
        if not available_cols:
            available_cols = list(df.select_dtypes(include=[np.number]).columns)

        self.features_fitted_ = available_cols
        X = df[available_cols].copy()

        # Numeric imputation
        for col in available_cols:
            median_val = float(X[col].median()) if not X[col].isna().all() else 0.0
            X[col] = X[col].fillna(median_val)

        vals = X.values
        if self.model_type == "heuristic":
            return vals

        if fit_scaler:
            return self.scaler.fit_transform(vals)
        else:
            return self.scaler.transform(vals)

    def train(self, X_train: pd.DataFrame, y_train: Union[pd.Series, np.ndarray]) -> None:
        """Trains the chosen classification model with balanced weights and calibration.

        Args:
            X_train: Training feature DataFrame.
            y_train: Binary churn target (1 = churned, 0 = active/repeat).
        """
        y = np.asarray(y_train).astype(int)
        logger.info(
            f"Training {self.model_type} churn model on {len(X_train)} samples. "
            f"Class balance: {np.mean(y == 1)*100:.1f}% Churn (1), {np.mean(y == 0)*100:.1f}% Retained (0)."
        )

        base_model = self._init_base_model()

        if self.model_type == "heuristic":
            self.model = base_model
            self.is_fitted = True
            return

        scaled_X = self.prepare_features(X_train, fit_scaler=True)

        # Check if both classes are present
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            logger.warning(
                f"Only 1 target class ({unique_classes[0]}) present in training data. "
                "Registering constant probability predictor."
            )
            self.single_class_ = int(unique_classes[0])
            self.is_fitted = True
            return

        self.single_class_ = None

        if self.calibrate_probabilities and len(scaled_X) >= 30:
            # Calibrated probabilities via 3-fold CV
            self.model = CalibratedClassifierCV(estimator=base_model, cv=3, method="sigmoid")
            self.model.fit(scaled_X, y)
        else:
            base_model.fit(scaled_X, y)
            self.model = base_model

        self.is_fitted = True
        logger.info(f"{self.model_type} model training completed.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts probability of churn [P(retained), P(churned)]."""
        if not self.is_fitted:
            raise RuntimeError("ChurnRiskPredictor has not been fitted yet.")

        if hasattr(self, "single_class_") and self.single_class_ is not None:
            n_samples = len(X)
            probs = np.zeros((n_samples, 2))
            if self.single_class_ == 1:
                probs[:, 1] = 1.0
            else:
                probs[:, 0] = 1.0
            return probs

        if self.model_type == "heuristic":
            return self.model.predict_proba(X)

        scaled_X = self.prepare_features(X, fit_scaler=False)
        return self.model.predict_proba(scaled_X)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predicts binary churn label using specified decision threshold."""
        prob_churn = self.predict_proba(X)[:, 1]
        return (prob_churn >= threshold).astype(int)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: Union[pd.Series, np.ndarray],
        top_k_pct: float = 0.10,
    ) -> Dict[str, float]:
        """Calculates comprehensive imbalanced-class evaluation metrics.

        Metrics:
            - PR-AUC (Average Precision Score): Primary metric for severe imbalance.
            - ROC-AUC: Discrimination ability.
            - Brier Score: Calibration quality (lower is better, 0 = perfect).
            - Precision, Recall, F1 at default 0.5 threshold.
            - Precision@Top-K%: Precision among top K% highest-risk predicted customers.

        Args:
            X_test: Test features.
            y_test: True binary churn labels.
            top_k_pct: Percentile for Top-K precision (default: 0.10 / top 10%).

        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        y_true = np.asarray(y_test).astype(int)
        probs = self.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        # Basic classification metrics
        prec = float(precision_score(y_true, preds, zero_division=0))
        rec = float(recall_score(y_true, preds, zero_division=0))
        f1 = float(f1_score(y_true, preds, zero_division=0))
        brier = float(brier_score_loss(y_true, probs))

        # Check for single-class test set
        if len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, probs))
            pr_auc = float(average_precision_score(y_true, probs))
        else:
            roc_auc = 0.5
            pr_auc = float(np.mean(y_true))

        # Precision@Top-K%
        n_top = max(int(np.ceil(len(y_true) * top_k_pct)), 1)
        top_indices = np.argsort(probs)[::-1][:n_top]
        precision_at_k = float(np.mean(y_true[top_indices]))

        metrics = {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            f"precision_at_top_{int(top_k_pct*100)}_pct": round(precision_at_k, 4),
            "test_sample_size": len(y_true),
            "churn_rate_actual": round(float(np.mean(y_true)), 4),
        }

        logger.info(
            f"Evaluation [{self.model_type}]: PR-AUC={metrics['pr_auc']:.4f}, "
            f"ROC-AUC={metrics['roc_auc']:.4f}, Brier={metrics['brier_score']:.4f}, "
            f"Precision@Top10%={precision_at_k:.4f}."
        )
        return metrics
