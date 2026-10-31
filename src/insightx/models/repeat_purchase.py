"""Repeat Purchase Prediction Model (Supervised Binary Classification).

Integrates with the point-in-time CustomerFeaturePipeline and supports
Logistic Regression, Random Forest, and Gradient Boosting baselines.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from insightx.utils.logger import get_logger

logger = get_logger(__name__)


class RepeatPurchasePredictor:
    """Trains, compares, and evaluates classification models for repeat buyer prediction."""

    def __init__(self, model_type: str = "logistic_regression", random_state: int = 42):
        """Args:
            model_type: 'logistic_regression' or 'random_forest'.
            random_state: Reproducibility seed.
        """
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model: Optional[Any] = None
        self.features_fitted_: List[str] = []

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = "is_repeat_buyer",
        test_size: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Performs stratified train/test split to handle class imbalance."""
        logger.info(f"Splitting dataset into train and test sets (test_size={test_size})...")
        drop_cols = [c for c in [target_col, "customer_unique_id", "customer_state", "latest_order_timestamp", "cutoff_timestamp"] if c in df.columns]
        X = df.drop(columns=drop_cols).select_dtypes(include=[np.number]).fillna(0.0)
        y = df[target_col].astype(int)

        # Check if stratification is possible (at least 2 samples per class)
        stratify = y if y.value_counts().min() >= 2 else None

        return train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=stratify
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Trains the chosen classification model."""
        self.features_fitted_ = list(X_train.columns)
        scaled_X = self.scaler.fit_transform(X_train.values)
        y = y_train.values

        logger.info(f"Training {self.model_type} model on {len(X_train)} samples...")

        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                max_depth=6,
                random_state=self.random_state,
            )
        else:
            self.model = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=self.random_state,
            )

        self.model.fit(scaled_X, y)

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predicts class probabilities."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        scaled_X = self.scaler.transform(X_test[self.features_fitted_].values)
        return self.model.predict_proba(scaled_X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Calculates evaluation metrics (ROC-AUC, PR-AUC, Precision, Recall, F1-Score)."""
        logger.info("Evaluating model performance on test set...")
        probs = self.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        y = y_test.values

        if len(np.unique(y)) > 1:
            roc_auc = float(roc_auc_score(y, probs))
            pr_auc = float(average_precision_score(y, probs))
        else:
            roc_auc = 0.5
            pr_auc = float(np.mean(y))

        return {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "f1": round(float(f1_score(y, preds, zero_division=0)), 4),
            "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y, preds, zero_division=0)), 4),
        }
