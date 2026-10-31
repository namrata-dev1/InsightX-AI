"""Repeat purchase prediction model (Supervised Binary Classification)."""

from typing import Any, Dict, Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from insightx.utils.logger import get_logger

logger = get_logger(__name__)


class RepeatPurchasePredictor:
    """Trains, compares, and evaluates classification models for repeat buyer prediction."""

    def __init__(self, model_type: str = "lightgbm", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model: Optional[Any] = None

    def prepare_data(
        self, df: pd.DataFrame, target_col: str = "is_repeat_buyer", test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Performs stratified train/test split to handle class imbalance."""
        logger.info(f"Splitting dataset into train and test sets (test_size={test_size})...")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Trains the chosen classification model."""
        logger.info(f"Training {self.model_type} model on {len(X_train)} samples...")

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Calculates evaluation metrics (ROC-AUC, Precision, Recall, F1-Score)."""
        logger.info("Evaluating model performance on test set...")
        return {"roc_auc": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
