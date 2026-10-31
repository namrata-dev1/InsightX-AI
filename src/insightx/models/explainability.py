"""Explainable Machine Learning and Feature Attribution Engine.

Supports:
1. Dynamic SHAP integration (TreeExplainer, LinearExplainer) when `shap` is installed (e.g. Colab).
2. Graceful model-native fallback (standardized coefficients, Gini importance) when SHAP is not installed locally.
3. Global feature importance and individual customer local attribution.
4. Explicit distinction between predictive correlation and causal attribution.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from insightx.utils.logger import get_logger

logger = get_logger(__name__)

# Check for optional SHAP dependency
try:
    import shap
    HAS_SHAP = True
except ImportError:
    shap = None
    HAS_SHAP = False


class ModelExplainer:
    """Computes global feature importance and individual sample local attribution."""

    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        background_samples: Optional[pd.DataFrame] = None,
        random_state: int = 42,
    ):
        """Args:
            model: Trained scikit-learn model, pipeline, or predictor instance.
            feature_names: Names of feature columns.
            background_samples: Background samples for SHAP explainers (optional).
            random_state: Random seed for sampling.
        """
        self.model = model
        self.feature_names = feature_names or []
        self.background_samples = background_samples
        self.random_state = random_state
        self.explainer: Optional[Any] = None
        self.explainer_type: str = "native"

        self._resolve_underlying_model()

    def _resolve_underlying_model(self) -> None:
        """Extracts the underlying estimator if wrapped in ChurnRiskPredictor or CalibratedClassifierCV."""
        m = self.model
        if hasattr(m, "model") and m.model is not None:
            m = m.model

        # If wrapped in CalibratedClassifierCV, extract base estimator
        if hasattr(m, "calibrated_classifiers_") and m.calibrated_classifiers_:
            self.underlying_estimator = m.calibrated_classifiers_[0].estimator
        elif hasattr(m, "estimator"):
            self.underlying_estimator = m.estimator
        else:
            self.underlying_estimator = m

    def fit_explainer(self) -> None:
        """Initializes the SHAP explainer if available; otherwise sets up native explainer."""
        if not HAS_SHAP:
            logger.info("SHAP package not installed in current environment. Using model-native feature attribution engine.")
            self.explainer_type = "model_native"
            return

        estimator = self.underlying_estimator
        model_name = type(estimator).__name__.lower()

        try:
            if "forest" in model_name or "tree" in model_name or "lgbm" in model_name or "xgb" in model_name:
                self.explainer = shap.TreeExplainer(estimator)
                self.explainer_type = "shap_tree"
                logger.info("Initialized SHAP TreeExplainer.")
            elif "logistic" in model_name or "linear" in model_name:
                if self.background_samples is not None:
                    bg = self.background_samples.select_dtypes(include=[np.number]).values
                    self.explainer = shap.LinearExplainer(estimator, bg)
                else:
                    self.explainer = shap.LinearExplainer(estimator)
                self.explainer_type = "shap_linear"
                logger.info("Initialized SHAP LinearExplainer.")
            else:
                self.explainer_type = "model_native"
                logger.info("Using model-native explainer for unsupported model class.")
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}. Falling back to model-native attribution.")
            self.explainer_type = "model_native"

    def compute_global_importance(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Computes global feature importance ranking.

        Returns:
            pd.DataFrame: Sorted table with feature, importance_score, normalized_score, and direction.
        """
        estimator = self.underlying_estimator
        features = self.feature_names
        if X is not None and not features:
            features = [c for c in X.columns if c in X.select_dtypes(include=[np.number]).columns]

        # 1. SHAP-based global importance if explainer is fitted
        if HAS_SHAP and self.explainer is not None and X is not None:
            try:
                numeric_X = X[features].select_dtypes(include=[np.number]).fillna(0.0).values
                shap_vals = self.explainer.shap_values(numeric_X)
                if isinstance(shap_vals, list) and len(shap_vals) == 2:
                    # Binary classification: positive class values
                    vals = np.abs(shap_vals[1]).mean(axis=0)
                else:
                    vals = np.abs(shap_vals).mean(axis=0)

                df_imp = pd.DataFrame({
                    "feature": features,
                    "importance_score": np.round(vals, 4),
                    "attribution_method": "shap_mean_absolute",
                })
                df_imp["normalized_score"] = np.round(
                    df_imp["importance_score"] / np.maximum(df_imp["importance_score"].sum(), 1e-6), 4
                )
                return df_imp.sort_values(by="importance_score", ascending=False).reset_index(drop=True)
            except Exception as e:
                logger.warning(f"SHAP global calculation failed: {e}. Falling back to model-native.")

        # 2. Model-native feature importance
        raw_scores = []
        directions = []

        if hasattr(estimator, "feature_importances_"):
            # Tree models (RandomForest, LightGBM, XGBoost)
            raw_scores = estimator.feature_importances_
            directions = ["positive"] * len(raw_scores)
            method = "gini_split_importance"
        elif hasattr(estimator, "coef_"):
            # Linear / Logistic models
            coefs = estimator.coef_[0] if estimator.coef_.ndim == 2 else estimator.coef_
            raw_scores = np.abs(coefs)
            directions = ["increases_risk" if c > 0 else "decreases_risk" for c in coefs]
            method = "absolute_standardized_coefficient"
        else:
            # Fallback uniform
            raw_scores = np.ones(len(features))
            directions = ["neutral"] * len(features)
            method = "uniform_heuristic"

        n_feats = min(len(features), len(raw_scores))
        features_clean = features[:n_feats] if features else [f"feature_{i}" for i in range(n_feats)]
        scores_clean = raw_scores[:n_feats]
        directions_clean = directions[:n_feats]

        df_imp = pd.DataFrame({
            "feature": features_clean,
            "importance_score": np.round(scores_clean, 4),
            "direction": directions_clean,
            "attribution_method": method,
        })
        total_score = df_imp["importance_score"].sum()
        df_imp["normalized_score"] = np.round(
            df_imp["importance_score"] / np.maximum(total_score, 1e-6), 4
        )
        return df_imp.sort_values(by="importance_score", ascending=False).reset_index(drop=True)

    def explain_instance(
        self,
        instance_row: Union[pd.Series, pd.DataFrame],
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """Explains an individual customer's prediction via local feature attribution.

        Args:
            instance_row: Single customer feature record.
            top_n: Number of top risk and top mitigating factors to report.

        Returns:
            Dict containing predicted_risk, top_risk_drivers, top_mitigating_factors,
            and an explicit methodology disclaimer distinguishing correlation from causation.
        """
        row = instance_row.iloc[0] if isinstance(instance_row, pd.DataFrame) else instance_row
        estimator = self.underlying_estimator

        features = [c for c in row.index if c in self.feature_names] if self.feature_names else list(row.index)
        features = [f for f in features if isinstance(row[f], (int, float, np.number))]

        # Predict probability (handling missing values cleanly)
        clean_row = row[features].fillna(0.0)
        if hasattr(self.model, "predict_proba"):
            X_df = pd.DataFrame([clean_row.to_dict()])
            prob_churn = float(self.model.predict_proba(X_df)[:, 1][0])
        elif hasattr(estimator, "predict_proba"):
            vals = np.nan_to_num(clean_row.values.astype(float)).reshape(1, -1)
            prob_churn = float(estimator.predict_proba(vals)[:, 1][0])
        else:
            prob_churn = 0.5

        # Compute local attribution contributions
        contributions = []
        if hasattr(estimator, "coef_"):
            coefs = estimator.coef_[0] if estimator.coef_.ndim == 2 else estimator.coef_
            for feat, coef in zip(features, coefs):
                val = float(row[feat]) if not np.isnan(row[feat]) else 0.0
                impact = val * float(coef)
                contributions.append({
                    "feature": feat,
                    "value": round(val, 2),
                    "impact": round(impact, 4),
                    "direction": "increases_churn_risk" if impact > 0 else "reduces_churn_risk",
                })
        else:
            # Fallback feature deviation from baseline
            for feat in features:
                val = float(row[feat]) if not np.isnan(row[feat]) else 0.0
                contributions.append({
                    "feature": feat,
                    "value": round(val, 2),
                    "impact": round(val, 4),
                    "direction": "observed_signal",
                })

        df_contrib = pd.DataFrame(contributions)
        df_contrib["abs_impact"] = df_contrib["impact"].abs()
        df_contrib = df_contrib.sort_values(by="abs_impact", ascending=False)

        risk_drivers = df_contrib[df_contrib["impact"] > 0].head(top_n).to_dict(orient="records")
        mitigators = df_contrib[df_contrib["impact"] < 0].head(top_n).to_dict(orient="records")

        return {
            "predicted_churn_probability": round(prob_churn, 4),
            "predicted_label": int(prob_churn >= 0.5),
            "top_risk_drivers": risk_drivers,
            "top_mitigating_factors": mitigators,
            "methodology": self.explainer_type,
            "causality_disclaimer": (
                "NOTE: These feature attributions reflect statistical associations and predictive "
                "weights within the model. They do NOT establish causal relationships."
            ),
        }

    def plot_summary(
        self,
        shap_values: Any,
        X_sample: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> None:
        """Saves a SHAP summary plot if matplotlib and SHAP are present; otherwise logs an informative note."""
        if not HAS_SHAP or shap_values is None:
            logger.info("SHAP summary plot skipped (SHAP not present or values empty). Use compute_global_importance().")
            return

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            shap.summary_plot(shap_values, X_sample, show=False)
            if save_path:
                plt.tight_layout()
                plt.savefig(save_path, dpi=150)
                logger.info(f"Saved SHAP summary plot to {save_path}.")
            plt.close()
        except Exception as e:
            logger.warning(f"Could not generate SHAP plot: {e}")
