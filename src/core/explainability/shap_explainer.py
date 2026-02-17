"""
SHAP Explainability
===================

Provides model-agnostic feature explanations using SHAP.

SHAP (SHapley Additive exPlanations) is based on game theory.
It tells you exactly HOW MUCH each feature contributed to a
specific prediction.

KEY METHODS:
- TreeExplainer: O(TLD^2) — fast for tree models (RF, XGBoost)
- LinearExplainer: O(D) — fast for linear models (LogReg)
- KernelExplainer: O(2^D) — slow, model-agnostic (avoid in prod)

WE SAVE BACKGROUND DATA:
Some SHAP explainers need a "background" sample of training data
to compute expected values. We save 100 training samples during
training so SHAP works correctly during inference without needing
the full training set.
"""

import numpy as np
import shap
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config.settings import PROJECT_ROOT, get_paths_config
from src.utils.logger import get_logger


logger = get_logger(__name__)


class ShapExplainer:
    """SHAP-based model explainer for churn predictions."""

    def __init__(
        self,
        model: object,
        feature_names: list[str],
        background_data: np.ndarray | None = None,
    ) -> None:
        """Initialize SHAP explainer.

        Args:
            model: Trained sklearn-compatible model.
            feature_names: Names of preprocessed features.
            background_data: Sample of training data for non-tree explainers.
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = self._create_explainer(model, background_data)

    def _create_explainer(
        self, model: object, background_data: np.ndarray | None
    ) -> shap.Explainer:
        """Create the appropriate SHAP explainer based on model type."""
        try:
            from xgboost import XGBClassifier

            if isinstance(model, (RandomForestClassifier, XGBClassifier)):
                logger.info(
                    "Using SHAP TreeExplainer (fast path for tree models)"
                )
                return shap.TreeExplainer(model)
        except ImportError:
            pass

        if isinstance(model, LogisticRegression):
            if background_data is not None:
                logger.info("Using SHAP LinearExplainer for logistic regression")
                return shap.LinearExplainer(model, background_data)

        # Fallback to general Explainer
        if background_data is not None:
            logger.info("Using SHAP generic Explainer with background data")
            return shap.Explainer(model.predict_proba, background_data)

        # Last resort — TreeExplainer (will fail for non-tree models)
        logger.warning("Falling back to TreeExplainer")
        return shap.TreeExplainer(model)

    def explain_single(self, instance: np.ndarray) -> dict:
        """Explain a single prediction.

        Args:
            instance: 1D or 2D array of preprocessed features.

        Returns:
            Dictionary with SHAP values, base value, and driver analysis.
        """
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        shap_values = self.explainer.shap_values(instance)

        # Handle different SHAP output formats
        # For binary classification, we want the churn class (index 1)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]  # Class 1 (churn), first instance
        elif shap_values.ndim == 3:
            shap_vals = shap_values[0, :, 1]  # Class 1 (churn)
        else:
            shap_vals = shap_values[0]

        # Get base value
        if hasattr(self.explainer, "expected_value"):
            base_value = self.explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = float(base_value[1])  # Class 1
            else:
                base_value = float(base_value)
        else:
            base_value = 0.0

        # Create feature contribution dict
        contributions: dict[str, float] = {}
        for name, value in zip(self.feature_names, shap_vals):
            contributions[name] = float(value)

        # Sort by absolute contribution
        sorted_contributions = dict(
            sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        # Identify top drivers
        top_positive = [k for k, v in sorted_contributions.items() if v > 0][:5]
        top_negative = [k for k, v in sorted_contributions.items() if v < 0][:5]

        prediction_value = base_value + sum(shap_vals)

        return {
            "feature_contributions": [
                {"feature": k, "shap_value": round(v, 6)}
                for k, v in sorted_contributions.items()
            ],
            "base_value": round(base_value, 6),
            "prediction_value": round(float(prediction_value), 6),
            "top_positive_drivers": top_positive,
            "top_negative_drivers": top_negative,
        }

    def global_importance(self, X: np.ndarray) -> dict[str, float]:
        """Compute global SHAP feature importance.

        Args:
            X: 2D array of preprocessed test/validation data.

        Returns:
            Dictionary of feature name → mean absolute SHAP value.
        """
        shap_values = self.explainer.shap_values(X)

        if isinstance(shap_values, list):
            vals = shap_values[1]  # Class 1
        elif shap_values.ndim == 3:
            vals = shap_values[:, :, 1]
        else:
            vals = shap_values

        mean_abs = np.abs(vals).mean(axis=0)
        importance = dict(zip(self.feature_names, mean_abs.tolist()))
        importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )

        return importance

    def save_global_plot(
        self, X: np.ndarray, path: str | Path | None = None
    ) -> Path:
        """Generate and save global SHAP summary plot."""
        if path is None:
            paths_config = get_paths_config()
            path = PROJECT_ROOT / paths_config["plots_dir"] / "shap_summary.png"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        shap_values = self.explainer.shap_values(X)

        if isinstance(shap_values, list):
            vals = shap_values[1]
        elif shap_values.ndim == 3:
            vals = shap_values[:, :, 1]
        else:
            vals = shap_values

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            vals, X, feature_names=self.feature_names, show=False
        )
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"SHAP summary plot saved to {path}")
        return path


# ---------------------------------------------------------------------------
# Background data persistence (for inference-time SHAP)
# ---------------------------------------------------------------------------


def save_background_data(
    X_sample: np.ndarray, path: str | Path | None = None
) -> Path:
    """Save background data sample for SHAP explanations during inference."""
    if path is None:
        paths_config = get_paths_config()
        path = PROJECT_ROOT / paths_config["models_dir"] / "shap_background.pkl"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(X_sample, path)
    logger.info(
        f"SHAP background data saved to {path} ({X_sample.shape[0]} samples)"
    )
    return path


def load_background_data(path: str | Path | None = None) -> np.ndarray:
    """Load saved background data for SHAP."""
    if path is None:
        paths_config = get_paths_config()
        path = PROJECT_ROOT / paths_config["models_dir"] / "shap_background.pkl"

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SHAP background data not found: {path}")

    return joblib.load(path)
