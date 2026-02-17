"""
Data Preprocessing Pipeline
============================

Handles encoding and scaling with a saved sklearn pipeline.

CRITICAL CONCEPT — Training vs Inference Consistency:
The preprocessor is FIT on training data and SAVED. During inference,
the same fitted preprocessor TRANSFORMS new data. This ensures:
1. Same encoding categories (no "unknown category" errors)
2. Same scaling parameters (mean/std from training data)
3. Same feature ordering

If you refit the preprocessor on new data, the feature values will
be on a different scale → model predictions will be wrong.

WHY ColumnTransformer:
- Applies different transforms to different column types
- Numerical → StandardScaler (zero mean, unit variance)
- Categorical → OneHotEncoder (binary indicators)
- Keeps everything in a single serializable pipeline

INTERVIEW INSIGHT: "How do you prevent train/inference skew?"
Answer: "We serialize the fitted preprocessor alongside the model.
Both training and inference use the same pipeline, ensuring
identical feature transformations."
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config.settings import PROJECT_ROOT, get_feature_config, get_paths_config
from src.utils.logger import get_logger


logger = get_logger(__name__)


class ChurnPreprocessor:
    """Handles preprocessing for both training and inference.

    Ensures consistency between training and inference transforms,
    preventing data leakage and train/serve skew.
    """

    def __init__(self) -> None:
        self.feature_config = get_feature_config()
        self.pipeline: ColumnTransformer | None = None
        self.feature_names: list[str] | None = None
        self._is_fitted = False

    def _build_pipeline(self) -> ColumnTransformer:
        """Build sklearn preprocessing pipeline."""
        numerical_cols = self.feature_config["numerical_columns"]
        categorical_cols = self.feature_config["categorical_columns"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_cols),
                (
                    "cat",
                    OneHotEncoder(
                        drop="first",
                        sparse_output=False,
                        handle_unknown="ignore",
                    ),
                    categorical_cols,
                ),
            ],
            remainder="drop",  # Drop columns not in either list
        )

        return preprocessor

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit preprocessor on training data and transform it.

        IMPORTANT: Only call this on the TRAINING set.
        """
        self.pipeline = self._build_pipeline()

        X = self.pipeline.fit_transform(df)
        self._extract_feature_names()
        self._is_fitted = True

        logger.info(
            f"Preprocessor fitted: {df.shape[1]} input features → {X.shape[1]} output features"
        )
        return X

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor.

        Use this for test data and inference.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Preprocessor not fitted. Call fit_transform() first "
                "or load a saved preprocessor."
            )
        return self.pipeline.transform(df)

    def _extract_feature_names(self) -> None:
        """Extract feature names after fitting."""
        names: list[str] = []

        # Numerical feature names stay the same
        names.extend(self.feature_config["numerical_columns"])

        # Get one-hot encoded names
        cat_encoder = self.pipeline.named_transformers_["cat"]
        cat_names = cat_encoder.get_feature_names_out(
            self.feature_config["categorical_columns"]
        )
        names.extend(cat_names.tolist())

        self.feature_names = names

    def get_feature_names(self) -> list[str]:
        """Return feature names after preprocessing."""
        if self.feature_names is None:
            raise RuntimeError(
                "Feature names not available. Fit the preprocessor first."
            )
        return self.feature_names

    def save(self, path: str | Path | None = None) -> Path:
        """Save fitted preprocessor to disk."""
        if path is None:
            paths_config = get_paths_config()
            path = PROJECT_ROOT / paths_config["models_dir"] / "preprocessor.pkl"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "pipeline": self.pipeline,
                "feature_names": self.feature_names,
                "feature_config": self.feature_config,
            },
            path,
        )

        logger.info(f"Preprocessor saved to {path}")
        return path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ChurnPreprocessor":
        """Load a saved preprocessor."""
        if path is None:
            paths_config = get_paths_config()
            path = PROJECT_ROOT / paths_config["models_dir"] / "preprocessor.pkl"

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {path}")

        data = joblib.load(path)

        preprocessor = cls()
        preprocessor.pipeline = data["pipeline"]
        preprocessor.feature_names = data["feature_names"]
        preprocessor.feature_config = data["feature_config"]
        preprocessor._is_fitted = True

        logger.info(f"Preprocessor loaded from {path}")
        return preprocessor
