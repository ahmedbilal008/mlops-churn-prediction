"""Tests for model training, artifacts, and feature engineering."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


class TestModelArtifacts:
    """Test that model artifacts exist and are valid after training."""

    MODELS_DIR = Path("models")

    def test_best_model_exists(self):
        path = self.MODELS_DIR / "best_model.pkl"
        if not path.exists():
            pytest.skip(
                "Models not trained yet. Run: "
                "uv run python -m src.pipelines.training_pipeline --stage full"
            )

        import joblib

        model = joblib.load(path)
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_preprocessor_exists(self):
        path = self.MODELS_DIR / "preprocessor.pkl"
        if not path.exists():
            pytest.skip("Preprocessor not saved yet. Run training pipeline.")

        from src.core.data.preprocessor import ChurnPreprocessor

        preprocessor = ChurnPreprocessor.load()
        assert preprocessor.feature_names is not None
        assert len(preprocessor.feature_names) > 0

    def test_model_prediction_shape(self):
        path = self.MODELS_DIR / "best_model.pkl"
        if not path.exists():
            pytest.skip("Models not trained yet.")

        import joblib

        model = joblib.load(path)
        n_features = model.n_features_in_
        dummy = np.zeros((1, n_features))

        pred = model.predict(dummy)
        assert pred.shape == (1,)

        proba = model.predict_proba(dummy)
        assert proba.shape == (1, 2)

    def test_metrics_file_exists(self):
        path = self.MODELS_DIR / "training_metrics.json"
        if not path.exists():
            pytest.skip("Metrics not saved yet. Run training pipeline.")

        import json

        with open(path) as f:
            metrics = json.load(f)

        assert "best_model" in metrics
        assert "models" in metrics


class TestFeatureEngineering:
    """Test feature engineering produces expected columns."""

    SAMPLE_ROW = {
        "customerID": "TEST001",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 49.99,
        "TotalCharges": 599.88,
        "Churn": "No",
    }

    def test_engineer_features_columns(self):
        from src.core.features.engineer import engineer_features

        df = pd.DataFrame([self.SAMPLE_ROW])
        result = engineer_features(df)

        expected_cols = [
            "tenure_group",
            "avg_charges_per_month",
            "charges_ratio",
            "num_services",
            "has_internet",
            "senior_high_charges",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_tenure_group_label(self):
        from src.core.features.engineer import engineer_features

        df = pd.DataFrame([self.SAMPLE_ROW])
        result = engineer_features(df)
        assert result["tenure_group"].iloc[0] == "0-12"

    def test_num_services_count(self):
        from src.core.features.engineer import engineer_features

        df = pd.DataFrame([self.SAMPLE_ROW])
        result = engineer_features(df)
        # PhoneService=Yes, InternetService=DSL, OnlineSecurity=Yes,
        # OnlineBackup=Yes, TechSupport=Yes = 5 services
        assert result["num_services"].iloc[0] == 5

    def test_single_row_inference(self):
        """Feature engineering must work on single rows (for inference)."""
        from src.core.features.engineer import engineer_features

        df = pd.DataFrame([self.SAMPLE_ROW])
        result = engineer_features(df)
        assert len(result) == 1
        assert result["charges_ratio"].iloc[0] > 0
