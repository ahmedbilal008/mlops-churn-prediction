"""Tests for MCP tool responses."""

import pytest
import json
from pathlib import Path


def _models_trained() -> bool:
    """Check if models are available for testing."""
    return (Path("models") / "best_model.pkl").exists()


class TestDatasetSummary:
    """Test dataset_summary tool."""

    def test_summary_returns_valid_json(self):
        if not Path("data/raw/churn.csv").exists():
            pytest.skip("Raw data not available.")

        from src.interfaces.mcp.server import get_dataset_summary

        result = json.loads(get_dataset_summary.fn())

        assert "total_rows" in result
        assert "churn_rate" in result
        assert result["total_rows"] > 0
        assert 0 < result["churn_rate"] < 1


class TestCompareModels:
    """Test compare_models tool."""

    def test_compare_returns_leaderboard(self):
        if not _models_trained():
            pytest.skip("Models not trained yet.")

        from src.interfaces.mcp.server import compare_models

        result = json.loads(compare_models.fn())

        assert "leaderboard" in result
        assert "best_model" in result
        assert len(result["leaderboard"]) > 0


class TestModelMetrics:
    """Test model_metrics tool."""

    def test_best_model_metrics(self):
        if not _models_trained():
            pytest.skip("Models not trained yet.")

        from src.interfaces.mcp.server import get_model_metrics

        result = json.loads(get_model_metrics.fn("best"))

        assert "accuracy" in result
        assert "f1_score" in result
        assert result["accuracy"] > 0


class TestPredictChurn:
    """Test predict_churn tool."""

    def test_prediction_returns_valid_json(self):
        if not _models_trained():
            pytest.skip("Models not trained yet.")

        from src.interfaces.mcp.server import predict_churn

        result = json.loads(
            predict_churn.fn(
                gender="Female",
                SeniorCitizen=0,
                Partner="No",
                Dependents="No",
                tenure=2,
                PhoneService="Yes",
                MultipleLines="No",
                InternetService="Fiber optic",
                OnlineSecurity="No",
                OnlineBackup="No",
                DeviceProtection="No",
                TechSupport="No",
                StreamingTV="No",
                StreamingMovies="No",
                Contract="Month-to-month",
                PaperlessBilling="Yes",
                PaymentMethod="Electronic check",
                MonthlyCharges=70.70,
                TotalCharges=151.65,
            )
        )

        assert "churn_probability" in result
        assert "risk_level" in result
        assert result["risk_level"] in ("LOW", "MEDIUM", "HIGH")
        assert 0 <= result["churn_probability"] <= 1

    def test_prediction_with_defaults_only(self):
        """Test that predict_churn works with no arguments (all defaults)."""
        if not _models_trained():
            pytest.skip("Models not trained yet.")

        from src.interfaces.mcp.server import predict_churn

        result = json.loads(predict_churn.fn())

        assert "churn_probability" in result
        assert "risk_level" in result
        assert 0 <= result["churn_probability"] <= 1

    def test_prediction_with_partial_features(self):
        """Test that predict_churn works with only a few key features."""
        if not _models_trained():
            pytest.skip("Models not trained yet.")

        from src.interfaces.mcp.server import predict_churn

        result = json.loads(
            predict_churn.fn(tenure=2, MonthlyCharges=80.0, Contract="Month-to-month")
        )

        assert "churn_probability" in result
        assert "risk_level" in result
        assert 0 <= result["churn_probability"] <= 1
