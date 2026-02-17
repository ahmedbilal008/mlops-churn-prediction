"""Tests for Pydantic validation schemas."""

import pytest
from pydantic import ValidationError

from src.schemas.models import (
    CustomerRecord,
    CustomerFeatures,
    PredictionResponse,
)


class TestCustomerRecord:
    """Test raw data validation."""

    VALID_CUSTOMER = {
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
    }

    def test_valid_record(self):
        record = CustomerRecord(**self.VALID_CUSTOMER)
        assert record.MonthlyCharges == 49.99
        assert record.tenure == 12

    def test_negative_monthly_charges(self):
        data = {**self.VALID_CUSTOMER, "MonthlyCharges": -10.0}
        with pytest.raises(ValidationError):
            CustomerRecord(**data)

    def test_negative_total_charges(self):
        data = {**self.VALID_CUSTOMER, "TotalCharges": -100.0}
        with pytest.raises(ValidationError):
            CustomerRecord(**data)

    def test_invalid_gender(self):
        data = {**self.VALID_CUSTOMER, "gender": "Other"}
        with pytest.raises(ValidationError):
            CustomerRecord(**data)

    def test_invalid_contract(self):
        data = {**self.VALID_CUSTOMER, "Contract": "Weekly"}
        with pytest.raises(ValidationError):
            CustomerRecord(**data)

    def test_senior_citizen_boundary(self):
        data = {**self.VALID_CUSTOMER, "SeniorCitizen": 2}
        with pytest.raises(ValidationError):
            CustomerRecord(**data)

    def test_optional_churn_field(self):
        """Churn field is optional (not present during inference)."""
        record = CustomerRecord(**self.VALID_CUSTOMER)
        assert record.Churn is None

        data_with_churn = {**self.VALID_CUSTOMER, "Churn": "Yes"}
        record = CustomerRecord(**data_with_churn)
        assert record.Churn == "Yes"


class TestCustomerFeatures:
    """Test inference input validation."""

    VALID_FEATURES = {
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.70,
        "TotalCharges": 151.65,
    }

    def test_valid_features(self):
        features = CustomerFeatures(**self.VALID_FEATURES)
        assert features.MonthlyCharges == 70.70

    def test_zero_tenure(self):
        """New customers can have 0 tenure."""
        data = {**self.VALID_FEATURES, "tenure": 0, "TotalCharges": 0}
        features = CustomerFeatures(**data)
        assert features.tenure == 0


class TestPredictionResponse:
    """Test prediction output validation."""

    def test_valid_response(self):
        response = PredictionResponse(
            churn_probability=0.85,
            risk_level="HIGH",
            top_drivers=[{"feature": "tenure", "importance": 0.3}],
            model_used="random_forest",
        )
        assert response.risk_level == "HIGH"
        assert response.churn_probability == 0.85
