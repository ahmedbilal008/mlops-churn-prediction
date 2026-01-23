import pytest
from pydantic import ValidationError
from src.schemas import CustomerData, PredictionRequest, PredictionResponse


def test_customer_data_valid():
    """Test valid customer data."""
    data = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'DSL',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'Yes',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'One year',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 49.99,
        'TotalCharges': 599.88
    }
    
    customer = CustomerData(**data)
    assert customer.MonthlyCharges == 49.99
    assert customer.tenure == 12


def test_customer_data_negative_monthly_charges():
    """Test that negative MonthlyCharges raises validation error."""
    data = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'DSL',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'Yes',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'One year',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': -10.0,
        'TotalCharges': 599.88
    }
    
    with pytest.raises(ValidationError):
        CustomerData(**data)


def test_customer_data_negative_total_charges():
    """Test that negative TotalCharges raises validation error."""
    data = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'DSL',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'Yes',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'One year',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 49.99,
        'TotalCharges': -100.0
    }
    
    with pytest.raises(ValidationError):
        CustomerData(**data)


def test_prediction_response():
    """Test prediction response schema."""
    response = PredictionResponse(
        churn_risk="HIGH",
        probability=0.85
    )
    
    assert response.churn_risk == "HIGH"
    assert response.probability == 0.85
