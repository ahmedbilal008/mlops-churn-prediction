import pytest
import joblib
from pathlib import Path
import json


def test_model_file_exists():
    """Test that model file exists after training."""
    model_path = Path('models/churn_model.pkl')
    
    # This test will pass only after training
    if model_path.exists():
        model = joblib.load(model_path)
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    else:
        pytest.skip("Model not trained yet. Run train.py first.")


def test_model_prediction_shape():
    """Test that model returns correct prediction shape."""
    model_path = Path('models/churn_model.pkl')
    
    if not model_path.exists():
        pytest.skip("Model not trained yet. Run train.py first.")
    
    model = joblib.load(model_path)
    
    # Create dummy input with correct number of features
    # Note: This is a placeholder - actual feature count depends on training
    n_features = model.n_features_in_
    dummy_input = [[0] * n_features]
    
    prediction = model.predict(dummy_input)
    assert prediction.shape == (1,)
    
    prediction_proba = model.predict_proba(dummy_input)
    assert prediction_proba.shape == (1, 2)
