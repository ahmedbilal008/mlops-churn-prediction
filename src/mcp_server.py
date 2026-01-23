from fastmcp import FastMCP
import joblib
import json
from pathlib import Path
from schemas import PredictionResponse


# Initialize FastMCP server
mcp = FastMCP("Agentic Sentinel")

# Load trained model
MODEL_PATH = Path(__file__).parent.parent / "models" / "churn_model.pkl"


def load_model():
    """Load the trained model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please train the model first.")
    return joblib.load(MODEL_PATH)


@mcp.tool()
def predict_churn(customer_data: str) -> str:
    """
    Predict customer churn risk based on customer data.
    
    Args:
        customer_data: JSON string containing customer features
        
    Returns:
        JSON string with churn risk level and probability
    """
    try:
        # Load model
        model = load_model()
        
        # Parse input JSON
        data = json.loads(customer_data)
        
        # Extract features (must match training data structure)
        # In production, this would need proper preprocessing pipeline
        features = list(data.values())
        
        # Make prediction
        prediction_proba = model.predict_proba([features])[0]
        churn_probability = prediction_proba[1]
        
        # Determine risk level
        if churn_probability >= 0.7:
            risk_level = "HIGH"
        elif churn_probability >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Create response
        response = PredictionResponse(
            churn_risk=risk_level,
            probability=round(churn_probability, 4)
        )
        
        return response.model_dump_json()
        
    except FileNotFoundError as e:
        return json.dumps({"error": str(e)})
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON format"})
    except Exception as e:
        return json.dumps({"error": f"Prediction failed: {str(e)}"})


@mcp.tool()
def get_model_info() -> str:
    """
    Get information about the loaded model.
    
    Returns:
        JSON string with model metadata
    """
    try:
        if not MODEL_PATH.exists():
            return json.dumps({"error": "Model not found. Please train the model first."})
        
        model = load_model()
        
        info = {
            "model_type": type(model).__name__,
            "n_estimators": getattr(model, 'n_estimators', None),
            "max_depth": getattr(model, 'max_depth', None),
            "model_path": str(MODEL_PATH),
            "status": "ready"
        }
        
        return json.dumps(info, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to get model info: {str(e)}"})


if __name__ == "__main__":
    mcp.run()
