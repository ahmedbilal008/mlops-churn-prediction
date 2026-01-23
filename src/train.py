import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
from schemas import CustomerData


# Configure MLflow
mlflow.set_tracking_uri('sqlite:///tracking/mlflow.db')
mlflow.set_experiment('Sentinel_Churn')


def load_and_validate_data(filepath):
    """Load data and validate using Pydantic."""
    df = pd.read_csv(filepath)
    
    # Convert TotalCharges to numeric (handles empty strings)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    print(f"Loaded {len(df)} records")
    
    # Validate sample of data
    validated_count = 0
    for _, row in df.head(100).iterrows():
        try:
            CustomerData(**row.to_dict())
            validated_count += 1
        except Exception as e:
            print(f"Validation warning: {e}")
    
    print(f"Validated {validated_count}/100 sample records")
    return df


def preprocess_data(df):
    """Encode categorical variables."""
    df_processed = df.copy()
    
    # Encode binary categorical variables
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
    
    # Encode multi-class categorical variables
    categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 
                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                       'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    # Encode target variable
    df_processed['Churn'] = df_processed['Churn'].map({'Yes': 1, 'No': 0})
    
    return df_processed


def train_model(X_train, y_train, n_estimators=100, max_depth=10):
    """Train Random Forest model."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    return metrics


def main():
    """Main training pipeline."""
    print("Starting training pipeline...")
    
    # Load and validate data
    df = load_and_validate_data('data/churn.csv')
    
    # Drop customerID column
    df = df.drop('customerID', axis=1)
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Split features and target
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Start MLflow run
    with mlflow.start_run(run_name="random_forest_churn"):
        # Hyperparameters
        n_estimators = 100
        max_depth = 10
        
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
        # Train model
        print("Training model...")
        model = train_model(X_train, y_train, n_estimators, max_depth)
        
        # Evaluate model
        print("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            print(f"{metric_name}: {metric_value:.4f}")
        
        # Save model locally
        model_path = Path('models/churn_model.pkl')
        model_path.parent.mkdir(exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        
        print("Training complete!")


if __name__ == "__main__":
    main()
