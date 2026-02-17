"""
Configuration Management
========================

Two-layer config system:

1. params.yaml — ML hyperparameters (DVC-tracked, reproducible experiments)
2. Environment variables — deployment config (port, paths, URIs)

The 12-Factor App methodology says: "Store config in the environment."
This means the same Docker image works in dev, staging, and production
with just different env vars. ML hyperparameters stay in params.yaml
because DVC needs them for pipeline reproducibility.

SUPPORTED ENVIRONMENT VARIABLES:
    MCP_PORT             — MCP server port (default: 8000)
    MCP_HOST             — MCP server host (default: 0.0.0.0)
    MODEL_PATH           — Override path to best model pickle
    MLFLOW_TRACKING_URI  — MLflow backend URI (default: sqlite:///tracking/mlflow.db)
    DVC_REMOTE_PATH      — DVC remote storage path
    LOG_LEVEL            — Logging level (default: INFO)
"""

import os
import yaml
from pathlib import Path
from typing import Any


# Project root is 3 levels up from src/config/settings.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_params(params_path: str | Path | None = None) -> dict[str, Any]:
    """Load parameters from params.yaml.

    Args:
        params_path: Override path to params file. Defaults to PROJECT_ROOT/params.yaml.

    Returns:
        Dictionary of all configuration parameters.
    """
    if params_path is None:
        params_path = PROJECT_ROOT / "params.yaml"
    else:
        params_path = Path(params_path)

    if not params_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {params_path}")

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    return params


# ---------------------------------------------------------------------------
# Environment-aware getters  (env var → params.yaml fallback)
# ---------------------------------------------------------------------------


def get_data_config() -> dict[str, Any]:
    """Get data-related configuration."""
    return load_params()["data"]


def get_feature_config() -> dict[str, Any]:
    """Get feature engineering configuration."""
    return load_params()["features"]


def get_model_config(model_type: str | None = None) -> dict[str, Any]:
    """Get model configuration.

    Args:
        model_type: Specific model (logistic_regression, random_forest, xgboost).
                    If None, returns entire models section.
    """
    params = load_params()
    models_config = params["models"]

    if model_type is None:
        return models_config

    if model_type not in models_config and model_type != "default_model":
        available = [k for k in models_config.keys() if k != "default_model"]
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")

    return models_config[model_type]


def get_mlflow_config() -> dict[str, Any]:
    """Get MLflow configuration — env vars override params.yaml."""
    config = load_params()["mlflow"]

    # MLFLOW_TRACKING_URI overrides the yaml value
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        config["tracking_uri"] = env_uri

    return config


def get_paths_config() -> dict[str, Any]:
    """Get file paths configuration — MODEL_PATH overrides best model location."""
    config = load_params()["paths"]

    env_model = os.environ.get("MODEL_PATH")
    if env_model:
        config["model_override"] = env_model

    return config


def get_server_config() -> dict[str, Any]:
    """Get MCP server deployment config from environment variables.

    These are deployment-specific and NOT in params.yaml,
    following the 12-Factor App principle.
    """
    return {
        "host": os.environ.get("MCP_HOST", "0.0.0.0"),
        "port": int(os.environ.get("MCP_PORT", "8000")),
        "log_level": os.environ.get("LOG_LEVEL", "INFO"),
    }
