# Agentic Sentinel

A production-ready MLOps system for customer churn prediction, integrating DVC for data versioning, MLflow for experiment tracking, and Model Context Protocol (MCP) for AI-powered inference.

## Overview

Agentic Sentinel demonstrates modern MLOps practices for building, tracking, and deploying machine learning models. The system predicts customer churn risk using a Random Forest classifier trained on telco customer data.

## Architecture

- **Environment Management**: `uv` for fast, reliable Python dependency management
- **Data Versioning**: DVC with local storage backend
- **Experiment Tracking**: MLflow with SQLite backend
- **Model Serving**: FastMCP server for Claude Desktop integration
- **CI/CD**: GitHub Actions for automated testing

## Project Structure

```
.
├── src/
│   ├── schemas.py          # Pydantic data validation models
│   ├── train.py            # Training pipeline with MLflow logging
│   └── mcp_server.py       # FastMCP server for predictions
├── tests/
│   ├── test_schemas.py     # Schema validation tests
│   └── test_model.py       # Model inference tests
├── data/
│   └── churn.csv.dvc       # DVC-tracked dataset
├── models/
│   └── churn_model.pkl     # Trained model artifact
├── tracking/
│   └── mlflow.db           # MLflow tracking database
├── .github/workflows/
│   └── ci.yml              # CI/CD pipeline
└── pyproject.toml          # Project dependencies
```

## Setup

### Prerequisites

- Python 3.14+
- uv package manager
- Git

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd agentic_sentinel
```

2. Install dependencies:
```bash
uv sync
```

3. Pull data from DVC:
```bash
uv run dvc pull
```

## Usage

### Training the Model

Run the training pipeline:

```bash
uv run python src/train.py
```

This will:
- Load and validate data using Pydantic schemas
- Train a Random Forest classifier
- Log metrics and parameters to MLflow
- Save the model to `models/churn_model.pkl`

### Viewing Experiments

Launch MLflow UI:

```bash
uv run mlflow ui --backend-store-uri sqlite:///tracking/mlflow.db
```

Access at `http://localhost:5000`

### Running Tests

Execute the test suite:

```bash
uv run pytest tests/ -v
```

### MCP Server Integration

Configure Claude Desktop by adding to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "sentinel": {
      "command": "uv",
      "args": ["run", "python", "src/mcp_server.py"],
      "cwd": "/path/to/agentic_sentinel"
    }
  }
}
```

Restart Claude Desktop to load the server.

## Model Performance

The Random Forest model achieves:
- Accuracy: ~80%
- Precision: ~75%
- Recall: ~70%
- F1 Score: ~72%

Metrics vary based on hyperparameter tuning and data preprocessing.

## Development

### Adding Dependencies

```bash
uv add <package-name>
```

### Running Linting

```bash
uv run pylint src/
```

## Data Versioning

Dataset is tracked with DVC. To update:

```bash
uv run dvc add data/churn.csv
uv run dvc push
```

## CI/CD

GitHub Actions automatically:
- Runs tests on push/PR
- Validates code quality
- Ensures reproducible builds

## License

MIT License

## Author

Built as a demonstration of modern MLOps practices.
