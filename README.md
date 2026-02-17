# Agentic Churn Intelligence Platform

## AI-Agent-Ready MLOps System for Customer Churn Prediction

A portfolio-grade MLOps system that combines multi-model machine learning, SHAP explainability, and the **Model Context Protocol (MCP)** to create a fully agentic churn prediction service. Built with production-grade practices: experiment tracking (MLflow), data versioning (DVC), reproducible pipelines, and comprehensive testing.

An AI agent (Gemini, Claude, etc.) connects via MCP and can predict churn, explain predictions, compare models, inspect data, and trigger retraining — all through structured tool calls.

---

## Key Features

- **7 MCP Tools** — predict, explain, compare, inspect, retrain, and add data — all callable by AI agents over SSE/stdio
- **3 Models** — Logistic Regression, Random Forest, XGBoost — trained, compared, and auto-selected by F1 score
- **SHAP Explainability** — per-prediction feature contributions and global importance plots via TreeExplainer / LinearExplainer
- **MLflow Experiment Tracking** — every run logs params, metrics, confusion matrices, ROC curves, feature importance plots
- **DVC Pipeline** — 4-stage reproducible pipeline (ingest → features → train → evaluate) with `params.yaml`-driven config
- **Pydantic Validation** — type-safe schemas at every boundary (data ingestion, API contracts, responses)
- **Class Imbalance Handling** — `class_weight='balanced'` (sklearn) and `scale_pos_weight=2.76` (XGBoost) for the 73/27 split
- **Modular Architecture** — clean separation: config → data → features → models → explainability → pipelines → MCP

---

## Quick Start

```bash
# 1. Install dependencies (requires Python 3.12 and uv)
uv sync

# 2. Train all models (full pipeline)
uv run python main.py train

# 3. Start MCP server (SSE transport for Gemini / AI agents)
uv run python main.py serve

# 4. Run tests
uv run pytest tests/ -v
```

### CLI Options

```bash
# Train with full pipeline (ingest → features → train → evaluate)
uv run python main.py train

# Start MCP server with SSE transport (default, port 8000)
uv run python main.py serve

# Start MCP server with stdio transport
uv run python main.py serve --transport stdio

# Run individual pipeline stages (used by DVC)
uv run python -m src.pipelines.training_pipeline --stage ingest
uv run python -m src.pipelines.training_pipeline --stage feature_engineering
uv run python -m src.pipelines.training_pipeline --stage train
uv run python -m src.pipelines.training_pipeline --stage evaluate
uv run python -m src.pipelines.training_pipeline --stage full
```

---

## MCP Tools

The system exposes 7 tools via the Model Context Protocol:

| Tool | Description |
|------|-------------|
| `predict_churn` | Predict churn probability for a customer (19 input features) — returns probability, risk level, top SHAP drivers |
| `explain_prediction` | Full SHAP explanation for a single customer — all feature contributions + base value |
| `model_metrics` | Get performance metrics for any model (or "best") — accuracy, precision, recall, F1, AUC |
| `compare_models` | Leaderboard of all trained models from MLflow — sorted by F1 score |
| `dataset_summary` | Dataset statistics — row counts, churn rate, feature distributions |
| `retrain_model` | Trigger full pipeline retraining — clears model cache, retrains all models |
| `add_customer_record` | Append a new validated customer record to the dataset |

### Connecting an AI Agent

**Gemini (SSE transport):**
```bash
uv run python main.py serve
# Server starts at http://localhost:8000/sse
# Point your Gemini MCP client to this endpoint
```

**Claude Desktop (stdio transport):**
```json
{
  "mcpServers": {
    "churn-intelligence": {
      "command": "uv",
      "args": ["run", "python", "main.py", "serve", "--transport", "stdio"],
      "cwd": "/path/to/MCP_ops"
    }
  }
}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    AI Agent (Gemini / Claude)            │
│                         ↕ MCP (SSE/stdio)               │
├─────────────────────────────────────────────────────────┤
│                    src/mcp/server.py                     │
│              7 MCP Tools + Model Cache                   │
├──────────┬──────────┬───────────┬───────────────────────┤
│ schemas/ │ models/  │ explain/  │ pipelines/            │
│ Pydantic │ trainer  │ SHAP      │ training_pipeline     │
│ models   │ registry │ explainer │ (4-stage orchestrator) │
├──────────┴──────────┴───────────┴───────────────────────┤
│   data/loader    data/validator    data/preprocessor     │
│   features/engineer    config/settings    utils/logger   │
├─────────────────────────────────────────────────────────┤
│  MLflow (tracking/mlflow.db)  │  DVC (dvc.yaml)         │
│  Experiment runs & artifacts  │  Reproducible pipeline   │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Ingest** — Load raw CSV → clean (fix types, fill NaN, dedup) → validate with Pydantic (sample) → save cleaned CSV
2. **Feature Engineering** — Add 6 derived features (tenure_group, avg_charges_per_month, charges_ratio, num_services, has_internet, senior_high_charges)
3. **Train** — Stratified 80/20 split → fit ColumnTransformer (StandardScaler + OneHotEncoder) → train 3 models → log to MLflow → select best by F1
4. **Evaluate** — Load best model → compute SHAP global importance → save summary plot + evaluation JSON

---

## Project Structure

```
├── params.yaml                  # Centralized configuration (DVC params)
├── dvc.yaml                     # DVC 4-stage pipeline definition
├── main.py                      # CLI entry point (train / serve)
├── pyproject.toml               # Dependencies and project config
│
├── src/
│   ├── config/
│   │   └── settings.py          # Load params.yaml, typed accessors
│   ├── data/
│   │   ├── loader.py            # Raw data ingestion & cleaning
│   │   ├── validator.py         # Pydantic-based row validation
│   │   └── preprocessor.py      # ColumnTransformer (scaler + encoder)
│   ├── features/
│   │   └── engineer.py          # 6 derived features (row-level, pre-split safe)
│   ├── models/
│   │   ├── trainer.py           # Model factory, training, MLflow logging
│   │   └── registry.py          # Best model selection, leaderboard
│   ├── explainability/
│   │   └── shap_explainer.py    # SHAP TreeExplainer / LinearExplainer
│   ├── pipelines/
│   │   └── training_pipeline.py # 4-stage orchestrator + CLI
│   ├── mcp/
│   │   └── server.py            # 7 MCP tools + FastMCP server
│   ├── schemas/
│   │   └── models.py            # Pydantic models (9 schemas)
│   └── utils/
│       └── logger.py            # Structured logging setup
│
├── tests/
│   ├── test_schemas.py          # Schema validation tests
│   ├── test_model.py            # Model artifact + feature engineering tests
│   └── test_mcp_tools.py        # MCP tool integration tests
│
├── data/
│   ├── raw/churn.csv            # Original Telco Customer Churn dataset
│   └── processed/               # Pipeline outputs (cleaned.csv, featured.csv, splits)
│
├── models/                      # Trained model artifacts
│   ├── best_model.pkl           # Auto-selected best model
│   ├── preprocessor.pkl         # Fitted ColumnTransformer
│   ├── shap_background.pkl      # SHAP background data (100 samples)
│   ├── training_metrics.json    # All model metrics
│   ├── evaluation.json          # Best model evaluation
│   └── plots/                   # Confusion matrices, ROC curves, SHAP summary
│
├── tracking/
│   └── mlflow.db                # MLflow SQLite backend
│
└── mlruns/                      # MLflow artifact store
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Runtime | Python 3.12 + uv | Fast dependency management, reproducible env |
| ML Models | scikit-learn, XGBoost | Logistic Regression, Random Forest, XGBoost |
| Explainability | SHAP | Per-prediction and global feature importance |
| Agent Protocol | FastMCP | MCP server with SSE/stdio/streamable-http transport |
| Experiment Tracking | MLflow | Params, metrics, artifacts, model registry |
| Data Versioning | DVC | Reproducible pipeline stages, data tracking |
| Validation | Pydantic | Type-safe schemas at every data boundary |
| Testing | pytest | Schema, model artifact, and MCP integration tests |

---

## Model Performance

Results from the latest training run on Telco Customer Churn (7,043 records, 26.5% churn rate):

| Model | F1 Score | AUC-ROC | Precision | Recall |
|-------|----------|---------|-----------|--------|
| Logistic Regression | 0.615 | 0.847 | — | — |
| **Random Forest** (best) | **0.622** | **0.843** | **0.544** | **0.725** |
| XGBoost | 0.611 | 0.830 | — | — |

Best model auto-selected by F1 score. All models handle class imbalance via balanced class weights.

---

## DVC Pipeline

Reproduce the full pipeline:

```bash
# Run all stages
dvc repro

# Run a single stage
dvc repro train

# View pipeline DAG
dvc dag
```

Pipeline stages and their dependencies are defined in `dvc.yaml`, parameterized by `params.yaml`.

---

## Viewing Experiments

Launch the MLflow UI to inspect all training runs:

```bash
uv run mlflow ui --backend-store-uri sqlite:///tracking/mlflow.db
```

Access at `http://localhost:5000`. Each run includes logged parameters, metrics, confusion matrix plots, ROC curves, and feature importance charts.

---

## License

MIT
