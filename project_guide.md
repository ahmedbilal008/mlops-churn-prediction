# 🏗️ Project Blueprint: Agentic Sentinel (Local Edition)

**Tagline:** A `uv`-managed MLOps System integrating DVC, MLflow, and Claude Desktop.

### **1. System Architecture (Local)**

* **Environment:** Managed by `uv`. No global Python mess.
* **Data:** Versioned locally by **DVC** (stored in a local folder, not S3).
* **Tracking:** **MLflow** runs locally, saving metrics to a `mlflow.db` file.
* **Interface:** **Claude Desktop** connects to your local running process.

### **2. Directory Structure**

*Note: No Dockerfile. Added `pyproject.toml` and `uv.lock`.*

```text
agentic_sentinel/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD: Runs tests using uv
├── .dvc/                       # DVC config (Auto-generated)
├── data/                       # Local folder for raw data
│   └── churn.csv.dvc           # DVC pointer file
├── mlruns/                     # MLflow local logs (Auto-generated)
├── models/                     # Stores the trained model artifact
│   └── churn_model.pkl         # The "brain"
├── src/
│   ├── __init__.py
│   ├── schemas.py              # Pydantic models (Shared Contract)
│   ├── train.py                # Training script + MLflow logging
│   └── mcp_server.py           # The MCP Server
├── tests/
│   ├── __init__.py
│   ├── test_schemas.py
│   └── test_model.py
├── pyproject.toml              # uv dependency definition
└── uv.lock                     # uv lockfile (Reproducibility)

```

---

### **3. Detailed Agent Instructions (Copy-Paste this to your Agent)**

**Role:** You are a Senior MLOps Engineer.
**Task:** Scaffold and build the "Agentic Sentinel" project.
**Constraints:** Do NOT use Docker. Use `uv` for dependency management. Configure DVC and MLflow for strictly local use.

#### **Phase 1: Project & Environment Setup (The `uv` Way)**

*Command to Agent:*

1. "Initialize a new directory `agentic_sentinel`."
2. "Run `uv init` to create the project structure."
3. "Add the following dependencies using `uv add`: `pandas`, `scikit-learn`, `mlflow`, `pydantic`, `fastmcp`, `joblib`, `pytest`."
4. "Add `uv add --dev pytest` for testing."

#### **Phase 2: Local DVC & MLflow Configuration**

*Command to Agent:*
"You must set up DVC and MLflow for local usage. Do not assume cloud storage."

1. **DVC Setup:**
* "Initialize DVC: `dvc init`."
* "Configure a local remote storage (simulating S3): `dvc remote add -d localstorage /tmp/dvc-storage`."
* "Download the Telco Churn dataset and save it to `data/churn.csv`."
* "Track it: `dvc add data/churn.csv`."


2. **MLflow Setup:**
* "In `src/train.py`, configure MLflow to use a local SQLite database."
* "Add this line at the start of `train.py`: `mlflow.set_tracking_uri('sqlite:///mlflow.db')`."
* "Ensure `mlflow.set_experiment('Sentinel_Churn')` is set."



#### **Phase 3: The Code Logic**

*Command to Agent:*

1. **`src/schemas.py`:** "Create a Pydantic model `CustomerData` that validates `MonthlyCharges` is non-negative."
2. **`src/train.py`:**
* "Load data using pandas."
* "Validate rows using `CustomerData`."
* "Train a Random Forest."
* "Log parameters (`n_estimators`) and metrics (`accuracy`) to MLflow."
* "Save the model locally to `models/churn_model.pkl`."


3. **`src/mcp_server.py`:**
* "Use `FastMCP` to create a server."
* "Load `models/churn_model.pkl`."
* "Create a tool `predict_churn` that takes a JSON string, validates it via Pydantic, and returns the risk level."



#### **Phase 4: CI/CD (GitHub Actions with `uv`)**

*Command to Agent:*
"Create `.github/workflows/ci.yml`. It must use `uv` to install dependencies."

* *Snippet for Agent:*
```yaml
steps:
  - uses: actions/checkout@v4
  - name: Install uv
    uses: astral-sh/setup-uv@v1
  - name: Install dependencies
    run: uv sync
  - name: Run tests
    run: uv run pytest tests/

```



---

### **4. Claude Desktop Configuration**

Once the agent finishes coding, you (the human) must add this to your config file. Note that we use `uv run` to execute the server, which handles the environment automatically.

**File:** `claude_desktop_config.json`

```json
{
  "mcpServers": {
    "sentinel": {
      "command": "uv",
      "args": [
        "run",
        "src/mcp_server.py"
      ]
    }
  }
}

```
