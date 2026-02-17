# =============================================================================
# Dockerfile — Churn Intelligence MCP Server
# =============================================================================
# Build:  docker build -t churn-mcp .
# Run:    docker run -p 8000:8000 churn-mcp
#
# Environment variables:
#   MCP_PORT             (default: 8000)
#   MCP_HOST             (default: 0.0.0.0)
#   MLFLOW_TRACKING_URI  (default: sqlite:///tracking/mlflow.db)
#   LOG_LEVEL            (default: INFO)
# =============================================================================

FROM python:3.12-slim

# Install uv — fast Python package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files first (Docker layer caching)
# This means deps are re-installed ONLY when pyproject.toml changes,
# not on every code change.
COPY pyproject.toml uv.lock ./

# Install dependencies into the system Python (no venv needed in container)
RUN uv sync --frozen --no-dev

# Copy application code
COPY params.yaml ./
COPY src/ ./src/
COPY data/raw/ ./data/raw/

# Create directories for generated artifacts
RUN mkdir -p models/plots tracking data/processed

# Expose MCP server port
EXPOSE 8000

# Health check — verify Python and imports work
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "from src.interfaces.mcp.server import mcp; print('ok')" || exit 1

# Start MCP server with SSE transport
CMD ["uv", "run", "python", "-m", "src.interfaces.mcp.server", "--transport", "sse"]
