# Dockerfile (API için - FastAPI/Uvicorn)

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install ALL dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# CRITICAL: Ensure Azure MLflow support is installed
RUN pip install --no-cache-dir \
    mlflow[azure] \
    azure-storage-blob \
    azure-identity \
    azure-core \
    azure-mgmt-containerinstance \
    adlfs

# Verify installations
RUN python -c "import mlflow; import azure.storage.blob; import azure.identity; print('✅ All dependencies OK')"

# Copy application code
COPY . /app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run with Uvicorn
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]