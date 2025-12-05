FROM python:3.11-slim

# Sistem bağımlılıkları (MLflow Azure için)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev libssl-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# API kopyala
COPY api.py .

EXPOSE 8000

# Sağlam healthcheck (model load için 90s bekle)
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Uvicorn – Tek worker, uzun timeout (mobil yavaş bağlantı için)
CMD ["uvicorn", "api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "300", \
     "--log-level", "info"]