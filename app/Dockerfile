FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from parent directory (root)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend from current directory (app/)
COPY app/frontend_v3.py .

# Streamlit config
RUN mkdir -p /root/.streamlit && \
    echo '[server]\n\
port = 7860\n\
address = "0.0.0.0"\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
fileWatcherType = "none"\n\
\n\
[browser]\n\
serverAddress = "0.0.0.0"\n\
gatherUsageStats = false' > /root/.streamlit/config.toml

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "frontend_v3.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]