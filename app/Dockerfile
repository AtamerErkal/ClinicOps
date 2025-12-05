FROM python:3.11-slim

# Sistem bağımlılıkları (Plotly ve Streamlit için)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements önce (cache için)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Frontend'i kopyala
COPY app/frontend_v3.py .

# Streamlit config – Mobil için port ve headless
RUN mkdir -p /root/.streamlit && \
    echo '[server]\n\
port = 7860\n\
address = "0.0.0.0"\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
fileWatcherType = "none"\n\
enableStaticServing = false\n\
\n\
[browser]\n\
serverAddress = "0.0.0.0"\n\
gatherUsageStats = false\n\
\n\
[theme]\n\
primaryColor = "#667eea"\n\
backgroundColor = "#ffffff"\n\
secondaryBackgroundColor = "#f0f2f6"\n\
textColor = "#262730"' > /root/.streamlit/config.toml

EXPOSE 7860

# Mobil için uzun start-period (Streamlit yavaş yüklenir)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "frontend_v3.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false"]