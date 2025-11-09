# Dockerfile

# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install necessary system packages (e.g., git, gcc, libgomp for some libraries)
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install FastAPI and Uvicorn
RUN pip install fastapi uvicorn[standard] 

# Install MLflow to load the model artifact
RUN pip install mlflow

# Copy the application code
COPY app /app/app
# Copy other necessary project files if needed for context (though not strictly for the API)
# COPY scripts /app/scripts 
# COPY data /app/data 

# Command to run the application using Uvicorn
# app.api:app refers to the 'app' object inside 'app/api.py'
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "80"]