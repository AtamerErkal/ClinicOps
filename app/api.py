# app/api.py - FINAL AUTH FIX

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from azure.storage.blob import BlobClient 
import logging
import os
from pydantic import BaseModel
import numpy as np 

# --- Setup ---
logging.basicConfig(level=logging.INFO)

# Get environment variables passed from ACI
AZURE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT") 
AZURE_KEY = os.getenv("AZURE_STORAGE_KEY") 
AZURE_CONN_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING") # New variable to check

CONTAINER_NAME = "clinicops-dvc" 

# CRITICAL FIX: Set the connection string globally. 
# MLflow's WASBS client prioritizes this environment variable for authentication.
if AZURE_CONN_STRING:
    os.environ['AZURE_STORAGE_CONNECTION_STRING'] = AZURE_CONN_STRING
else:
    # If connection string is not set (e.g., during local testing), use the key as a fallback
    # The ACI will inject the key into both variables, making this check robust.
    if AZURE_ACCOUNT and AZURE_KEY:
        os.environ['AZURE_STORAGE_ACCESS_KEY'] = AZURE_KEY
        

MODEL_URI = None
model = None

def get_latest_run_id():
    """
    Downloads the latest_model_run.txt file using the direct storage key.
    """
    if not AZURE_ACCOUNT or not AZURE_KEY:
        logging.error("Azure storage credentials (AZURE_STORAGE_ACCOUNT or AZURE_STORAGE_KEY) are missing.")
        return None

    try:
        blob_url = f"https://{AZURE_ACCOUNT}.blob.core.windows.net"
        
        # Use BlobClient for reading the pointer file, authenticated via AZURE_KEY
        blob_client = BlobClient(
            account_url=blob_url,
            container_name=CONTAINER_NAME,
            blob_name="latest_model_run.txt",
            credential=AZURE_KEY
        )

        logging.info(f"Attempting to download RUN_ID from {blob_client.url}")
        
        downloaded_blob = blob_client.download_blob()
        run_id = downloaded_blob.readall().decode("utf-8").strip()
        
        logging.info(f"Successfully retrieved RUN_ID: {run_id}")
        return run_id

    except Exception as e:
        logging.error(f"FATAL: Error retrieving latest RUN_ID from Blob Storage. Error: {e}")
        return None

# --- FastAPI App and Model Loading ---
app = FastAPI(
    title="KlinikOps Prediction Service",
    version="1.0",
    description="Length of Stay Prediction API"
)


@app.on_event("startup")
def load_model():
    """
    Loads the latest MLflow model. Authentication is now handled by the global environment variable
    AZURE_STORAGE_CONNECTION_STRING set earlier.
    """
    global model, MODEL_URI
    
    run_id = get_latest_run_id()
    if not run_id:
        logging.error("Cannot proceed without a valid RUN_ID from pointer file.")
        return

    # MLflow URI: will now use the AZURE_STORAGE_CONNECTION_STRING env var to authenticate
    MODEL_URI = f"wasbs://{CONTAINER_NAME}@{AZURE_ACCOUNT}.blob.core.windows.net/mlruns/{run_id}/artifacts/model"
    
    try:
        logging.info(f"Loading model from MLflow URI: {MODEL_URI}")
        model = mlflow.sklearn.load_model(MODEL_URI)
        logging.info("Model successfully loaded.")
        
    except Exception as e:
        logging.error(f"CRITICAL: Error loading model from Azure using URI {MODEL_URI}. Error: {e}")

# --- Data Schema (Must match the *final* training features) ---
class PatientData(BaseModel):
    # Numeric Features (9)
    hematocrit: float
    neutrophils: float
    sodium: float
    glucose: float
    bloodureanitro: float
    creatinine: float
    bmi: float
    pulse: float
    respiration: float
    
    # Categorical Features (16)
    rcount: str
    gender: str
    dialysisrenalendstage: str
    asthma: str
    irondef: str
    pneum: str
    substancedependence: str
    psychologicaldisordermajor: str
    depress: str
    psychother: str
    fibrosisandother: str
    malnutrition: str
    hemo: str
    secondarydiagnosisnonicd9: str
    discharged: str
    facid: str
    
# --- API Endpoint ---
@app.post("/predict", tags=["Prediction"])
def predict_length_of_stay(data: PatientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not yet loaded. Check logs for MLflow connection errors.")

    try:
        input_df = pd.DataFrame([data.model_dump()])
        prediction = model.predict(input_df)

        return {
            "predicted_length_of_stay": round(float(prediction[0]), 2),
            "unit": "Days"
        }

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Health Check Endpoint
@app.get("/health", tags=["Check"])
def health_check():
    return {"status": "ok", "model_loaded": model is not None, "api_version": app.version}