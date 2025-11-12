# app/api.py

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
# --- CRITICAL IMPORT: Needed to read files from Azure Blob ---
from azure.storage.blob import BlobClient 
import logging
import os
from pydantic import BaseModel
import numpy as np # Ensure numpy is imported if used in prediction

# --- Setup ---
logging.basicConfig(level=logging.INFO)

# AZURE_STORAGE_ACCOUNT and AZURE_STORAGE_KEY are now CRITICAL
AZURE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT") 
AZURE_KEY = os.getenv("AZURE_STORAGE_KEY") 

# ASSUMPTION: The MLflow artifacts are stored in a container named 'mlflow'
CONTAINER_NAME = "mlflow" 

# Model URI (initialized to None)
MODEL_URI = None
model = None

# --- CRITICAL NEW FUNCTION: Retrieves RUN_ID from Azure Blob ---
def get_latest_run_id():
    """
    Downloads the latest_model_run.txt file from Azure Blob Storage to get the RUN_ID.
    """
    if not AZURE_ACCOUNT or not AZURE_KEY:
        logging.error("Azure storage credentials (AZURE_STORAGE_ACCOUNT or AZURE_STORAGE_KEY) are missing.")
        return None

    try:
        # Construct the URL and BlobClient
        blob_url = f"https://{AZURE_ACCOUNT}.blob.core.windows.net"
        
        blob_client = BlobClient(
            account_url=blob_url,
            container_name=CONTAINER_NAME,
            blob_name="latest_model_run.txt",
            credential=AZURE_KEY
        )

        logging.info(f"Attempting to download RUN_ID from {blob_client.url}")
        
        # Download and read the file content
        downloaded_blob = blob_client.download_blob()
        # Read the content, decode from bytes to string, and remove leading/trailing whitespace
        run_id = downloaded_blob.readall().decode("utf-8").strip()
        
        logging.info(f"Successfully retrieved RUN_ID: {run_id}")
        return run_id

    except Exception as e:
        logging.error(f"FATAL: Error retrieving latest RUN_ID from Blob Storage. Check blob name/container/key. Error: {e}")
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
    Loads the latest MLflow model from Azure Blob Storage using the pseudo-registry.
    """
    global model, MODEL_URI
    
    # 1. Get RUN ID from pseudo-registry file
    run_id = get_latest_run_id()
    if not run_id:
        logging.error("Cannot proceed without a valid RUN_ID from pointer file.")
        return

    # 2. Construct the Model URI
    MODEL_URI = f"wasbs://{CONTAINER_NAME}@{AZURE_ACCOUNT}.blob.core.windows.net/mlruns/{run_id}/model"
    
    # 3. Load the model
    try:
        logging.info(f"Loading model from MLflow URI: {MODEL_URI}")
        
        # MLflow automatically uses AZURE_STORAGE_ACCOUNT and AZURE_STORAGE_KEY 
        # environment variables for authentication with WASBS protocol.
        model = mlflow.sklearn.load_model(MODEL_URI)
        logging.info("Model successfully loaded.")
        
    except Exception as e:
        logging.error(f"CRITICAL: Error loading model from Azure using URI {MODEL_URI}. Error: {e}")

# --- Data Schema (Ensure this matches your expected 26 features) ---
class PatientData(BaseModel):
    rcount: int
    gender: str
    dialysis: str
    mcd: str
    ecodes: str
    hmo: str
    health: str
    age: int
    eclaim: float
    pridx: int
    sdimd: int
    procedure: str
    pcode: str
    zid: str
    plos: float
    clmds: int
    disch: str
    orproc: str
    comorb: str
    diag: str
    ipros: str
    DRG: str
    last: str
    PG: str
    payer: str
    primaryphy: str
    
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