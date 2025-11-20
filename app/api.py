# app/api.py - POINTER FILE VERSION (FIXED AUTH)

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from azure.storage.blob import BlobClient 
import logging
import os
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)

# Env Vars
AZURE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT") 
AZURE_KEY = os.getenv("AZURE_STORAGE_KEY") 
AZURE_CONN_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "clinicops-dvc" 

# --- Auth Setup ---
# This part was WORKING in your logs, so we keep it!
if AZURE_CONN_STRING:
    os.environ['AZURE_STORAGE_CONNECTION_STRING'] = AZURE_CONN_STRING
    logging.info("✅ Auth: Using CONNECTION_STRING")
elif AZURE_KEY:
    os.environ['AZURE_STORAGE_ACCOUNT'] = AZURE_ACCOUNT 
    os.environ['AZURE_STORAGE_ACCESS_KEY'] = AZURE_KEY
else:
    logging.error("❌ Auth: No credentials found!")

model = None

def get_latest_run_id():
    """Downloads the pointer file to get the correct Run ID."""
    try:
        # Use Connection String for BlobClient if available
        if AZURE_CONN_STRING:
            blob_client = BlobClient.from_connection_string(
                conn_str=AZURE_CONN_STRING,
                container_name=CONTAINER_NAME,
                blob_name="latest_model_run.txt"
            )
        else:
            blob_client = BlobClient(
                account_url=f"https://{AZURE_ACCOUNT}.blob.core.windows.net",
                container_name=CONTAINER_NAME,
                blob_name="latest_model_run.txt",
                credential=AZURE_KEY
            )

        logging.info("📥 Downloading latest_model_run.txt...")
        run_id = blob_client.download_blob().readall().decode("utf-8").strip()
        logging.info(f"✅ Found Run ID: {run_id}")
        return run_id
    except Exception as e:
        logging.error(f"❌ Failed to download pointer file: {e}")
        return None

def load_model():
    global model
    run_id = get_latest_run_id()
    
    if not run_id:
        logging.error("❌ No Run ID found. Model cannot load.")
        return

    # Construct WASBS URI (Direct path to artifacts, bypassing Registry DB)
    # Note: Standard path is mlruns/<experiment_id>/<run_id>/artifacts/model
    # Local run usually has experiment_id '0'
    model_uri = f"wasbs://{CONTAINER_NAME}@{AZURE_ACCOUNT}.blob.core.windows.net/mlruns/0/{run_id}/artifacts/model"
    
    logging.info(f"⏳ Loading model from: {model_uri}")
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        logging.info("✅ Model loaded successfully!")
    except Exception as e:
        logging.error(f"❌ Model load failed: {e}")
        # Fallback: try without '0' experiment ID if structure differs
        try:
            alt_uri = f"wasbs://{CONTAINER_NAME}@{AZURE_ACCOUNT}.blob.core.windows.net/mlruns/{run_id}/artifacts/model"
            logging.info(f"🔄 Retrying with: {alt_uri}")
            model = mlflow.pyfunc.load_model(alt_uri)
            logging.info("✅ Model loaded (Fallback)!")
        except Exception as e2:
            logging.error(f"❌ Fallback failed: {e2}")

app = FastAPI()
load_model()

# ... (Pydantic models and endpoints remain the same) ...

class PatientData(BaseModel):
    # Numeric Features
    hematocrit: float
    neutrophils: float
    sodium: float
    glucose: float
    bloodureanitro: float
    creatinine: float
    bmi: float
    pulse: float
    respiration: float
    # Categorical Features
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

@app.post("/predict")
def predict(data: PatientData):
    if not model: raise HTTPException(503, "Model not loaded")
    try:
        if hasattr(data, 'model_dump'): df = pd.DataFrame([data.model_dump()])
        else: df = pd.DataFrame([data.dict()])
        return {"prediction": float(model.predict(df)[0])}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}