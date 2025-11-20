# app/api.py - POINTER FILE LOADER

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

# Auth Configuration
if AZURE_CONN_STRING:
    os.environ['AZURE_STORAGE_CONNECTION_STRING'] = AZURE_CONN_STRING
elif AZURE_KEY:
    os.environ['AZURE_STORAGE_ACCOUNT'] = AZURE_ACCOUNT 
    os.environ['AZURE_STORAGE_ACCESS_KEY'] = AZURE_KEY

model = None

def get_latest_run_id():
    """Downloads latest_model_run.txt from Azure."""
    try:
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

        run_id = blob_client.download_blob().readall().decode("utf-8").strip()
        return run_id
    except Exception as e:
        logging.error(f"Failed to download pointer: {e}")
        return None

def load_model():
    global model
    run_id = get_latest_run_id()
    
    if not run_id:
        logging.error("No Run ID found.")
        return

    # Direct path to artifacts in Azure Blob
    # Format: mlruns/0/<run_id>/artifacts/model
    model_uri = f"wasbs://{CONTAINER_NAME}@{AZURE_ACCOUNT}.blob.core.windows.net/mlruns/0/{run_id}/artifacts/model"
    
    logging.info(f"Loading from: {model_uri}")
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        logging.info("✅ Model loaded successfully!")
    except Exception:
        # Fallback if Experiment ID is missing from path
        try:
            alt_uri = f"wasbs://{CONTAINER_NAME}@{AZURE_ACCOUNT}.blob.core.windows.net/mlruns/{run_id}/artifacts/model"
            logging.info(f"Retrying with: {alt_uri}")
            model = mlflow.pyfunc.load_model(alt_uri)
            logging.info("✅ Model loaded (Fallback)!")
        except Exception as e:
            logging.error(f"❌ Load failed: {e}")

app = FastAPI()
load_model()

# --- Schema ---
class PatientData(BaseModel):
    hematocrit: float
    neutrophils: float
    sodium: float
    glucose: float
    bloodureanitro: float
    creatinine: float
    bmi: float
    pulse: float
    respiration: float
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