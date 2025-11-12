# app/api.py - IMPROVED MODEL LOADING

import mlflow
import mlflow.pyfunc
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
AZURE_CONN_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

CONTAINER_NAME = "clinicops-dvc" 

# CRITICAL FIX: Set the connection string globally
if AZURE_CONN_STRING:
    os.environ['AZURE_STORAGE_CONNECTION_STRING'] = AZURE_CONN_STRING
else:
    if AZURE_ACCOUNT and AZURE_KEY:
        os.environ['AZURE_STORAGE_ACCESS_KEY'] = AZURE_KEY

# Set MLflow tracking URI
TRACKING_URI = f"wasbs://{CONTAINER_NAME}@{AZURE_ACCOUNT}.blob.core.windows.net"
mlflow.set_tracking_uri(TRACKING_URI)
logging.info(f"MLflow Tracking URI set to: {TRACKING_URI}")

MODEL_URI = None
model = None

def get_latest_run_id():
    """
    Downloads the latest_model_run.txt file using the direct storage key.
    """
    if not AZURE_ACCOUNT or not AZURE_KEY:
        logging.error("Azure storage credentials are missing.")
        return None

    try:
        blob_url = f"https://{AZURE_ACCOUNT}.blob.core.windows.net"
        
        blob_client = BlobClient(
            account_url=blob_url,
            container_name=CONTAINER_NAME,
            blob_name="latest_model_run.txt",
            credential=AZURE_KEY
        )

        logging.info(f"Attempting to download RUN_ID from {blob_client.url}")
        
        downloaded_blob = blob_client.download_blob()
        run_id = downloaded_blob.readall().decode("utf-8").strip()
        
        logging.info(f"✅ Successfully retrieved RUN_ID: {run_id}")
        return run_id

    except Exception as e:
        logging.error(f"❌ Error retrieving latest RUN_ID: {e}")
        return None

# --- FastAPI App ---
app = FastAPI(
    title="KlinikOps Prediction Service",
    version="1.0",
    description="Length of Stay Prediction API"
)


@app.on_event("startup")
def load_model():
    """
    Loads the latest MLflow model with proper Azure Blob authentication
    """
    global model, MODEL_URI
    
    run_id = get_latest_run_id()
    if not run_id:
        logging.error("❌ Cannot proceed without a valid RUN_ID")
        return

    # Direct WASBS path (most reliable for Azure Blob)
    # MLflow structure: {run_id}/artifacts/model/
    MODEL_URI = f"wasbs://{CONTAINER_NAME}@{AZURE_ACCOUNT}.blob.core.windows.net/{run_id}/artifacts/model"
    
    try:
        logging.info(f"🔄 Loading model from: {MODEL_URI}")
        logging.info(f"Authentication method: {'CONNECTION_STRING' if AZURE_CONN_STRING else 'ACCESS_KEY'}")
        
        # Use pyfunc for better compatibility
        model = mlflow.pyfunc.load_model(MODEL_URI)
        logging.info("✅ Model successfully loaded!")
        
    except Exception as e:
        logging.error(f"❌ CRITICAL: Model loading failed!")
        logging.error(f"Error type: {type(e).__name__}")
        logging.error(f"Error details: {str(e)}")
        
        # Try alternative path (in case MLflow uses different structure)
        logging.info("🔄 Trying alternative path without 'artifacts'...")
        try:
            ALT_URI = f"wasbs://{CONTAINER_NAME}@{AZURE_ACCOUNT}.blob.core.windows.net/{run_id}/model"
            logging.info(f"Alternative URI: {ALT_URI}")
            model = mlflow.pyfunc.load_model(ALT_URI)
            MODEL_URI = ALT_URI
            logging.info("✅ Model loaded via alternative path!")
        except Exception as alt_error:
            logging.error(f"❌ Alternative path also failed: {alt_error}")
            
            # Last resort: try with mlruns prefix
            logging.info("🔄 Last resort: trying with /mlruns/ prefix...")
            try:
                LAST_URI = f"wasbs://{CONTAINER_NAME}@{AZURE_ACCOUNT}.blob.core.windows.net/mlruns/{run_id}/artifacts/model"
                logging.info(f"Last resort URI: {LAST_URI}")
                model = mlflow.pyfunc.load_model(LAST_URI)
                MODEL_URI = LAST_URI
                logging.info("✅ Model loaded via /mlruns/ path!")
            except Exception as last_error:
                logging.error(f"❌ All loading attempts failed. Final error: {last_error}")

# --- Data Schema ---
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
    
# --- API Endpoints ---
@app.post("/predict", tags=["Prediction"])
def predict_length_of_stay(data: PatientData):
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Check logs for errors."
        )

    try:
        input_df = pd.DataFrame([data.model_dump()])
        prediction = model.predict(input_df)

        return {
            "predicted_length_of_stay": round(float(prediction[0]), 2),
            "unit": "Days",
            "model_uri": MODEL_URI
        }

    except Exception as e:
        logging.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health", tags=["Check"])
def health_check():
    return {
        "status": "ok", 
        "model_loaded": model is not None,
        "model_uri": MODEL_URI,
        "api_version": app.version,
        "mlflow_tracking_uri": TRACKING_URI
    }

@app.get("/debug", tags=["Check"])
def debug_info():
    """Debug endpoint to check environment and model status"""
    return {
        "azure_account": AZURE_ACCOUNT,
        "container": CONTAINER_NAME,
        "tracking_uri": TRACKING_URI,
        "model_uri": MODEL_URI,
        "model_loaded": model is not None,
        "env_vars": {
            "AZURE_STORAGE_CONNECTION_STRING": "SET" if AZURE_CONN_STRING else "NOT SET",
            "AZURE_STORAGE_ACCESS_KEY": "SET" if AZURE_KEY else "NOT SET"
        }
    }