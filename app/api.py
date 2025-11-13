# app/api.py - AML MLflow Registry Load

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from azure.storage.blob import BlobClient 
import logging
import os
from pydantic import BaseModel

# --- Setup ---
logging.basicConfig(level=logging.INFO)

# Get environment variables
AZURE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT") 
AZURE_KEY = os.getenv("AZURE_STORAGE_KEY") 
AZURE_CONN_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

# AML Env
AML_TRACKING_URI = os.getenv("AML_TRACKING_URI")

CONTAINER_NAME = "clinicops-dvc" 

# Set MLflow Tracking URI to AML (auth otomatik)
if AML_TRACKING_URI:
    mlflow.set_tracking_uri(AML_TRACKING_URI)
    logging.info(f"✅ MLflow Tracking: AML Workspace ({AML_TRACKING_URI})")
else:
    logging.warning("⚠️ No AML_TRACKING_URI—using local fallback")

# Set authentication for fallback (wasbs)
if AZURE_CONN_STRING:
    os.environ['AZURE_STORAGE_CONNECTION_STRING'] = AZURE_CONN_STRING
    logging.info("✅ Using CONNECTION_STRING for authentication")
elif AZURE_KEY:
    os.environ['AZURE_STORAGE_ACCESS_KEY'] = AZURE_KEY
    logging.info("✅ Using ACCESS_KEY for authentication")
else:
    logging.error("❌ No Azure credentials found!")

MODEL_URI = None
model = None

# get_latest_run_id() fallback için aynı (kullanılmayabilir)

# --- FastAPI App ---
app = FastAPI(
    title="KlinikOps Prediction Service",
    version="1.0",
    description="Length of Stay Prediction API"
)

@app.on_event("startup")
def load_model():
    """Loads the latest MLflow model from AML Registry"""
    global model, MODEL_URI
    
    # YENİ: AML Registry'den Load (auth yok, AML backend yönetir)
    model_name = "length_of_stay_model"
    version = "latest"
    try:
        logging.info(f"🔄 Loading from AML registry: models://{model_name}/{version}")
        model = mlflow.pyfunc.load_model(f"models://{model_name}/{version}")
        MODEL_URI = f"models://{model_name}/{version}"
        logging.info("✅ Model loaded from AML registry!")
        return
    except Exception as e:
        logging.error(f"❌ AML registry load failed: {e}")
        
        # Fallback: Eski wasbs (nadir)
        # ... (mevcut fallback kodunu buraya koy, ama AML ile gerek kalmayacak)

# --- Data Schema --- (aynı)
class PatientData(BaseModel):
    # ... (mevcut features aynı)

# --- API Endpoints --- (aynı)
@app.post("/predict", tags=["Prediction"])
def predict_length_of_stay(data: PatientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check logs.")
    try:
        input_df = pd.DataFrame([data.model_dump()])
        prediction = model.predict(input_df)
        return {"predicted_length_of_stay": round(float(prediction[0]), 2), "unit": "Days"}
    except Exception as e:
        logging.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health", tags=["Check"])
def health_check():
    return {"status": "ok", "model_loaded": model is not None, "model_uri": MODEL_URI, "api_version": app.version}

@app.get("/debug", tags=["Debug"])
def debug_info():
    return {
        "azure_account": AZURE_ACCOUNT,
        "container": CONTAINER_NAME,
        "model_uri": MODEL_URI,
        "model_loaded": model is not None,
        "credentials": {
            "connection_string": "SET" if AZURE_CONN_STRING else "NOT SET",
            "access_key": "SET" if AZURE_KEY else "NOT SET"
        },
        "aml_tracking_uri": AML_TRACKING_URI or "NOT SET"
    }