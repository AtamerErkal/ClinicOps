# app/api.py - FINAL WORKING VERSION

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
import logging
import os
from pydantic import BaseModel
import sys
# HATALI SATIR SİLİNDİ: from azure.ai.ml import MLClient (Artık yok)

# --- Setup ---
logging.basicConfig(level=logging.INFO)

# Model Ayarları
MODEL_NAME = "ClinicOpsLengthOfStayModel"
MODEL_STAGE = "Production"
MODEL_URI_REGISTRY = f"models:/{MODEL_NAME}/{MODEL_STAGE}" 

# Ortam Değişkenleri
AZURE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT") 
AZURE_KEY = os.getenv("AZURE_STORAGE_KEY") 
AZURE_CONN_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "clinicops-dvc" 

# --- MLflow Konfigürasyonu ---
MLFLOW_TRACKING_URI = f"wasbs://{CONTAINER_NAME}@{AZURE_ACCOUNT}.blob.core.windows.net/mlflow_tracking"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
logging.info(f"✅ MLflow Tracking set: {MLFLOW_TRACKING_URI}")

# Kimlik Doğrulama
if AZURE_CONN_STRING:
    os.environ['AZURE_STORAGE_CONNECTION_STRING'] = AZURE_CONN_STRING
    logging.info("✅ Auth: Using CONNECTION_STRING")
elif AZURE_KEY:
    os.environ['AZURE_STORAGE_ACCOUNT'] = AZURE_ACCOUNT 
    os.environ['AZURE_STORAGE_ACCESS_KEY'] = AZURE_KEY
    logging.info("✅ Auth: Using ACCESS_KEY")
else:
    logging.error("❌ Auth: No Azure credentials found!")

# --- Global Model ---
model = None

def load_model():
    global model
    logging.info(f"⏳ Loading model from: {MODEL_URI_REGISTRY}")
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI_REGISTRY) 
        logging.info("✅ Model loaded successfully!")
    except Exception as e:
        logging.error(f"❌ Model load failed: {e}")
        import traceback
        traceback.print_exc()
        model = None

# API Başlatma
app = FastAPI(title="KlinikOps API", version="1.0")

# Başlangıçta modeli yükle
load_model()

# --- Şema ---
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

@app.post("/predict", tags=["Prediction"])
def predict(data: PatientData):
    if not model:
        raise HTTPException(503, "Model not loaded")
    try:
        # Pydantic v2 uyumluluğu için model_dump, yoksa dict
        if hasattr(data, 'model_dump'):
             df = pd.DataFrame([data.model_dump()])
        else:
             df = pd.DataFrame([data.dict()])
        
        pred = model.predict(df)
        return {"prediction": float(pred[0])}
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(500, str(e))

@app.get("/health", tags=["Check"])
def health():
    return {
        "status": "ok", 
        "model_loaded": model is not None,
        "tracking_uri": MLFLOW_TRACKING_URI
    }