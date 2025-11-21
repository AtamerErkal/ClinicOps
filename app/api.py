# app/api.py - POINTER FILE LOADER

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from azure.storage.blob import BlobClient 
import logging
import os
from pydantic import BaseModel
from fastapi import APIRouter

logging.basicConfig(level=logging.INFO)

# Env Vars (ACI'dan gelenler)
AZURE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT") 
AZURE_KEY = os.getenv("AZURE_STORAGE_KEY") 
AZURE_CONN_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "clinicops-dvc" 

# Set authentication for MLflow
if AZURE_CONN_STRING:
    os.environ['AZURE_STORAGE_CONNECTION_STRING'] = AZURE_CONN_STRING
elif AZURE_KEY:
    os.environ['AZURE_STORAGE_ACCOUNT'] = AZURE_ACCOUNT 
    os.environ['AZURE_STORAGE_ACCESS_KEY'] = AZURE_KEY

model = None
MODEL_URI = None

def get_latest_run_id():
    """Downloads the latest_model_run.txt file from Azure Blob"""
    if not AZURE_ACCOUNT or not (AZURE_KEY or AZURE_CONN_STRING):
        logging.error("❌ Azure credentials missing")
        return None

    try:
        # Blob Client using Connection String or Account Key
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
    global model, MODEL_URI
    run_id = get_latest_run_id()
    
    if not run_id:
        logging.error("❌ No Run ID found.")
        return

    # CRITICAL FIX: Model path includes experiment_id (0 by default)
    model_uri = f"wasbs://{CONTAINER_NAME}@{AZURE_ACCOUNT}.blob.core.windows.net/mlruns/0/{run_id}/artifacts/model"
    MODEL_URI = model_uri
    
    logging.info(f"🔄 Loading from: {model_uri}")
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        logging.info("✅ Model loaded successfully!")
    except Exception as e:
        logging.error(f"❌ Load failed: {e}")
        logging.error(f"Error type: {type(e).__name__}")

app = FastAPI(version="1.0.0")
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



@app.get("/feature_importance", tags=["Prediction"])
def feature_importance():
    if model is None:
        raise HTTPException(503, "Model not loaded")
    
    try:
        # Only works if the model is sklearn RandomForest wrapped in MLflow
        rf_model = model._model_impl.python_model.model  # PyFunc wrapper internal
        if hasattr(rf_model, "feature_importances_"):
            fi = rf_model.feature_importances_
            # Map to feature names
            features = [
                'hematocrit','neutrophils','sodium','glucose','bloodureanitro',
                'creatinine','bmi','pulse','respiration',
                'rcount','gender','dialysisrenalendstage','asthma','irondef','pneum',
                'substancedependence','psychologicaldisordermajor','depress','psychother',
                'fibrosisandother','malnutrition','hemo','secondarydiagnosisnonicd9',
                'discharged','facid'
            ]
            fi_dict = dict(zip(features, fi))
            return fi_dict
        else:
            return {"error": "Feature importance not available for this model type."}
    except Exception as e:
        raise HTTPException(500, detail=f"Feature importance failed: {str(e)}")


@app.post("/predict", tags=["Prediction"])
def predict(data: PatientData):
    if model is None: raise HTTPException(503, "Model not loaded")
    try:
        input_df = pd.DataFrame([data.model_dump()])
        prediction = model.predict(input_df)
        return {"predicted_length_of_stay": round(float(prediction[0]), 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health", tags=["Check"])
def health():
    return {"status": "ok", "model_loaded": model is not None, "model_uri": MODEL_URI}