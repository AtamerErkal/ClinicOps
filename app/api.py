# app/api.py

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import os

# --- Setup ---
logging.basicConfig(level=logging.INFO)

# The RUN_ID will be dynamically set by the GitHub Actions pipeline as an environment variable (MLFLOW_RUN_ID)
RUN_ID = os.getenv("MLFLOW_RUN_ID")
MODEL_URI = f"runs:/{RUN_ID}/model" if RUN_ID else None 

# --- FastAPI App and Model Loading ---
app = FastAPI(
    title="KlinikOps Prediction Service",
    version="1.0",
    description="Length of Stay Prediction API"
)
model = None

@app.on_event("startup")
def load_model():
    """
    Loads the MLflow model when the application starts. 
    It requires the MLFLOW_RUN_ID, AZURE_STORAGE_ACCOUNT, and AZURE_STORAGE_KEY
    environment variables to be set by the ACI deployment step.
    """
    global model
    
    if not MODEL_URI:
        logging.error("MLFLOW_RUN_ID environment variable is missing. Cannot proceed with model loading.")
        return
        
    # Crucial check for Azure Storage access
    if not os.getenv("AZURE_STORAGE_ACCOUNT") or not os.getenv("AZURE_STORAGE_KEY"):
        logging.error("Azure Storage credentials are NOT set as environment variables. Model loading will likely fail.")
        return

    try:
        logging.info(f"Loading model from MLflow URI: {MODEL_URI}")
        model = mlflow.sklearn.load_model(MODEL_URI)
        logging.info("Model successfully loaded.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")

# --- Data Schema (Actual 26-Feature Schema from CSV) ---
# This ensures 26 features are shown on the Swagger UI.

class PatientData(BaseModel):
    """Schema representing the features expected by the trained model."""
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
    """Makes a prediction for the length of stay based on patient data."""
    
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
    """Checks if the service is up and the model is loaded."""
    return {"status": "ok", "model_loaded": model is not None, "api_version": app.version}