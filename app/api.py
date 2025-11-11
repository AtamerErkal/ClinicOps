# app/api.py

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

# --- Setup ---
logging.basicConfig(level=logging.INFO)

# TODO: Please update the {run_id} below with your latest successful MLflow run ID
# (e.g., 08d4114c23bf411587f47ab1b6fe0ff6)
MODEL_URI = "runs:/09833a3aaa5848eda02f9cf2a75fd3ec/model" 

# --- FastAPI App and Model Loading ---
app = FastAPI(
    title="ClinicOps Prediction Service",
    version="1.0",
    description="Length of Stay Prediction API"
)
model = None

@app.on_event("startup")
def load_model():
    """Loads the MLflow model when the application starts."""
    global model
    try:
        # You must enter your MLflow Run ID here
        logging.info(f"Loading model from MLflow URI: {MODEL_URI}")
        model = mlflow.sklearn.load_model(MODEL_URI)
        logging.info("Model successfully loaded.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        # It might be safer to stop the API if the model fails to load.

# --- Data Schema (Your actual CSV Schema) ---

# This schema reflects the 26 feature columns from Patient_Stay_Data.csv.
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
    """Makes a prediction for the length of stay based on patient data."""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not yet loaded.")

    try:
        # 1. Convert the incoming Pydantic object to a Pandas DataFrame
        input_df = pd.DataFrame([data.model_dump()])

        # 2. Make the prediction
        # The model pipeline handles necessary pre-processing (like OneHotEncoding).
        prediction = model.predict(input_df)

        # 3. Return the result
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