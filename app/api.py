# app/api.py

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

# --- Setup ---
logging.basicConfig(level=logging.INFO)
# TODO: Update the {run_id} below with your latest successful MLflow run ID
MODEL_URI = "runs:/{run_id}/model" 

# --- FastAPI App and Model Loading ---
app = FastAPI(
    title="KlinikOps Prediction Service",
    version="1.0",
    description="Length of Stay Prediction API"
)
model = None

@app.on_event("startup")
def load_model():
    """Loads the MLflow model when the application starts."""
    global model
    try:
        # Example: runs:/08d4114c23bf411587f47ab1b6fe0ff6/model
        logging.info(f"MLflow URI: {MODEL_URI}")
        model = mlflow.sklearn.load_model(MODEL_URI)
        logging.info("Model successfully loaded.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        # In a real environment, you might stop the service if the model fails to load.

# --- Data Schema (Structure of incoming prediction data) ---

# CRITICAL: This schema MUST reflect ALL 25 features/columns used in your model training.
class PatientData(BaseModel):
    # Example: Gender (Categorical)
    gender: str 
    # Example: Age (Numerical)
    age: int
    # Example: Insurance Type (Categorical)
    insurance: str
    # Example: Bed Type (Categorical)
    bed_type: str
    # TODO: ADD THE OTHER 21 COLUMNS FROM YOUR TRAINING DATA HERE!
    # ...
    # ... 

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