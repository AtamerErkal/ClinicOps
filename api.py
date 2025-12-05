# api.py - ClinicOps FastAPI Prediction Service
# Fully compatible with Azure Container Instances + Mobile Streamlit UI
# English comments & logs for clarity

import os
import logging
import time
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from azure.storage.blob import BlobClient
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from pydantic import BaseModel
import traceback

# Configure logging (visible in Azure ACI logs)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure & MLflow Configuration
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
AZURE_STORAGE_KEY = os.getenv("AZURE_STORAGE_KEY")
AZURE_CONN_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME", "clinicops-dvc")
RESOURCE_GROUP = os.getenv("RG_NAME", "ClinicOps-RG-2025")
LOCATION = os.getenv("LOCATION", "germanywestcentral")
RUN_ID = os.getenv("MLFLOW_RUN_ID")  # From training step

# Global model variables
model = None
MODEL_URI = None
expected_features = None

app = FastAPI(
    title="ClinicOps AI Prediction API",
    description="Predicts hospital length of stay using Random Forest + MLflow",
    version="1.0.0"
)


def get_latest_run_id() -> str | None:
    """Robust way to fetch latest run ID – works with connection string OR managed identity"""
    try:
        # CASE 1: Connection string varsa (GitHub Actions'ta var)
        if AZURE_CONN_STRING:
            logger.info("Using AZURE_STORAGE_CONNECTION_STRING to fetch latest run ID")
            blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONN_STRING)
            blob_client = blob_service_client.get_blob_client(
                container=CONTAINER_NAME,
                blob="latest_model_run.txt"
            )
        else:
            # CASE 2: Connection string yoksa → Managed Identity veya Storage Key kullan
            logger.info("AZURE_CONN_STRING not found → trying DefaultAzureCredential")
            credential = DefaultAzureCredential()
            account_url = f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net"
            blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
            blob_client = blob_service_client.get_blob_client(
                container=CONTAINER_NAME,
                blob="latest_model_run.txt"
            )

        # Download & return run ID
        download = blob_client.download_blob()
        run_id = download.readall().decode("utf-8").strip()
        logger.info(f"Successfully fetched latest Run ID: {run_id}")
        return run_id

    except Exception as e:
        logger.warning(f"Could not fetch latest run ID: {e}. Falling back to training-time RUN_ID")
        return RUN_ID


def load_model():
    """Load MLflow model from Azure Blob with retry logic (critical for ACI cold start)"""
    global model, MODEL_URI, expected_features

    run_id = get_latest_run_id()
    if not run_id:
        logger.error("No Run ID available - cannot load model")
        return

    model_uri = f"wasbs://{CONTAINER_NAME}@{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/models/{run_id}/model"
    MODEL_URI = model_uri
    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = AZURE_CONN_STRING

    max_retries = 8  # 8 attempts × 30s = 4 minutes total
    for attempt in range(max_retries):
        try:
            logger.info(f"Loading model from {model_uri} - Attempt {attempt + 1}/{max_retries}")
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info("Model loaded successfully!")

            # Extract expected feature names (robust across MLflow versions)
            try:
                underlying = model._model_impl.python_model
                if hasattr(underlying, "model"):
                    expected_features = list(underlying.model.feature_names_in_)
                elif hasattr(underlying, "feature_names_in_"):
                    expected_features = list(underlying.feature_names_in_)
                else:
                    expected_features = list(model._model_impl.sklearn_model.feature_names_in_)
                logger.info(f"Extracted {len(expected_features)} feature names")
            except Exception as e:
                logger.warning(f"Could not extract feature names: {e}")
                expected_features = None

            return

        except Exception as e:
            logger.warning(f"Model load failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(30)
            else:
                logger.error("Model loading failed after all retries")

    # Do NOT crash - allow health endpoint to report warming_up
    logger.error("Model could not be loaded - API will return warming status")


# Startup: Load model with patience (no cleanup - handled by destroy workflow)
@app.on_event("startup")
async def startup_event():
    logger.info("ClinicOps API starting up...")
    load_model()  # Non-blocking, retry-enabled


# Input schema - flexible for mobile frontend
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

    rcount: str = "0"
    gender: str = "F"
    dialysisrenalendstage: int | str = 0
    asthma: int | str = 0
    irondef: int | str = 0
    pneum: int | str = 0
    substancedependence: int | str = 0
    psychologicaldisordermajor: int | str = 0
    depress: int | str = 0
    psychother: int | str = 0
    fibrosisandother: int | str = 0
    malnutrition: int | str = 0
    hemo: int | str = 0
    secondarydiagnosisnonicd9: str = "0"
    facid: str = "A"

    class Config:
        extra = "ignore"


@app.get("/")
def root():
    return {
        "service": "ClinicOps AI Prediction API",
        "status": "running",
        "model_loaded": model is not None,
        "docs": "/docs"
    }


@app.get("/health")
def health():
    """Mobile-friendly health check - returns status even if model is warming"""
    status = "ok" if model is not None else "warming_up"
    return {
        "status": status,
        "model_loaded": bool(model),
        "run_id": RUN_ID,
        "model_uri": MODEL_URI,
        "expected_features": len(expected_features) if expected_features else "unknown",
        "timestamp": time.time()
    }


@app.post("/predict")
def predict(data: PatientData):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Please try again in 30 seconds."
        )

    try:
        input_dict = data.model_dump()
        df = pd.DataFrame([input_dict])

        # Preprocessing - must match train.py exactly
        numeric_cols = ['hematocrit', 'neutrophils', 'sodium', 'glucose',
                        'bloodureanitro', 'creatinine', 'bmi', 'pulse', 'respiration']
        cat_cols = ['rcount', 'gender', 'dialysisrenalendstage', 'asthma', 'irondef',
                    'pneum', 'substancedependence', 'psychologicaldisordermajor', 'depress',
                    'psychother', 'fibrosisandother', 'malnutrition', 'hemo',
                    'secondarydiagnosisnonicd9', 'facid']

        df[cat_cols] = df[cat_cols].astype(str)

        # Comorbidity score (exact match with training)
        binary_cols = ['dialysisrenalendstage', 'asthma', 'irondef', 'pneum',
                       'substancedependence', 'psychologicaldisordermajor', 'depress',
                       'psychother', 'fibrosisandother', 'malnutrition', 'hemo']
        df['comorbidity_score'] = df[binary_cols].apply(
            lambda row: sum(int(x) for x in row), axis=1
        ).astype(float)
        numeric_cols.append('comorbidity_score')

        # One-hot encoding (full, no drop_first - matches training)
        df_encoded = pd.get_dummies(df[cat_cols], dtype=int)

        # Align with training features
        if expected_features is None:
            raise ValueError("Model feature names not available")

        dummy_cols = [c for c in expected_features if c not in numeric_cols]
        df_encoded = df_encoded.reindex(columns=dummy_cols, fill_value=0)

        # Final input
        df_numeric = df[numeric_cols].astype(float)
        df_final = pd.concat([df_numeric, df_encoded], axis=1)
        df_final = df_final.reindex(columns=expected_features, fill_value=0.0)

        # Ensure dummy columns are bool (MLflow sometimes expects this)
        for col in df_final.columns:
            if col not in numeric_cols:
                df_final[col] = df_final[col].astype(bool)

        # Prediction (log-scale model)
        pred_log = model.predict(df_final)
        prediction = float(np.expm1(pred_log)[0])

        return {
            "predicted_length_of_stay": round(prediction, 2),
            "model_run_id": RUN_ID,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/feature_importance")
def feature_importance():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        rf_model = model._model_impl.python_model
        if hasattr(rf_model, "model"):
            rf_model = rf_model.model

        importances = rf_model.feature_importances_
        features = rf_model.feature_names_in_

        top_10 = dict(sorted(zip(features, importances), key=lambda x: x[1], reverse=True)[:10])
        return {"top_10_features": top_10}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/expected_features")
def debug_features():
    return {
        "feature_count": len(expected_features) if expected_features else 0,
        "features": expected_features or "not_yet_loaded",
        "hint": "Wait for /health to return 'ok'"
    }