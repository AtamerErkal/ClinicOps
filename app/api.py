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
    
    # AML Registry'den Load (auth yok, AML backend yönetir)
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
        
        # Fallback: Eski wasbs (nadir, mevcut kodunu buraya koy)
        run_id = get_latest_run_id()
        if not run_id:
            logging.error("❌ Cannot proceed without RUN_ID")
            return

        MODEL_URI = f"wasbs://{CONTAINER_NAME}@{AZURE_ACCOUNT}.blob.core.windows.net/mlruns/{run_id}/artifacts/model"
        
        try:
            logging.info(f"🔄 Loading model from: {MODEL_URI}")
            model = mlflow.pyfunc.load_model(MODEL_URI)
            logging.info("✅ Model loaded successfully!")
        except Exception as e:
            logging.error(f"❌ Model loading failed: {e}")
            logging.error(f"Error type: {type(e).__name__}")
            
            # Alternative paths
            alternative_paths = [
                f"wasbs://{CONTAINER_NAME}@{AZURE_ACCOUNT}.blob.core.windows.net/{run_id}/artifacts/model",
                f"wasbs://{CONTAINER_NAME}@{AZURE_ACCOUNT}.blob.core.windows.net/mlruns/{run_id}/model"
            ]
            
            for i, alt_path in enumerate(alternative_paths, 1):
                try:
                    logging.info(f"🔄 Trying alternative {i}: {alt_path}")
                    model = mlflow.pyfunc.load_model(alt_path)
                    MODEL_URI = alt_path
                    logging.info(f"✅ Model loaded from alternative {i}!")
                    return
                except Exception as alt_error:
                    logging.warning(f"⚠️ Alternative {i} failed: {alt_error}")
                    continue
            
            logging.error("❌ All loading attempts failed. Check Azure portal for blob existence.")

def get_latest_run_id():
    """Downloads the latest_model_run.txt file from Azure Blob"""
    if not AZURE_ACCOUNT or not (AZURE_CONN_STRING or AZURE_KEY):
        logging.error("❌ Azure credentials missing")
        return None

    try:
        from azure.core.credentials import AzureNamedKeyCredential
        
        blob_url = f"https://{AZURE_ACCOUNT}.blob.core.windows.net"
        if AZURE_CONN_STRING:
            credential = AZURE_CONN_STRING
        else:
            credential = AzureNamedKeyCredential(name=AZURE_ACCOUNT, key=AZURE_KEY)
            
        blob_client = BlobClient(
            account_url=blob_url,
            container_name=CONTAINER_NAME,
            blob_name="latest_model_run.txt",
            credential=credential
        )

        logging.info(f"📥 Downloading RUN_ID from {blob_client.url}")
        
        downloaded_blob = blob_client.download_blob()
        run_id = downloaded_blob.readall().decode("utf-8").strip()
        
        logging.info(f"✅ Retrieved RUN_ID: {run_id}")
        return run_id

    except Exception as e:
        logging.error(f"❌ Error retrieving RUN_ID: {e}")
        return None

# --- Data Schema ---
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
    
# --- API Endpoints ---
@app.post("/predict", tags=["Prediction"])
def predict_length_of_stay(data: PatientData):
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Check logs."
        )

    try:
        input_df = pd.DataFrame([data.model_dump()])
        prediction = model.predict(input_df)

        return {
            "predicted_length_of_stay": round(float(prediction[0]), 2),
            "unit": "Days"
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
        "api_version": app.version
    }

@app.get("/debug", tags=["Debug"])
def debug_info():
    """Debug endpoint to check configuration"""
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