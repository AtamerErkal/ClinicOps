import os
import logging
import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from azure.storage.blob import BlobClient
from azure.identity import DefaultAzureCredential
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
from pydantic import BaseModel
import time

logging.basicConfig(level=logging.INFO)

# Azure Config
AZURE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
AZURE_KEY = os.getenv("AZURE_STORAGE_KEY")
AZURE_CONN_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "clinicops-dvc"
RESOURCE_GROUP = os.getenv("RG_NAME")
ACI_NAME_PREFIX = "clinicops-api"
LOCATION = os.getenv("LOCATION", "westeurope")

# Model globals
model = None
MODEL_URI = None
RUN_ID = os.getenv("MLFLOW_RUN_ID")

def get_latest_run_id():
    """Fetch latest run ID from Azure Blob"""
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
        logging.info(f"✅ Latest Run ID from blob: {run_id}")
        return run_id
    except Exception as e:
        logging.error(f"❌ Failed to fetch run ID from blob: {e}")
        # Fallback to environment variable
        if RUN_ID:
            logging.info(f"Using RUN_ID from env: {RUN_ID}")
            return RUN_ID
        return None

def delete_old_aci_instances():
    """Delete old ACI instances to save costs"""
    try:
        credential = DefaultAzureCredential()
        subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        if not subscription_id:
            logging.warning("AZURE_SUBSCRIPTION_ID not set, skipping ACI cleanup")
            return
            
        client = ContainerInstanceManagementClient(credential, subscription_id)
        groups = client.container_groups.list_by_resource_group(RESOURCE_GROUP)
        
        for g in groups:
            if g.name.startswith(ACI_NAME_PREFIX):
                logging.info(f"Deleting old ACI: {g.name}")
                client.container_groups.begin_delete(RESOURCE_GROUP, g.name)
    except Exception as e:
        logging.error(f"ACI cleanup failed: {e}")

def load_model():
    """Load model from Azure Blob Storage via MLflow"""
    global model, MODEL_URI
    
    run_id = get_latest_run_id()
    if not run_id:
        logging.error("❌ No Run ID found!")
        return

    # Construct wasbs URI - CRITICAL: must match upload path
    model_uri = f"wasbs://{CONTAINER_NAME}@{AZURE_ACCOUNT}.blob.core.windows.net/models/{run_id}/model"
    MODEL_URI = model_uri
    
    logging.info(f"🔄 Attempting to load model from: {model_uri}")
    
    # Configure Azure credentials for MLflow
    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = AZURE_CONN_STRING
    
    # Retry logic for network delays
    max_retries = 5
    for attempt in range(max_retries):
        try:
            logging.info(f"Load attempt {attempt + 1}/{max_retries}...")
            model = mlflow.pyfunc.load_model(model_uri)
            logging.info("✅ Model loaded successfully!")
            
            # Verify model type
            logging.info(f"Model type: {type(model)}")
            return
            
        except Exception as e:
            logging.error(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 30 * (attempt + 1)  # Exponential backoff
                logging.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                logging.error("❌ Model load failed after all retries")

# FastAPI App
app = FastAPI(title="ClinicOps API", version="1.0.0")

@app.on_event("startup")
def startup_event():
    logging.info("🚀 Starting ClinicOps API...")
    delete_old_aci_instances()
    load_model()

# Pydantic schema
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

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_uri": MODEL_URI,
        "run_id": RUN_ID
    }

@app.get("/feature_importance")
def feature_importance():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Access underlying sklearn model
        # MLflow pyfunc wraps the model, need to unwrap it
        if hasattr(model, '_model_impl'):
            sklearn_model = model._model_impl.python_model
            if hasattr(sklearn_model, 'model'):
                rf_model = sklearn_model.model
            else:
                rf_model = sklearn_model
        else:
            raise Exception("Cannot access underlying model")
        
        # Get feature importances
        importances = rf_model.feature_importances_
        feature_names = rf_model.feature_names_in_
        
        # Sort by importance
        feature_importance_dict = dict(zip(feature_names, importances))
        sorted_features = dict(sorted(
            feature_importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return sorted_features
        
    except Exception as e:
        logging.error(f"Feature importance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(data: PatientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_dict = data.model_dump()
        df = pd.DataFrame([input_dict])
        
        # Apply same preprocessing as training
        # 1. Label encode specific columns (if they weren't one-hot encoded)
        label_mappings = {
            'rcount': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5+': 5},
            'gender': {'M': 0, 'F': 1},
            'discharged': {'A': 0, 'B': 1, 'C': 2, 'D': 3},
            'facid': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}
        }
        
        for col, mapping in label_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0).astype(int)
        
        # 2. One-hot encode remaining categorical columns
        df_encoded = pd.get_dummies(df, drop_first=True)
        
        # 3. Get expected features from model
        if hasattr(model, '_model_impl'):
            sklearn_model = model._model_impl.python_model
            if hasattr(sklearn_model, 'model'):
                expected_features = sklearn_model.model.feature_names_in_
            else:
                expected_features = sklearn_model.feature_names_in_
        else:
            raise Exception("Cannot access model features")
        
        # 4. Align columns with training data
        df_final = df_encoded.reindex(columns=expected_features, fill_value=0)
        
        # 5. Predict using MLflow pyfunc wrapper
        prediction = model.predict(df_final)
        
        return {
            "predicted_length_of_stay": round(float(prediction[0]), 2),
            "input_features_count": len(df_final.columns)
        }
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {
        "message": "ClinicOps API is running",
        "endpoints": ["/health", "/predict", "/feature_importance"],
        "model_status": "loaded" if model else "not loaded"
    }