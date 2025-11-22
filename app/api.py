import os
import logging
import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from azure.storage.blob import BlobClient
from azure.identity import DefaultAzureCredential
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
from azure.mgmt.containerinstance.models import ContainerGroup
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)

# --- Azure Config ---
AZURE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
AZURE_KEY = os.getenv("AZURE_STORAGE_KEY")
AZURE_CONN_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "clinicops-dvc"
RESOURCE_GROUP = os.getenv("RG_NAME")
ACI_NAME_PREFIX = "clinicops-api"
LOCATION = os.getenv("LOCATION", "westeurope")

# --- MLflow & Model ---
model = None
MODEL_URI = None

def get_latest_run_id():
    """Download latest_model_run.txt from Azure Blob"""
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
        logging.info(f"Latest Run ID: {run_id}")
        return run_id
    except Exception as e:
        logging.error(f"Failed to get latest run id: {e}")
        return None

def delete_old_aci_instances():
    """Delete previous ACI instances to save costs"""
    try:
        credential = DefaultAzureCredential()
        client = ContainerInstanceManagementClient(credential, os.getenv("AZURE_SUBSCRIPTION_ID"))
        groups = client.container_groups.list_by_resource_group(RESOURCE_GROUP)
        for g in groups:
            if g.name.startswith(ACI_NAME_PREFIX):
                logging.info(f"Deleting old ACI: {g.name}")
                client.container_groups.begin_delete(RESOURCE_GROUP, g.name)
    except Exception as e:
        logging.error(f"ACI cleanup failed: {e}")

def load_model():
    global model, MODEL_URI
    run_id = get_latest_run_id()
    if not run_id:
        logging.error("No Run ID found.")
        return

    # Model path includes default experiment 0 (mlflow local storage)
    model_uri = f"wasbs://{CONTAINER_NAME}@{AZURE_ACCOUNT}.blob.core.windows.net/mlruns/0/{run_id}/artifacts/model"
    MODEL_URI = model_uri
    logging.info(f"Loading model from {model_uri}")
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        logging.info("✅ Model loaded successfully!")
    except Exception as e:
        logging.error(f"Model load failed: {e}")

# --- FastAPI App ---
app = FastAPI(version="1.0.0")

@app.on_event("startup")
def startup_event():
    delete_old_aci_instances()  # clean old containers
    load_model()                # load latest model

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

# --- Endpoints ---
@app.get("/feature_importance")
def feature_importance():
    if model is None:
        raise HTTPException(503, "Model not loaded")
    try:
        rf_model = model._model_impl.python_model.model
        fi = rf_model.feature_importances_ if hasattr(rf_model, "feature_importances_") else None
        if fi is None:
            return {"error": "Feature importance not available for this model."}
        features = list(PatientData.model_fields.keys())
        return dict(zip(features, fi))
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/predict")
def predict(data: PatientData):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    try:
        df = pd.DataFrame([data.model_dump()])
        pred = model.predict(df)
        return {"predicted_length_of_stay": round(float(pred[0]), 2)}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "model_uri": MODEL_URI}
