import os
import logging
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
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
expected_features = None

def get_latest_run_id():
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
        logging.info(f"✅ Latest Run ID: {run_id}")
        return run_id
    except Exception as e:
        logging.error(f"❌ Failed to fetch run ID: {e}")
        return RUN_ID or None

def delete_old_aci_instances():
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
    global model, MODEL_URI, expected_features
    
    run_id = get_latest_run_id()
    if not run_id:
        logging.error("❌ No Run ID found!")
        return

    model_uri = f"wasbs://{CONTAINER_NAME}@{AZURE_ACCOUNT}.blob.core.windows.net/models/{run_id}/model"
    MODEL_URI = model_uri
    
    logging.info(f"📥 Loading model from: {model_uri}")
    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = AZURE_CONN_STRING
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            logging.info(f"Attempt {attempt + 1}/{max_retries}...")
            model = mlflow.pyfunc.load_model(model_uri)
            logging.info("✅ Model loaded successfully!")
            
            # Extract features - try multiple approaches
            try:
                # Approach 1: Direct sklearn model access
                underlying_model = model._model_impl.sklearn_model
                expected_features = list(underlying_model.feature_names_in_)
                logging.info(f"✅ Got features via sklearn_model: {len(expected_features)}")
            except:
                try:
                    # Approach 2: Python model wrapper
                    underlying_model = model._model_impl.python_model
                    expected_features = list(underlying_model.feature_names_in_)
                    logging.info(f"✅ Got features via python_model: {len(expected_features)}")
                except:
                    try:
                        # Approach 3: Nested model attribute
                        underlying_model = model._model_impl.python_model.model
                        expected_features = list(underlying_model.feature_names_in_)
                        logging.info(f"✅ Got features via nested model: {len(expected_features)}")
                    except Exception as e:
                        logging.warning(f"⚠️ Could not extract features: {e}")
                        expected_features = None
            
            return
            
        except Exception as e:
            logging.error(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(30 * (attempt + 1))
            else:
                logging.error("❌ Model load failed")

app = FastAPI(title="ClinicOps API", version="1.0.0")

@app.on_event("startup")
def startup_event():
    logging.info("🚀 Starting ClinicOps API...")
    delete_old_aci_instances()
    load_model()

# Pydantic schema - accept both string and numeric values
class PatientData(BaseModel):
    # Numeric features (all float for flexibility)
    hematocrit: float
    neutrophils: float
    sodium: float
    glucose: float
    bloodureanitro: float
    creatinine: float
    bmi: float
    pulse: float
    respiration: float
    
    # String features
    rcount: str  # "0", "1", "2", "3", "4", "5+"
    gender: str  # "M", "F"
    
    # Binary features (accept int/str, but cast to str in preprocessing)
    dialysisrenalendstage: int | str = 0  # Allow both, default 0
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
    
    # Categorical numeric-like (str for dummies)
    secondarydiagnosisnonicd9: str = "0"  # "0" to "10", cast in preprocessing
    facid: str = "A"  # "A", "B", "C", "D", "E"
    
    class Config:
        # Allow extra fields (for future compatibility)
        extra = "ignore"

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_uri": MODEL_URI,
        "run_id": RUN_ID,
        "expected_features": len(expected_features) if expected_features else None
    }

@app.post("/predict")
def predict(data: PatientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_dict = data.model_dump()
        df = pd.DataFrame([input_dict])
        
        logging.info(f"📥 Input shape: {df.shape}, columns: {df.columns.tolist()}")
        
        # CRITICAL: Apply EXACT same preprocessing as train.py
        numeric_cols = ['hematocrit', 'neutrophils', 'sodium', 'glucose', 
                        'bloodureanitro', 'creatinine', 'bmi', 'pulse', 'respiration']
        cat_cols = ['rcount', 'gender', 'dialysisrenalendstage', 'asthma', 
                    'irondef', 'pneum', 'substancedependence', 
                    'psychologicaldisordermajor', 'depress', 'psychother', 
                    'fibrosisandother', 'malnutrition', 'hemo', 
                    'secondarydiagnosisnonicd9', 'facid']
    
        # Cast ALL cat_cols to str (matches train.py dtype=str)
        df[cat_cols] = df[cat_cols].astype(str)
        
        # FIX: Calculate comorbidity_score (matches train.py – sum binary as float)
        binary_cols = ['dialysisrenalendstage', 'asthma', 'irondef', 'pneum', 'substancedependence', 
                    'psychologicaldisordermajor', 'depress', 'psychother', 'fibrosisandother', 
                    'malnutrition', 'hemo']
        df['comorbidity_score'] = df[binary_cols].apply(lambda row: sum(int(val) for val in row), axis=1).astype(float)
        numeric_cols.append('comorbidity_score')  # Add to numeric list
        
        logging.info(f"Comorbidity score: {df['comorbidity_score'].iloc[0]}")  # Debug
        
        # One-hot encode categorical columns WITHOUT drop_first
        df_cat = df[cat_cols]
        df_encoded_full = pd.get_dummies(df_cat, drop_first=False, dtype=int)
        
        logging.info(f"🔄 Full encoded shape: {df_encoded_full.shape}")
        logging.info(f"Full encoded columns sample: {df_encoded_full.columns.tolist()[:10]}")
        
        # Get expected features (mevcut)
        if expected_features is None:
            if hasattr(model, '_model_impl'):
                sklearn_model = model._model_impl.python_model
                if hasattr(sklearn_model, 'model'):
                    model_features = list(sklearn_model.model.feature_names_in_)
                elif hasattr(sklearn_model, 'feature_names_in_'):
                    model_features = list(sklearn_model.feature_names_in_)
                else:
                    raise Exception("Cannot access model features")
            else:
                raise Exception("Cannot access model implementation")
        else:
            model_features = expected_features
        
        logging.info(f"🎯 Model expects {len(model_features)} features")
        logging.info(f"Expected (first 10): {model_features[:10]}")
        
        # Extract expected dummy columns from model_features (exclude numerics)
        dummy_cols = [col for col in model_features if col not in numeric_cols]
        
        # Reindex to TRAIN's dummy columns (adds missing as 0, drops extra)
        df_encoded = df_encoded_full.reindex(columns=dummy_cols, fill_value=0)
        
        logging.info(f"🔄 Reindexed dummies shape: {df_encoded.shape}")
        logging.info(f"Reindexed dummy columns sample: {df_encoded.columns.tolist()[:10]}")
        
        # Combine numeric + aligned dummies
        df_final_temp = pd.concat([df[numeric_cols].astype(float), df_encoded], axis=1)
        
        logging.info(f"🔄 Temp shape before full reindex: {df_final_temp.shape}")
        
        # Align FULL input to model_features (fill_value=0.0 for float safety)
        df_final = df_final_temp.reindex(columns=model_features, fill_value=0.0)  # ← DEĞİŞİKLİK: 0.0 float
        
        logging.info(f"✅ Final shape: {df_final.shape}")
        
        # Convert dummy columns to bool (skip numerics including comorbidity)
        for col in df_final.columns:
            if col not in numeric_cols:
                df_final[col] = df_final[col].astype(bool)
        
        # DEBUG: Log first row values
        logging.info(f"🔍 First 5 feature values: {df_final.iloc[0, :5].tolist()}")
        logging.info(f"🔍 Non-zero features: {(df_final.iloc[0] != 0).sum()}")
        
        # Verify no NaN
        if df_final.isnull().any().any():
            nan_cols = df_final.columns[df_final.isnull().any()].tolist()
            raise ValueError(f"NaN values in: {nan_cols}")
        
        # Predict (log scale)
        prediction_log = model.predict(df_final)
        
        # Inverse transform to original scale
        prediction = np.expm1(prediction_log)[0]
        
        logging.info(f"✅ Prediction (original): {prediction}")
        
        return {
            "predicted_length_of_stay": round(float(prediction), 2),
            "debug": {
                "input_features": int(df.shape[1]),
                "full_encoded_features": int(df_encoded_full.shape[1]),
                "reindexed_dummies": int(df_encoded.shape[1]),
                "model_features": int(len(model_features)),
                "missing_features": int(len(set(model_features) - set(df_final_temp.columns))),
                "extra_features": int(len(set(df_final_temp.columns) - set(model_features))),
                "non_zero_dummies": int((df_encoded != 0).sum().sum()),
                "comorbidity_score": float(df['comorbidity_score'].iloc[0])  # Debug
            }
        }
        
    except Exception as e:
            logging.error(f"❌ Prediction failed: {e}")
            import traceback
            error_trace = traceback.format_exc()
            logging.error(error_trace)
            
            raise HTTPException(
                status_code=500,
                detail={
                    "error": str(e),
                    "type": type(e).__name__,
                    "trace_preview": error_trace.split('\n')[-5:]
                }
            )

@app.get("/feature_importance")
def feature_importance():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        if hasattr(model, '_model_impl'):
            sklearn_model = model._model_impl.python_model
            if hasattr(sklearn_model, 'model'):
                rf_model = sklearn_model.model
            else:
                rf_model = sklearn_model
        else:
            raise Exception("Cannot access model")
        
        importances = rf_model.feature_importances_
        feature_names = rf_model.feature_names_in_
        
        feature_dict = dict(zip(feature_names, importances))
        sorted_features = dict(sorted(
            feature_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return {
            "total_features": len(feature_names),
            "top_10": dict(list(sorted_features.items())[:10]),
            "all_features": sorted_features
        }
        
    except Exception as e:
        logging.error(f"Feature importance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/expected_features")
def debug_features():
    """Return expected feature names"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        if expected_features:
            features = expected_features
        elif hasattr(model, '_model_impl'):
            sklearn_model = model._model_impl.python_model
            if hasattr(sklearn_model, 'model'):
                features = list(sklearn_model.model.feature_names_in_)
            else:
                features = list(sklearn_model.feature_names_in_)
        else:
            raise Exception("Cannot access features")
        
        return {
            "count": len(features),
            "features": features,
            "sample_row_format": {
                "hematocrit": "float",
                "neutrophils": "float",
                "gender": "string (e.g., 'M' or 'F')",
                "asthma": "string (e.g., 'Yes' or 'No')",
                "note": "Categorical values will be one-hot encoded automatically"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {
        "service": "ClinicOps Prediction API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "feature_importance": "GET /feature_importance",
            "debug_features": "GET /debug/expected_features",
            "docs": "GET /docs"
        }
    }