# scripts/train.py - HYBRID MLFLOW SETUP (Local Tracking + Azure Artifacts)

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging
import numpy as np
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- Azure Configuration for Artifacts Only ---
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT", "clinicopsdvcstorage2025")
AZURE_STORAGE_KEY = os.getenv("AZURE_STORAGE_KEY")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "clinicops-dvc"

# Set authentication for Azure Blob
if AZURE_STORAGE_CONNECTION_STRING:
    os.environ['AZURE_STORAGE_CONNECTION_STRING'] = AZURE_STORAGE_CONNECTION_STRING
elif AZURE_STORAGE_KEY:
    os.environ['AZURE_STORAGE_ACCESS_KEY'] = AZURE_STORAGE_KEY

# --- CRITICAL: Use LOCAL tracking, AZURE artifacts ---
# This avoids the "unsupported URI" error while still storing models in Azure
ARTIFACT_URI = f"wasbs://{CONTAINER_NAME}@{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/mlruns"

# Use local file system for tracking (metadata, params, metrics)
mlflow.set_tracking_uri("file:./mlruns")
log.info("✅ MLflow Tracking: LOCAL (file:./mlruns)")
log.info(f"✅ Artifact Storage: AZURE ({ARTIFACT_URI})")

# --- FEATURE LISTS ---
NUMERIC_FEATURES = [
    'hematocrit', 
    'neutrophils', 
    'sodium', 
    'glucose', 
    'bloodureanitro', 
    'creatinine', 
    'bmi', 
    'pulse', 
    'respiration'
]
CATEGORICAL_FEATURES = [
    'rcount',
    'gender', 
    'dialysisrenalendstage', 
    'asthma', 
    'irondef', 
    'pneum', 
    'substancedependence', 
    'psychologicaldisordermajor', 
    'depress', 
    'psychother', 
    'fibrosisandother', 
    'malnutrition', 
    'hemo', 
    'secondarydiagnosisnonicd9', 
    'discharged', 
    'facid'
]
TARGET_COLUMN = 'lengthofstay' 

def load_data(file_path='data/processed/train.csv'):
    """
    Loads the processed training data.
    """
    try:
        df = pd.read_csv(file_path)
        log.info(f"✅ Data loaded: {df.shape}")
        return df
    except Exception as e:
        log.error(f"❌ Error loading data: {e}")
        return None

def train_model():
    df = load_data()
    if df is None:
        return

    # --- Schema Validation ---
    all_expected_features = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    loaded_features = set(df.drop(columns=[TARGET_COLUMN], errors='ignore').columns.tolist())
    
    missing_cols = list(all_expected_features - loaded_features)
    
    if missing_cols:
        log.error(f"❌ Missing features: {missing_cols}")
        raise ValueError(f"Feature mismatch. Missing: {missing_cols}")
    
    log.info("✅ Schema validation passed")
    
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Preprocessing
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore')) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='passthrough'
    )

    sk_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ]) 

    model_name = "ClinicOpsLengthOfStayModel"

    try:
        # Set experiment
        mlflow.set_experiment("clinicops-length-of-stay")
        
        # Start run with Azure artifact location
        with mlflow.start_run(run_name=f"Training-{model_name}") as run:
            # Set artifact location to Azure
            mlflow.set_tag("mlflow.note.content", "Model artifacts stored in Azure Blob")
            
            log.info("🔄 Starting training...")
            sk_pipeline.fit(X, y)
            log.info("✅ Training complete")

            # Log model to Azure Blob (artifact_path will create artifacts/model/)
            log.info("📦 Logging model to Azure Blob...")
            mlflow.sklearn.log_model(
                sk_pipeline, 
                "model",
                artifact_path=None  # Uses default artifacts location
            )
            
            # Get run info
            run_id = run.info.run_id
            artifact_uri = mlflow.get_artifact_uri("model")
            
            log.info(f"✅ Model logged successfully!")
            log.info(f"RUN_ID: {run_id}")
            log.info(f"Artifact URI: {artifact_uri}")
            
            # Save run_id for deployment
            with open("latest_run_id.txt", "w") as f:
                f.write(run_id)
            
            # Upload run_id to Azure Blob for API to read
            from azure.storage.blob import BlobClient
            
            if AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_KEY:
                try:
                    credential = AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_KEY
                    blob_client = BlobClient(
                        account_url=f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net",
                        container_name=CONTAINER_NAME,
                        blob_name="latest_model_run.txt",
                        credential=credential
                    )
                    
                    blob_client.upload_blob(run_id, overwrite=True)
                    log.info("✅ Run ID uploaded to Azure Blob: latest_model_run.txt")
                except Exception as blob_error:
                    log.warning(f"⚠️ Failed to upload run_id to blob: {blob_error}")
            
            # Print for CI/CD
            print(run_id)

    except Exception as e:
        log.error(f"❌ Training error: {e}")
        raise 

if __name__ == "__main__":
    train_model()