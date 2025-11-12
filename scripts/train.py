# scripts/train.py - FIXED MLFLOW TRACKING

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

# --- Azure MLflow Configuration ---
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT", "clinicopsdvcstorage2025")
AZURE_STORAGE_KEY = os.getenv("AZURE_STORAGE_KEY")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "clinicops-dvc"

# Set authentication
if AZURE_STORAGE_CONNECTION_STRING:
    os.environ['AZURE_STORAGE_CONNECTION_STRING'] = AZURE_STORAGE_CONNECTION_STRING
elif AZURE_STORAGE_KEY:
    os.environ['AZURE_STORAGE_ACCESS_KEY'] = AZURE_STORAGE_KEY

# Set MLflow tracking URI
TRACKING_URI = f"wasbs://{CONTAINER_NAME}@{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net"
mlflow.set_tracking_uri(TRACKING_URI)
log.info(f"✅ MLflow Tracking URI: {TRACKING_URI}")

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
    Loads the processed training data (train.csv) created by data_processing.py.
    """
    try:
        df = pd.read_csv(file_path)
        log.info(f"✅ Data loaded: {df.shape}")
        return df
    except Exception as e:
        log.error(f"❌ Error loading train data: {e}")
        return None

def train_model():
    df = load_data()
    if df is None:
        return

    # --- Schema Validation Check ---
    all_expected_features = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    loaded_features = set(df.drop(columns=[TARGET_COLUMN], errors='ignore').columns.tolist())
    
    missing_cols = list(all_expected_features - loaded_features)
    
    if missing_cols:
        log.error(f"❌ FATAL: Missing features: {missing_cols}")
        log.error(f"Loaded columns: {df.columns.tolist()}")
        raise ValueError(f"Feature set mismatch. Missing: {missing_cols}")
    
    log.info("✅ Schema validation passed")
    
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Preprocessing pipelines
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

    # Full pipeline
    sk_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ]) 

    model_name = "ClinicOpsLengthOfStayModel"

    try:
        # Start MLflow run
        with mlflow.start_run(run_name=f"Training Run - {model_name}") as run:
            log.info("🔄 Starting model training...")
            sk_pipeline.fit(X, y)
            log.info("✅ Model training complete")

            # Log model
            mlflow.sklearn.log_model(
                sk_pipeline, 
                "model",  # This creates artifacts/model/ structure
                registered_model_name=model_name
            )
            
            run_id = run.info.run_id
            log.info(f"✅ Model logged with RUN_ID: {run_id}")
            
            # Save run_id for deployment
            with open("latest_run_id.txt", "w") as f:
                f.write(run_id)
            
            # Print for CI/CD pipeline
            print(run_id)
            
            # Verify artifact path
            artifact_uri = mlflow.get_artifact_uri("model")
            log.info(f"📦 Artifact URI: {artifact_uri}")

    except Exception as e:
        log.error(f"❌ Training error: {e}")
        raise 

if __name__ == "__main__":
    train_model()