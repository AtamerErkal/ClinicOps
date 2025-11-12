# scripts/train.py - FINAL SIMPLE VERSION

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- Azure Configuration ---
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT", "clinicopsdvcstorage2025")
AZURE_STORAGE_KEY = os.getenv("AZURE_STORAGE_KEY")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "clinicops-dvc"

# Set authentication for MLflow
if AZURE_STORAGE_CONNECTION_STRING:
    os.environ['AZURE_STORAGE_CONNECTION_STRING'] = AZURE_STORAGE_CONNECTION_STRING
    log.info("✅ Using AZURE_STORAGE_CONNECTION_STRING")
elif AZURE_STORAGE_KEY:
    os.environ['AZURE_STORAGE_ACCESS_KEY'] = AZURE_STORAGE_KEY
    log.info("✅ Using AZURE_STORAGE_ACCESS_KEY")
else:
    log.error("❌ No Azure credentials found!")

# Set MLflow tracking URI to Azure Blob
TRACKING_URI = f"wasbs://{CONTAINER_NAME}@{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net"
mlflow.set_tracking_uri(TRACKING_URI)
log.info(f"✅ MLflow Tracking URI: {TRACKING_URI}")

# --- FEATURE LISTS ---
NUMERIC_FEATURES = [
    'hematocrit', 'neutrophils', 'sodium', 'glucose', 
    'bloodureanitro', 'creatinine', 'bmi', 'pulse', 'respiration'
]
CATEGORICAL_FEATURES = [
    'rcount', 'gender', 'dialysisrenalendstage', 'asthma', 'irondef', 
    'pneum', 'substancedependence', 'psychologicaldisordermajor', 
    'depress', 'psychother', 'fibrosisandother', 'malnutrition', 
    'hemo', 'secondarydiagnosisnonicd9', 'discharged', 'facid'
]
TARGET_COLUMN = 'lengthofstay' 

def load_data(file_path='data/processed/train.csv'):
    """Loads the processed training data."""
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

    # Schema Validation
    all_expected_features = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    loaded_features = set(df.drop(columns=[TARGET_COLUMN], errors='ignore').columns.tolist())
    missing_cols = list(all_expected_features - loaded_features)
    
    if missing_cols:
        log.error(f"❌ Missing features: {missing_cols}")
        raise ValueError(f"Missing: {missing_cols}")
    
    log.info("✅ Schema validation passed")
    
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Build pipeline
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
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

    try:
        # Set experiment
        mlflow.set_experiment("clinicops-length-of-stay")
        
        # Start MLflow run - everything goes to Azure automatically
        with mlflow.start_run(run_name="Training-LengthOfStay") as run:
            
            log.info("🔄 Starting training...")
            sk_pipeline.fit(X, y)
            log.info("✅ Training complete")

            # Log model to Azure Blob
            # CRITICAL: NO registered_model_name parameter
            log.info("📦 Logging model to Azure Blob...")
            mlflow.sklearn.log_model(
                sk_model=sk_pipeline,
                artifact_path="model"
            )
            
            run_id = run.info.run_id
            artifact_uri = mlflow.get_artifact_uri("model")
            
            log.info(f"✅ Model logged successfully!")
            log.info(f"RUN_ID: {run_id}")
            log.info(f"Artifact URI: {artifact_uri}")
            
            # Save run_id locally for CI/CD
            with open("latest_run_id.txt", "w") as f:
                f.write(run_id)
            
            # Also try to upload run_id pointer to Azure
            try:
                from azure.storage.blob import BlobClient
                
                # Use same credentials
                credential = AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_KEY
                
                if credential:
                    blob_client = BlobClient(
                        account_url=f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net",
                        container_name=CONTAINER_NAME,
                        blob_name="latest_model_run.txt",
                        credential=credential
                    )
                    
                    blob_client.upload_blob(run_id, overwrite=True)
                    log.info("✅ Run ID uploaded to Azure: latest_model_run.txt")
                else:
                    log.warning("⚠️ No credentials for run_id upload")
                    
            except Exception as blob_error:
                log.warning(f"⚠️ Failed to upload run_id (non-critical): {blob_error}")
            
            # Print for CI/CD pipeline
            print(run_id)

    except Exception as e:
        log.error(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        raise 

if __name__ == "__main__":
    train_model()