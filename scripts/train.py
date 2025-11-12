# scripts/train.py - WORKING AZURE BLOB SETUP

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging
import os
from azure.storage.blob import BlobClient

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- Azure Configuration ---
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT", "clinicopsdvcstorage2025")
AZURE_STORAGE_KEY = os.getenv("AZURE_STORAGE_KEY")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "clinicops-dvc"

# Set authentication
if AZURE_STORAGE_CONNECTION_STRING:
    os.environ['AZURE_STORAGE_CONNECTION_STRING'] = AZURE_STORAGE_CONNECTION_STRING
elif AZURE_STORAGE_KEY:
    os.environ['AZURE_STORAGE_ACCESS_KEY'] = AZURE_STORAGE_KEY

# --- MLflow Setup: Local tracking BUT specify Azure artifact location per run ---
mlflow.set_tracking_uri("file:./mlruns")
log.info("✅ MLflow Tracking: LOCAL (file:./mlruns)")

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

    model_name = "ClinicOpsLengthOfStayModel"

    try:
        mlflow.set_experiment("clinicops-length-of-stay")
        
        # CRITICAL: Set artifact location to Azure BEFORE starting run
        azure_artifact_root = f"wasbs://{CONTAINER_NAME}@{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net/mlruns"
        
        with mlflow.start_run(run_name=f"Training-{model_name}") as run:
            # Set artifact location explicitly
            client = mlflow.tracking.MlflowClient()
            run_id = run.info.run_id
            
            # Update run to use Azure for artifacts
            client.set_tag(run_id, "mlflow.note.content", "Artifacts in Azure Blob")
            
            log.info("🔄 Starting training...")
            sk_pipeline.fit(X, y)
            log.info("✅ Training complete")

            # Log model - this will go to local mlruns by default
            log.info("📦 Logging model locally first...")
            mlflow.sklearn.log_model(sk_pipeline, "model")
            
            # Get local artifact path
            local_artifact_path = mlflow.get_artifact_uri("model")
            log.info(f"Local artifact path: {local_artifact_path}")
            
            # Now manually upload to Azure
            log.info("☁️ Uploading model to Azure Blob...")
            upload_model_to_azure(run_id, sk_pipeline)
            
            log.info(f"✅ Model logged successfully!")
            log.info(f"RUN_ID: {run_id}")
            
            # Save run_id locally
            with open("latest_run_id.txt", "w") as f:
                f.write(run_id)
            
            # Upload run_id to Azure
            upload_run_id_to_azure(run_id)
            
            # Print for CI/CD
            print(run_id)

    except Exception as e:
        log.error(f"❌ Training error: {e}")
        raise 

def upload_model_to_azure(run_id, model):
    """Manually upload model to Azure Blob Storage in MLflow structure"""
    import tempfile
    import shutil
    
    try:
        # Save model to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model")
            mlflow.sklearn.save_model(model, model_path)
            
            # Upload each file to Azure Blob
            from azure.storage.blob import ContainerClient
            
            credential = AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_KEY
            container_client = ContainerClient(
                account_url=f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net",
                container_name=CONTAINER_NAME,
                credential=credential
            )
            
            # Upload all files in model directory
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    local_file = os.path.join(root, file)
                    # Create blob path: mlruns/{run_id}/artifacts/model/{file}
                    relative_path = os.path.relpath(local_file, tmpdir)
                    blob_name = f"mlruns/{run_id}/artifacts/{relative_path}"
                    
                    with open(local_file, "rb") as data:
                        blob_client = container_client.get_blob_client(blob_name)
                        blob_client.upload_blob(data, overwrite=True)
                        log.info(f"  ✓ Uploaded: {blob_name}")
            
            log.info("✅ Model uploaded to Azure successfully!")
            
    except Exception as e:
        log.error(f"❌ Error uploading to Azure: {e}")
        raise

def upload_run_id_to_azure(run_id):
    """Upload run_id pointer file to Azure"""
    try:
        credential = AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_KEY
        blob_client = BlobClient(
            account_url=f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net",
            container_name=CONTAINER_NAME,
            blob_name="latest_model_run.txt",
            credential=credential
        )
        
        blob_client.upload_blob(run_id, overwrite=True)
        log.info("✅ Run ID uploaded to Azure: latest_model_run.txt")
    except Exception as e:
        log.warning(f"⚠️ Failed to upload run_id: {e}")

if __name__ == "__main__":
    train_model()