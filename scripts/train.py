# scripts/train.py - LOCAL TRACKING + AZURE UPLOAD

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging
import os

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Azure credentials
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT", "clinicopsdvcstorage2025")
AZURE_STORAGE_KEY = os.getenv("AZURE_STORAGE_KEY")
CONTAINER_NAME = "clinicops-dvc"

# Use local tracking
mlflow.set_tracking_uri("file:./mlruns")

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
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        log.error(f"Error: {e}")
        return None

def upload_model_to_azure(local_model_path, run_id):
    """Upload model files to Azure Blob Storage"""
    if not AZURE_STORAGE_KEY:
        log.warning("⚠️ No Azure credentials - skipping upload")
        return False
        
    try:
        from azure.storage.blob import ContainerClient
        
        container_client = ContainerClient(
            account_url=f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net",
            container_name=CONTAINER_NAME,
            credential=AZURE_STORAGE_KEY
        )
        
        uploaded_count = 0
        for root, dirs, files in os.walk(local_model_path):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_model_path)
                
                # Azure path: mlruns/0/{run_id}/artifacts/model/{file}
                blob_name = f"mlruns/0/{run_id}/artifacts/model/{relative_path}"
                
                with open(local_path, 'rb') as data:
                    blob_client = container_client.get_blob_client(blob_name)
                    blob_client.upload_blob(data, overwrite=True)
                    uploaded_count += 1
                    
        log.info(f"✅ Uploaded {uploaded_count} files to Azure")
        return True
        
    except Exception as e:
        log.error(f"❌ Azure upload failed: {e}")
        return False

def upload_run_id_pointer(run_id):
    """Upload run_id pointer file to Azure"""
    if not AZURE_STORAGE_KEY:
        return False
        
    try:
        from azure.storage.blob import BlobClient
        
        blob_client = BlobClient(
            account_url=f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net",
            container_name=CONTAINER_NAME,
            blob_name="latest_model_run.txt",
            credential=AZURE_STORAGE_KEY
        )
        
        blob_client.upload_blob(run_id, overwrite=True)
        log.info("✅ Run ID pointer uploaded to Azure")
        return True
        
    except Exception as e:
        log.error(f"❌ Pointer upload failed: {e}")
        return False

def train_model():
    df = load_data()
    if df is None: 
        return

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

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

    mlflow.set_experiment("clinicops-length-of-stay")
    
    with mlflow.start_run() as run:
        log.info("🔄 Training model...")
        sk_pipeline.fit(X, y)
        
        # Log model and get model info
        model_info = mlflow.sklearn.log_model(sk_pipeline, "model")
        
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        
        # Try multiple possible paths
        possible_paths = [
            f"mlruns/{experiment_id}/{run_id}/artifacts/model",
            f"mlruns/{experiment_id}/models/{model_info.model_uri.split('/')[-1]}/artifacts" if hasattr(model_info, 'model_uri') else None,
        ]
        
        local_model_path = None
        for path in possible_paths:
            if path and os.path.exists(path):
                local_model_path = path
                break
        
        if not local_model_path:
            # Find it manually
            models_dir = f"mlruns/{experiment_id}/models"
            if os.path.exists(models_dir):
                # Get latest model
                model_dirs = [d for d in os.listdir(models_dir) if d.startswith('m-')]
                if model_dirs:
                    latest_model = sorted(model_dirs, key=lambda x: os.path.getmtime(os.path.join(models_dir, x)))[-1]
                    local_model_path = f"{models_dir}/{latest_model}/artifacts"
        
        log.info(f"✅ Model trained. RUN_ID: {run_id}")
        log.info(f"📁 Local path: {local_model_path}")
        
        # Verify model exists locally
        if local_model_path and os.path.exists(local_model_path):
            file_count = sum([len(files) for r, d, files in os.walk(local_model_path)])
            log.info(f"📦 Found {file_count} files to upload")
            
            # Upload to Azure
            log.info("☁️ Uploading model to Azure...")
            upload_model_to_azure(local_model_path, run_id)
        else:
            log.error(f"❌ Model path not found!")
            if local_model_path:
                log.error(f"Tried: {local_model_path}")
        
        # Upload pointer
        upload_run_id_pointer(run_id)
        
        # Write Run ID to file for pipeline
        with open("latest_run_id.txt", "w") as f:
            f.write(run_id)
            
        # Print for CI/CD
        print(run_id)

if __name__ == "__main__":
    train_model()