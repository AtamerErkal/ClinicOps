# scripts/train.py - AML MLflow Integration + Registry

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

# AML Configuration
AML_TRACKING_URI = os.getenv("AML_TRACKING_URI")
AML_RESOURCE_GROUP = os.getenv("AML_RESOURCE_GROUP")
AML_WORKSPACE_NAME = os.getenv("AML_WORKSPACE_NAME")

# Verify credentials
if AZURE_STORAGE_CONNECTION_STRING:
    log.info("✅ Using AZURE_STORAGE_CONNECTION_STRING")
elif AZURE_STORAGE_KEY:
    log.info("✅ Using AZURE_STORAGE_ACCESS_KEY")
else:
    log.warning("⚠️ No Azure credentials found! Training will work but upload may fail.")

# Set MLflow Tracking URI to AML (if set, else local fallback)
if AML_TRACKING_URI:
    mlflow.set_tracking_uri(AML_TRACKING_URI)
    log.info(f"✅ MLflow Tracking: AML Workspace ({AML_TRACKING_URI})")
else:
    mlflow.set_tracking_uri("file:./mlruns")
    log.info("✅ MLflow Tracking: LOCAL (file:./mlruns)")

log.info("✅ Artifacts will be manually uploaded to Azure")

# --- FEATURE LISTS --- (aynı)
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

def upload_to_azure_blob(local_dir, run_id):
    """Upload model artifacts to Azure Blob Storage""" (mevcut kod aynı)
    try:
        from azure.storage.blob import ContainerClient
        from azure.core.credentials import AzureNamedKeyCredential
        
        if AZURE_STORAGE_CONNECTION_STRING:
            credential = AZURE_STORAGE_CONNECTION_STRING
            log.info("✅ Using CONNECTION_STRING")
        elif AZURE_STORAGE_KEY:
            credential = AzureNamedKeyCredential(
                name=AZURE_STORAGE_ACCOUNT, 
                key=AZURE_STORAGE_KEY
            )
            log.info("✅ Using ACCOUNT KEY with NamedKeyCredential")
        else:
            log.warning("⚠️ No Azure credentials - skipping upload")
            return False
            
        container_client = ContainerClient(
            account_url=f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net",
            container_name=CONTAINER_NAME,
            credential=credential
        )
        
        uploaded_count = 0
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_dir)
                
                blob_name = f"mlruns/{run_id}/artifacts/model/{relative_path}"
                
                with open(local_path, 'rb') as data:
                    blob_client = container_client.get_blob_client(blob_name)
                    blob_client.upload_blob(data, overwrite=True)
                    uploaded_count += 1
                    
        log.info(f"✅ Uploaded {uploaded_count} files to Azure Blob")
        return True
        
    except Exception as e:
        log.error(f"❌ Error uploading to Azure: {e}")
        return False

def train_model():
    df = load_data()
    if df is None:
        return

    # Schema Validation (aynı)
    all_expected_features = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    loaded_features = set(df.drop(columns=[TARGET_COLUMN], errors='ignore').columns.tolist())
    missing_cols = list(all_expected_features - loaded_features)
    
    if missing_cols:
        log.error(f"❌ Missing features: {missing_cols}")
        raise ValueError(f"Missing: {missing_cols}")
    
    log.info("✅ Schema validation passed")
    
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Build pipeline (aynı)
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
        
        # Start MLflow run
        with mlflow.start_run(run_name="Training-LengthOfStay") as run:
            
            log.info("🔄 Starting training...")
            sk_pipeline.fit(X, y)
            log.info("✅ Training complete")

            # Log model locally (AML artifacts'ı otomatik yönetir)
            log.info("📦 Logging model...")
            mlflow.sklearn.log_model(
                sk_model=sk_pipeline,
                artifact_path="model"
            )
            
            run_id = run.info.run_id
            local_model_path = f"mlruns/{run.info.experiment_id}/{run_id}/artifacts/model" if not AML_TRACKING_URI else f"runs:/{run_id}/model"
            
            log.info(f"✅ Model logged!")
            log.info(f"RUN_ID: {run_id}")
            log.info(f"Local path: {local_model_path}")
            
            # YENİ: Model'i AML Registry'ye Register Et (AML backend ile)
            model_name = "length_of_stay_model"
            model_uri = f"runs:/{run_id}/model"
            try:
                registered_model_version = mlflow.register_model(model_uri, model_name)
                log.info(f"✅ Model registered in AML: {model_name} v{registered_model_version.version}")
            except Exception as reg_error:
                log.warning(f"⚠️ Registry registration failed: {reg_error}")
            
            # Upload to Azure (AML artifacts'ı da Blob'a yedekle)
            log.info("☁️ Uploading model to Azure Blob...")
            upload_success = upload_to_azure_blob(local_model_path, run_id)
            
            if upload_success:
                log.info("✅ Model uploaded to Azure!")
            else:
                log.warning("⚠️ Model saved locally but Azure upload failed")
            
            # Save run_id (aynı)
            with open("latest_run_id.txt", "w") as f:
                f.write(run_id)
            log.info("✅ Run ID saved to latest_run_id.txt")
            
            # Upload run_id pointer to Azure (aynı)
            try:
                from azure.storage.blob import BlobClient
                from azure.core.credentials import AzureNamedKeyCredential
                
                if AZURE_STORAGE_CONNECTION_STRING:
                    credential = AZURE_STORAGE_CONNECTION_STRING
                elif AZURE_STORAGE_KEY:
                    credential = AzureNamedKeyCredential(name=AZURE_STORAGE_ACCOUNT, key=AZURE_STORAGE_KEY)
                else:
                    raise ValueError("No credentials")
                    
                blob_client = BlobClient(
                    account_url=f"https://{AZURE_STORAGE_ACCOUNT}.blob.core.windows.net",
                    container_name=CONTAINER_NAME,
                    blob_name="latest_model_run.txt",
                    credential=credential
                )
                blob_client.upload_blob(run_id, overwrite=True)
                log.info("✅ Run ID uploaded to Azure: latest_model_run.txt")
            except Exception as e:
                log.warning(f"⚠️ Failed to upload run_id pointer: {e}")
            
            # Print for CI/CD
            print(run_id)

    except Exception as e:
        log.error(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        raise 

if __name__ == "__main__":
    train_model()