import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import os
from azure.storage.blob import BlobClient, ContainerClient
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("azure").setLevel(logging.WARNING)

EXPERIMENT_NAME = "ClinicOps_LoS"
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "clinicops-dvc"
TRAIN_PATH = os.path.join('data', 'processed', 'train.csv')
TEST_PATH = os.path.join('data', 'processed', 'test.csv')

print("Starting train.py – Connection string:", "SET" if AZURE_STORAGE_CONNECTION_STRING else "MISSING")

def upload_directory_to_blob(local_path, blob_prefix):
    """Upload entire directory to Azure Blob Storage"""
    try:
        container = ContainerClient.from_connection_string(
            AZURE_STORAGE_CONNECTION_STRING,
            CONTAINER_NAME
        )
        uploaded_files = 0
        for root, dirs, files in os.walk(local_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Relative path from local_path
                relative_path = os.path.relpath(file_path, local_path)
                blob_path = f"{blob_prefix}/{relative_path}".replace("\\", "/")
                
                blob_client = container.get_blob_client(blob=blob_path)
                with open(file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                uploaded_files += 1
                logging.info(f"Uploaded: {blob_path}")
        
        logging.info(f"✅ Upload complete: {uploaded_files} files to {blob_prefix}")
        return True
    except Exception as e:
        logging.error(f"❌ Upload error: {e}")
        raise

def upload_latest_run_id(run_id):
    """Upload run ID to blob for API to fetch"""
    try:
        blob_client = BlobClient.from_connection_string(
            AZURE_STORAGE_CONNECTION_STRING,
            CONTAINER_NAME,
            "latest_model_run.txt",
        )
        blob_client.upload_blob(run_id.encode('utf-8'), overwrite=True)
        logging.info(f"✅ Uploaded run ID: {run_id}")
    except Exception as e:
        logging.error(f"❌ Run ID upload failed: {e}")
        raise

def train_and_log_model():
    try:
        logging.info("🚀 Starting model training...")
        
        # MLflow setup
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)

        # Load data
        logging.info("Loading train/test with robust parser...")
        train_df = pd.read_csv(TRAIN_PATH, low_memory=False, dtype=str)  # Initial str load (parse hatası önle)
        test_df = pd.read_csv(TEST_PATH, low_memory=False, dtype=str)
        logging.info(f"Data loaded: train={train_df.shape}, test={test_df.shape}")

        # Düzeltme: Sadece categorical kolonlara get_dummies uygula (dummy explosion önle)
        cat_cols = ["rcount", "gender", "dialysisrenalendstage", "asthma", "irondef", "pneum", "substancedependence", "psychologicaldisordermajor", "depress", "psychother", "fibrosisandother", "malnutrition", "hemo", "secondarydiagnosisnonicd9", "facid"]  # 16 categorical (lengthofstay hariç)
        # Explicit target koru
        target_col = 'lengthofstay'
        # Numeric kolonlar (target hariç)
        numeric_cols = ['hematocrit', 'neutrophils', 'sodium', 'glucose', 'bloodureanitro', 'creatinine', 'bmi', 'pulse', 'respiration']

        logging.info("Enforcing float dtype for numeric features...")
        for col in numeric_cols:
            if col in train_df.columns:
                train_df[col] = pd.to_numeric(train_df[col], errors='coerce').astype(float)
            if col in test_df.columns:
                test_df[col] = pd.to_numeric(test_df[col], errors='coerce').astype(float)

        train_target = train_df[target_col]
        test_target = test_df[target_col]

        # Categorical'leri dummies
        train_dummies = pd.get_dummies(train_df[cat_cols], drop_first=True)
        test_dummies = pd.get_dummies(test_df[cat_cols], drop_first=True)
        
        # Test dummies'i train'e align et (eksik 0)
        test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)
        
        # Numeric + dummies birleştir (target ayrı)
        train_df = pd.concat([train_df[numeric_cols], train_dummies], axis=1)
        test_df = pd.concat([test_df[numeric_cols], test_dummies], axis=1)
        
        # Target'ı df'ye ekle (son kolon)
        train_df[target_col] = train_target
        test_df[target_col] = test_target
        
        logging.info(f"Features after dummies (with target): {train_df.shape[1]}")  # Debug: ~27 bekle
        logging.info(f"Columns: {list(train_df.columns)}")  # Kolon listesi (lengthofstay son mu?)

        # Split features and target (target artık var)
        X_train = train_df.drop(target_col, axis=1)
        y_train = train_df[target_col]
        X_test = test_df.drop(target_col, axis=1)
        y_test = test_df[target_col]

        # Train model
        logging.info("Training Random Forest model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        logging.info("✅ Model training complete")

        # MLflow logging
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            
            # Log model with input example
            input_example = X_train.iloc[:1].copy()
            numeric_cols = ['hematocrit', 'neutrophils', 'sodium', 'glucose', 'bloodureanitro', 'creatinine', 'bmi', 'pulse', 'respiration']
            ohe_cols = [col for col in input_example.columns if col not in numeric_cols]
    
            input_example[ohe_cols] = input_example[ohe_cols].astype(bool)
        
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",  # This creates /model subfolder
                input_example=input_example
            )
            
            # Log metrics
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            mlflow.log_metric("train_r2", train_score)
            mlflow.log_metric("test_r2", test_score)
            
            logging.info(f"MLflow Run ID: {run_id}")
            logging.info(f"Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")

        # Download artifacts from MLflow
        logging.info("Downloading MLflow artifacts...")
        local_model_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, 
            artifact_path="model"
        )
        logging.info(f"Local model path: {local_model_path}")

        # Upload to Azure Blob
        logging.info("Uploading model to Azure Blob Storage...")
        blob_prefix = f"models/{run_id}/model"
        upload_directory_to_blob(local_model_path, blob_prefix)
        
        # Upload run ID
        upload_latest_run_id(run_id)

        # Print for GitHub Actions to capture
        print(f"Run ID: {run_id}")
        
        return run_id
        
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

# Crash hook: Tüm exception'ları stdout'a yaz
import sys
def crash_handler(type, value, tb):
    import traceback
    print(f"CRASH: {type.__name__}: {value}")
    traceback.print_exception(type, value, tb)
    sys.__excepthook__(type, value, tb)

sys.excepthook = crash_handler

if __name__ == "__main__":
    try:
        run_id = train_and_log_model()
        print(run_id)  # Final output for grep
    except Exception as e:
        logging.error(f"❌ Main error: {e}")
        exit(1)