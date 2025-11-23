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
    try:
        container = ContainerClient.from_connection_string(
            AZURE_STORAGE_CONNECTION_STRING,
            CONTAINER_NAME
        )
        uploaded_files = 0
        for root, dirs, files in os.walk(local_path):
            for file in files:
                file_path = os.path.join(root, file)
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
        
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)

        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
        logging.info(f"Data loaded: train={train_df.shape}, test={test_df.shape}")

        train_df = pd.get_dummies(train_df, drop_first=True)
        test_df = pd.get_dummies(test_df, drop_first=True)
        test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

        X_train = train_df.drop('lengthofstay', axis=1)
        y_train = train_df['lengthofstay']
        X_test = test_df.drop('lengthofstay', axis=1)
        y_test = test_df['lengthofstay']

        logging.info("Training Random Forest model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        logging.info("✅ Model training complete")

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            
            input_example = X_train.iloc[:1]
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example
            )
            
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            mlflow.log_metric("train_r2", train_score)
            mlflow.log_metric("test_r2", test_score)
            
            logging.info(f"MLflow Run ID: {run_id}")
            logging.info(f"Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")

        logging.info("Downloading MLflow artifacts...")
        local_model_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, 
            artifact_path="model"
        )
        logging.info(f"Local model path: {local_model_path}")

        logging.info("Uploading model to Azure Blob Storage...")
        blob_prefix = f"models/{run_id}/model"
        upload_directory_to_blob(local_model_path, blob_prefix)
        
        upload_latest_run_id(run_id)

        print(f"Run ID: {run_id}")
        
        return run_id
        
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    try:
        run_id = train_and_log_model()
        print(run_id)
    except Exception as e:
        logging.error(f"❌ Main error: {e}")
        exit(1)