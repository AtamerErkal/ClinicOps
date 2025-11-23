import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os
from azure.storage.blob import BlobClient, ContainerClient
from mlflow import artifacts
import logging

logging.getLogger("azure").setLevel(logging.WARNING)

# Dinamik experiment
EXPERIMENT_NAME = "ClinicOps_LoS"

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "clinicops-dvc"

TRAIN_PATH = os.path.join('data', 'processed', 'train.csv')
TEST_PATH = os.path.join('data', 'processed', 'test.csv')

print("Starting train.py – Env check: AZURE_STORAGE_CONNECTION_STRING set? ", "Yes" if AZURE_STORAGE_CONNECTION_STRING else "No")  # Debug print

def upload_directory_to_blob(local_path, blob_prefix):
    try:
        container = ContainerClient.from_connection_string(
            AZURE_STORAGE_CONNECTION_STRING,
            CONTAINER_NAME
        )
        for root, dirs, files in os.walk(local_path):
            for file in files:
                file_path = os.path.join(root, file)
                blob_path = os.path.join(blob_prefix, file_path.replace(local_path, "").lstrip("/\\"))
                blob_path = blob_path.replace("\\", "/")
                blob_client = container.get_blob_client(blob=blob_path)
                with open(file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
        print("Upload success")  # Debug
    except Exception as e:
        print(f"Upload error: {e}")  # Debug crash
        raise

def upload_latest_run_id(run_id):
    blob_client = BlobClient.from_connection_string(
        AZURE_STORAGE_CONNECTION_STRING,
        CONTAINER_NAME,
        "latest_model_run.txt",
    )
    blob_client.upload_blob(run_id, overwrite=True)

def train_and_log_model():
    try:
        print("train_and_log_model started")  # Debug
        # MLflow tracking
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)

        # Load train/test (preprocessed raw)
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
        print(f"Data loaded: train shape {train_df.shape}, test shape {test_df.shape}")  # Debug

        # --- Categorical → Numeric (tüm cat'leri get_dummies ile) ---
        train_df = pd.get_dummies(train_df, drop_first=True)
        test_df = pd.get_dummies(test_df, drop_first=True)
        test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

        X_train = train_df.drop('lengthofstay', axis=1)
        y_train = train_df['lengthofstay']
        X_test = test_df.drop('lengthofstay', axis=1)
        y_test = test_df['lengthofstay']

        # --- MODEL TRAINING ---
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        print("Model fitted")  # Debug

        # --- MLflow Logging ---
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.sklearn.log_model(model, "model")  # artifact_path yerine "model"

        # --- AZURE UPLOAD ---
        local_model_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
        upload_directory_to_blob(local_model_path, f"models/{run_id}/model")  # Sabit path
        upload_latest_run_id(run_id)

        logging.info("✅ Model uploaded successfully.")
        logging.info(f"Run ID: {run_id}")
        print(f"Run ID: {run_id}")  # Stdout print, grep için
        return run_id
    except Exception as e:
        print(f"train_and_log_model error: {e}")  # Crash yakala
        raise

if __name__ == "__main__":
    try:
        run_id = train_and_log_model()
        print(run_id)  # Tekrar print, capture için
    except Exception as e:
        print(f"Main error: {e}")  # Ana crash yakala