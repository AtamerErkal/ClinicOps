import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import os
from azure.storage.blob import BlobClient, ContainerClient
from data_processing import process_data
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("azure").setLevel(logging.WARNING)

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "clinicops-dvc"

TRAIN_PATH = os.path.join('data', 'processed', 'train.csv')
TEST_PATH = os.path.join('data', 'processed', 'test.csv')

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "ClinicOps_Experiment")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")

def upload_directory_to_blob(local_path, blob_prefix):
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

def upload_latest_run_id(run_id):
    blob_client = BlobClient.from_connection_string(
        AZURE_STORAGE_CONNECTION_STRING,
        CONTAINER_NAME,
        "latest_model_run.txt",
    )
    blob_client.upload_blob(run_id, overwrite=True)

def train_and_log_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id

    logging.info(f"Tracking URI: {MLFLOW_TRACKING_URI}, Experiment ID: {experiment_id}")

    # --- DATA PROCESSING ---
    process_data()

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # --- One-hot Encoding ---
    train_df = pd.get_dummies(train_df, drop_first=True)
    test_df = pd.get_dummies(test_df, drop_first=True)
    test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

    X_train = train_df.drop('lengthofstay', axis=1)
    y_train = train_df['lengthofstay']
    X_test = test_df.drop('lengthofstay', axis=1)
    y_test = test_df['lengthofstay']

    # --- MODEL TRAINING ---
    with mlflow.start_run(experiment_id=experiment_id) as run:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_metric("train_r2", model.score(X_train, y_train))
        mlflow.log_metric("test_r2", model.score(X_test, y_test))

        run_id = run.info.run_id
        logging.info(f"MLflow run ID: {run_id}")

    # --- UPLOAD TO AZURE ---
    local_model_path = f"mlruns/{experiment_id}/{run_id}/artifacts/model"
    upload_directory_to_blob(local_model_path, f"mlruns/{experiment_id}/{run_id}/artifacts/model")
    upload_latest_run_id(run_id)

    print(run_id)  # GitHub Actions output için

if __name__ == "__main__":
    train_and_log_model()
