# scripts/train.py

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Column definitions (Assuming these are correct based on your previous work)
NUMERIC_FEATURES = ['rcount', 'age', 'eclaim', 'pridx', 'sdimd', 'plos', 'clmds']
CATEGORICAL_FEATURES = ['gender', 'dialysis', 'mcd', 'ecodes', 'hmo', 'health', 
                        'procedure', 'pcode', 'zid', 'disch', 'orproc', 'comorb', 
                        'diag', 'ipros', 'DRG', 'last', 'PG', 'payer', 'primaryphy']
TARGET_COLUMN = 'lengthofstay'

# --- CRITICAL FIX: Changed file_path to 'data/processed/train.csv' ---
def load_data(file_path='data/processed/train.csv'):
    """
    Loads the processed training data (train.csv) created by data_processing.py.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        # Improved error message to reflect the exact file not found
        log.error(f"Error loading train data: [Errno 2] No such file or directory: '{file_path}'. "
                  f"Ensure data_processing.py created the train.csv file.")
        return None

def train_model():
    df = load_data()
    if df is None:
        return

    # Assuming 'long_stay' column has been created (0 or 1)
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

    # Full pipeline: Preprocessor + Model
    sk_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', LogisticRegression(solver='liblinear'))])

    model_name = "ClinicOpsLengthOfStayModel"

    try:
        with mlflow.start_run(run_name=f"Training Run - {model_name}") as run:
            log.info("Starting model training...")
            sk_pipeline.fit(X, y)
            log.info("Model training complete.")

            # Log model, params, and metrics (metrics logging omitted for brevity)
            mlflow.sklearn.log_model(
                sk_pipeline, 
                "model", 
                registered_model_name=model_name
            )
            
            # --- Pseudo Registry Logic ---
            # 1. Get the current Run ID
            run_id = run.info.run_id
            
            # 2. Write the Run ID to a local file
            with open("latest_run_id.txt", "w") as f:
                f.write(run_id)
            
            # CRITICAL: Print the Run ID to the console for the GitHub Action to capture
            print(run_id) 

    except Exception as e:
        log.error(f"An error occurred during training or MLflow logging: {e}")
        raise 

if __name__ == "__main__":
    train_model()