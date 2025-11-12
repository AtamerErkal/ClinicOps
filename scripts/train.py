# scripts/train.py - FINAL FIX for 'rcount' and SyntaxError

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- FEATURE LISTS ---
NUMERIC_FEATURES = [
    # 'rcount' was removed from this list
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
    'rcount', # 'rcount' is now correctly treated as categorical
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
# ----------------------------------------------------------------------

def load_data(file_path='data/processed/train.csv'):
    """
    Loads the processed training data (train.csv) created by data_processing.py.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        log.error(f"Error loading train data: [Errno 2] No such file or directory: '{file_path}'. ")
        return None

def train_model():
    df = load_data()
    if df is None:
        return

    # --- Schema Validation Check ---
    all_expected_features = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    # Check against the columns *after* dropping the target
    loaded_features = set(df.drop(columns=[TARGET_COLUMN], errors='ignore').columns.tolist())
    
    missing_cols = list(all_expected_features - loaded_features)
    
    if missing_cols:
        log.error(f"FATAL: Feature set mismatch. Missing features in processed data: {missing_cols}")
        log.error(f"Loaded columns: {df.columns.tolist()}")
        raise ValueError(f"Feature set mismatch. Missing features: {missing_cols}")
    # --- END Schema Validation Code ---
    
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN] # Regression target

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
                                  ('regressor', LinearRegression())]) 

    model_name = "ClinicOpsLengthOfStayModel"

    try:
        # --- SYNTAX ERROR FIX: Added '=' after run_name ---
        with mlflow.start_run(run_name=f"Training Run - {model_name}") as run:
            log.info("Starting model training...")
            sk_pipeline.fit(X, y)
            log.info("Model training complete.")

            # Log model, params, and metrics 
            mlflow.sklearn.log_model(
                sk_pipeline, 
                "model", 
                registered_model_name=model_name
            )
            
            # --- Pseudo Registry Logic ---
            run_id = run.info.run_id
            with open("latest_run_id.txt", "w") as f:
                f.write(run_id)
            
            # This print is essential for the CI/CD pipeline
            print(run_id) 

    except Exception as e:
        log.error(f"An error occurred during training or MLflow logging: {e}")
        raise 

if __name__ == "__main__":
    train_model()