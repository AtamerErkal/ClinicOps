# scripts/train.py - LOCAL TRACKING VERSION

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

# --- Configuration ---
# We use LOCAL tracking. The CI/CD pipeline will handle the upload to Azure.
mlflow.set_tracking_uri("file:./mlruns")
log.info("✅ MLflow Tracking set to LOCAL: file:./mlruns")

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
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        log.error(f"❌ Error loading data: {e}")
        return None

def train_model():
    df = load_data()
    if df is None: return

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Pipeline
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
        
        with mlflow.start_run() as run:
            log.info("🔄 Starting training...")
            sk_pipeline.fit(X, y)
            log.info("✅ Training complete")

            # Log model locally
            mlflow.sklearn.log_model(sk_pipeline, "model")
            
            # Save Run ID to a file for the CI/CD pipeline to use
            run_id = run.info.run_id
            with open("latest_run_id.txt", "w") as f:
                f.write(run_id)
                
            log.info(f"✅ Model saved locally. RUN_ID: {run_id}")
            
            # CRITICAL: Print RUN_ID so GitHub Actions can capture it
            print(run_id)

    except Exception as e:
        log.error(f"❌ Training error: {e}")
        raise 

if __name__ == "__main__":
    train_model()