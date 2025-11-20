# scripts/train.py - LOCAL TRACKING

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

# --- Configuration ---
# Yerel dosya sistemine kaydet
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

def train_model():
    df = load_data()
    if df is None: return

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
        log.info("Training model...")
        sk_pipeline.fit(X, y)
        
        # Modeli yerel mlruns klasörüne kaydet
        mlflow.sklearn.log_model(sk_pipeline, "model")
        
        run_id = run.info.run_id
        
        # Run ID'yi dosyaya yaz (pipeline bunu kullanacak)
        with open("latest_run_id.txt", "w") as f:
            f.write(run_id)
            
        log.info(f"✅ Model saved locally. RUN_ID: {run_id}")
        
        # CI/CD'nin ID'yi yakalaması için yazdır
        print(run_id)

if __name__ == "__main__":
    train_model()