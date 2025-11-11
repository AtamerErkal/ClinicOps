# scripts/train.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import mlflow
import mlflow.sklearn
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def train_model():
    # --- Load Data ---
    try:
        df = pd.read_csv('data/processed/train.csv')
    except FileNotFoundError:
        log.error("Processed training data not found. Run data_processing.py first.")
        raise

    X = df.drop('lengthofstay', axis=1, errors='ignore')
    y = df['lengthofstay']

    # --- Feature Engineering & Preprocessing ---
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            
            # OPTIMIZATION: Prevents feature explosion (fixes 10-min hang)
            ('cat', OneHotEncoder(handle_unknown='ignore', max_categories=50), categorical_features)
        ],
        remainder='drop' 
    )

    # --- Model Definition ---
    # OPTIMIZATION: Reduces tree count for faster training (fixes 10-min hang)
    regressor = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1) # n_jobs=-1 uses all cores
    
    sk_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    
    model_name = "ClinicOpsLengthOfStayModel"

    # --- MLflow Tracking & Training ---
    try:
        with mlflow.start_run(run_name=f"Training Run - {model_name}") as run:
            
            log.info("Starting model training (Optimized)...")
            sk_pipeline.fit(X, y)
            log.info("Model training finished.")

            y_pred = sk_pipeline.predict(X)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse) 

            # Log parameters and metrics (these go to MLflow, not stdout)
            mlflow.log_param("model_type", "RandomForestRegressor")
            mlflow.log_param("n_estimators", 50)
            mlflow.log_param("max_categories", 50)
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("rmse", rmse)
            
            mlflow.sklearn.log_model(
                sk_pipeline, 
                "model", 
                registered_model_name=model_name
            )
            
            # --- CRITICAL FIX: This MUST be the ONLY 'print' statement ---
            # This output is captured by GitHub Actions to set the Run ID.
            print(run.info.run_id) 

    except Exception as e:
        log.error(f"An error occurred during training or MLflow logging: {e}")
        raise 

if __name__ == "__main__":
    train_model()