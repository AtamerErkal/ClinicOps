import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# DÜZELTME: mean_squared_error'ı kullanıyoruz
from sklearn.metrics import r2_score, mean_squared_error 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import mlflow
import mlflow.sklearn

def train_model():
    # --- Load Data ---
    try:
        df = pd.read_csv('data/processed/train.csv')
    except FileNotFoundError:
        print("Error: Processed training data not found. Run data_processing.py first.")
        raise

    X = df.drop('lengthofstay', axis=1, errors='ignore')
    y = df['lengthofstay']

    # --- Feature Engineering & Preprocessing ---
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop' 
    )

    # --- Model Definition ---
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    
    sk_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    
    model_name = "ClinicOpsLengthOfStayModel"

    # --- MLflow Tracking & Training ---
    try:
        with mlflow.start_run(run_name=f"Training Run - {model_name}") as run:
            
            # Train the model
            sk_pipeline.fit(X, y)

            # Make predictions and calculate metrics
            y_pred = sk_pipeline.predict(X)
            r2 = r2_score(y, y_pred)
            
            # HATA DÜZELTİLDİ: MSE hesapla, sonra np.sqrt ile RMSE'yi al
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse) 

            # Log parameters and metrics
            mlflow.log_param("model_type", "RandomForestRegressor")
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("rmse", rmse) # Artık doğru hesaplanmış RMSE
            print(f"R2 Score: {r2}")
            
            mlflow.sklearn.log_model(
                sk_pipeline, 
                "model", 
                registered_model_name=model_name
            )
            
            # CRITICAL FIX: Print the Run ID for the CI/CD pipeline to capture
            print(run.info.run_id) 

    except Exception as e:
        print(f"An error occurred during training or MLflow logging: {e}")
        raise 

if __name__ == "__main__":
    train_model()