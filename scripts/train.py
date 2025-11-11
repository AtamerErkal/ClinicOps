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

# Set MLflow tracking URI (Placeholder: Replace with your actual tracking setup if necessary)
# For Azure Blob Storage (if configured via DVC), this step might be optional but is good practice.
# os.environ["MLFLOW_TRACKING_URI"] = "azureml://azdops.azureml.net/mlflows/v1.0" 

def train_model():
    # --- Load Data ---
    try:
        # Assuming data_processing.py saves the final processed data here
        df = pd.read_csv('data/processed/train.csv')
    except FileNotFoundError:
        print("Error: Processed training data not found. Run data_processing.py first.")
        # We raise an error instead of returning, to fail the GitHub action job
        raise

    # Define features (X) and target (y)
    # Target is lengthofstay, which must be dropped from features.
    X = df.drop('lengthofstay', axis=1, errors='ignore')
    y = df['lengthofstay']

    # --- Feature Engineering & Preprocessing ---
    # Determine features based on data types in the processed data
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        # Drop columns not used (like eid, vdate, facid if they weren't dropped earlier)
        remainder='drop' 
    )

    # --- Model Definition ---
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Create the full modeling pipeline (Fixes F821 undefined name 'sk_pipeline')
    sk_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    
    # Define model name (Fixes F821 undefined name 'model_name')
    model_name = "ClinicOpsLengthOfStayModel"

    # --- MLflow Tracking & Training ---
    try:
        # Start MLflow run (Fixes F821 undefined name 'model_name')
        with mlflow.start_run(run_name=f"Training Run - {model_name}") as run:
            
            # Train the model
            sk_pipeline.fit(X, y)

            # Make predictions and calculate metrics
            y_pred = sk_pipeline.predict(X)
            r2 = r2_score(y, y_pred)
            rmse = mean_squared_error(y, y_pred, squared=False)

            # Log parameters and metrics
            mlflow.log_param("model_type", "RandomForestRegressor")
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("rmse", rmse)
            print(f"R2 Score: {r2}")
            
            # Log the model (Fixes F821 undefined name 'sk_pipeline')
            mlflow.sklearn.log_model(
                sk_pipeline, 
                "model", 
                registered_model_name=model_name
            )
            
            # --- CRITICAL FIX: Print the Run ID for the CI/CD pipeline to capture ---
            # This is the last thing printed to stdout and is captured by GitHub Actions
            print(run.info.run_id) 

    except Exception as e:
        print(f"An error occurred during training or MLflow logging: {e}")
        # Raise the exception to ensure the GitHub Action job fails on error
        raise 

if __name__ == "__main__":
    train_model()