import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import os
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
TRAIN_PATH = os.path.join('data', 'processed', 'train.csv')
TEST_PATH = os.path.join('data', 'processed', 'test.csv')

def train_model(train_path, test_path):
    """
    Loads processed data, trains a model, and logs
    everything with MLflow.
    """
    try:
        logging.info("Loading processed data...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        target = 'lengthofstay' 
        
        # Define features (X) and target (y)
        X_train = train_df.drop(target, axis=1)
        y_train = train_df[target]
        X_test = test_df.drop(target, axis=1)
        y_test = test_df[target]

        # Identify categorical features for encoding
        # This list must be updated based on the actual categorical columns in the data
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

        # --- Create a Scikit-learn Pipeline ---
        
        # Create a preprocessor (OneHotEncoder for categorical features)
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough' # Keep numerical columns as-is
        )

        # Create the full model pipeline
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
        ])

        # --- MLflow Tracking ---
        mlflow.set_experiment("KlinikOps-Length-of-Stay")

        with mlflow.start_run(run_name=f"Training Run - {model_name}") as run:
            logging.info("Starting MLflow run...")
            
            # Log parameters
            mlflow.log_param("model_type", "RandomForestRegressor")
            mlflow.log_param("n_estimators", 50)

            # --- Train the Model ---
            logging.info("Training the model...")
            model_pipeline.fit(X_train, y_train)

            # --- Evaluate the Model ---
            logging.info("Evaluating the model...")
            y_pred = model_pipeline.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            logging.info(f"Evaluation Metrics: MAE={mae:.4f}, R2={r2:.4f}")

            # --- Log Metrics & Model ---
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.sklearn.log_model(
                sk_pipeline, 
                "model", 
                registered_model_name="ClinicOpsModel"
            )
            
            logging.info(f"Run {run.info.run_id} finished successfully.")

    except FileNotFoundError:
        logging.error(f"Error: Processed data files not found at {train_path} or {test_path}")
    except Exception as e:
        logging.error(f"An error occurred during model training: {e}")
    print(run.info.run_id)    
if __name__ == "__main__":
    train_model(TRAIN_PATH, TEST_PATH)