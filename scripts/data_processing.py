# scripts/data_processing.py

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
RAW_DATA_PATH = os.path.join('data', 'raw', 'Patient_Stay_Data.csv.dvc')
PROCESSED_DATA_DIR = os.path.join('data', 'processed')
TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(PROCESSED_DATA_DIR, 'test.csv')

def process_data(raw_path, train_path, test_path, test_size=0.2):
    """
    Loads raw data, performs basic cleaning, splits it into
    train and test sets, and saves them to the processed data directory.
    """
    try:
        logging.info(f"Starting data processing. Loading data from {raw_path}...")
        df = pd.read_csv(raw_path)

        # --- Data Cleaning & Feature Engineering ---
        
        # 1. Drop unnecessary ID and date columns (based on actual data analysis)
        columns_to_drop = ['eid', 'vdate', 'facid']
        df = df.drop(columns_to_drop, axis=1, errors='ignore')
        
        # 2. Check for the target variable (lengthofstay)
        target = 'lengthofstay'
        if target not in df.columns:
            logging.error(f"Target variable '{target}' not found in data.")
            raise ValueError(f"Target variable '{target}' not found.")

        # --- Split the Data ---
        logging.info(f"Splitting data... Target: {target}, Test size: {test_size}")
        
        # Stratify by a relevant categorical column if needed, but simple split is fine for now
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

        # --- Save the Data ---
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logging.info(f"Data successfully processed and saved.")
        logging.info(f"Train data shape: {train_df.shape}")
        logging.info(f"Test data shape: {test_df.shape}")

    except FileNotFoundError:
        logging.error(f"Error: Raw data file not found at {raw_path}")
    except Exception as e:
        logging.error(f"An error occurred during data processing: {e}")
        
if __name__ == "__main__":
    process_data(RAW_DATA_PATH, TRAIN_PATH, TEST_PATH)
