import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
RAW_DATA_PATH = os.path.join('data', 'raw', 'Patient_Stay_Data.csv')
PROCESSED_DATA_DIR = os.path.join('data', 'processed')
TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(PROCESSED_DATA_DIR, 'test.csv')

def process_data(raw_path=RAW_DATA_PATH, train_path=TRAIN_PATH, test_path=TEST_PATH, test_size=0.2):
    """
    Loads raw data, cleans it, converts categorical to numeric,
    splits into train/test, and saves processed CSVs.
    """
    try:
        logging.info(f"Starting data processing. Loading data from {raw_path}...")
        df = pd.read_csv(raw_path)

        # --- Data Cleaning ---
        columns_to_drop = ['eid', 'vdate']  # facid is kept
        df = df.drop(columns_to_drop, axis=1, errors='ignore')

        # --- Target Check ---
        target = 'lengthofstay'
        if target not in df.columns:
            logging.error(f"Target variable '{target}' not found in data.")
            raise ValueError(f"Target variable '{target}' not found.")

        # --- Categorical → Numeric ---
        cat_cols = df.select_dtypes(include='object').columns
        for col in cat_cols:
            df[col] = df[col].astype('category').cat.codes
            logging.info(f"Converted {col} to numeric codes.")

        # --- Split ---
        logging.info(f"Splitting data... Target: {target}, Test size: {test_size}")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

        # --- Save CSV ---
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logging.info(f"Data successfully processed and saved.")
        logging.info(f"Train data shape: {train_df.shape}")
        logging.info(f"Test data shape: {test_df.shape}")

    except FileNotFoundError:
        logging.error(f"Raw data file not found at {raw_path}. Did 'dvc pull' run successfully?")
    except Exception as e:
        logging.error(f"An error occurred during data processing: {e}")


if __name__ == "__main__":
    process_data()
