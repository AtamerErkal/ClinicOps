# scripts/data_processing.py
import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

RAW_PATH = os.path.join("data", "raw", "Patient_Stay_Data.csv")
PROCESSED_DIR = os.path.join("data", "processed")
TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")


def process_data(test_size=0.2, random_state=42):
    """Load raw CSV, drop unnecessary columns, split, and save train/test CSVs."""
    logging.info(f"Starting data processing. Loading data from {RAW_PATH}...")

    df = pd.read_csv(RAW_PATH)
    
    # Düzeltme: Gereksiz kolonları drop et (eid, vdate, discharged – tarih/ID, LoS tahmini için gereksiz)
    unnecessary_cols = ['eid', 'vdate', 'discharged']  # Explicit list
    df = df.drop([col for col in unnecessary_cols if col in df.columns], axis=1, errors='ignore')
    logging.info(f"Dropped unnecessary columns: {unnecessary_cols}. New shape: {df.shape}")

    # Split into train/test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    logging.info("Data successfully processed and saved.")
    logging.info(f"Train data shape: {train_df.shape}")
    logging.info(f"Test data shape: {test_df.shape}")


if __name__ == "__main__":
    process_data()