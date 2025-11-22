import pandas as pd
import numpy as np
import logging
import os
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

RAW_DATA_PATH = "data/raw/Patient_Stay_Data.csv"
PROCESSED_DATA_PATH = "data/processed"

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

def preprocess_data():
    logging.info(f"Starting data processing. Loading data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)

    # Basit encoding
    for col in ['rcount','gender','discharged','facid']:
        df[col] = df[col].astype('category').cat.codes
        logging.info(f"Converted {col} to numeric codes.")

    # Split
    target = 'lengthofstay'
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Save
    train_path = os.path.join(PROCESSED_DATA_PATH, "train.csv")
    test_path = os.path.join(PROCESSED_DATA_PATH, "test.csv")
    pd.concat([X_train, y_train], axis=1).to_csv(train_path, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_path, index=False)

    logging.info(f"Data successfully processed and saved.")
    logging.info(f"Train data shape: {X_train.shape}")
    logging.info(f"Test data shape: {X_test.shape}")

if __name__ == "__main__":
    preprocess_data()
