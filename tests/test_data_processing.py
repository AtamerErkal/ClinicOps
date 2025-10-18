# tests/test_data_processing.py
import pytest
import pandas as pd
import os
from scripts.data_processing import process_data

@pytest.fixture(scope="session")
def dummy_raw_data(tmp_path_factory):
    """
    Creates a temporary dummy CSV file mimicking the actual data structure
    and returns the path to the file.
    """
    
    # 1. Klasör yapısını oluştur
    data_dir = tmp_path_factory.mktemp("data") / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / "Patient_Stay_Data.csv"

    # 2. Gerçek veri setindeki kritik sütunları içeren sahte veri oluştur
    data = {
        'eid': [1, 2, 3, 4, 5],
        'vdate': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-03'],
        'gender': ['F', 'M', 'F', 'M', 'F'],
        'asthma': [1, 0, 1, 0, 1],
        'hemo': [12.1, 15.5, 13.0, 14.5, 11.8],
        'glucose': [100, 120, 95, 150, 110],
        'facid': [1001, 1002, 1001, 1003, 1002],
        # !!! KRİTİK GÜNCELLEME: HEDEF SÜTUNUNU EKLEDİK !!!
        'lengthofstay': [2, 1, 3, 2, 4] 
    }
    
    df = pd.DataFrame(data)
    df.to_csv(raw_path, index=False)
    
    return str(raw_path)

def test_process_data(dummy_raw_data, tmp_path):
    """Test the data processing script."""
    processed_dir = tmp_path / "data" / "processed"
    train_path = str(processed_dir / "train.csv")
    test_path = str(processed_dir / "test.csv")
    
    # Run the function to be tested
    process_data(dummy_raw_data, train_path, test_path, test_size=0.4) # Use 0.4 for 5 rows
    
    # Check if files were created
    assert os.path.exists(train_path)
    assert os.path.exists(test_path)
    
    # Check if files have correct content
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    assert train_df.shape[0] == 3 # 60% of 5
    assert test_df.shape[0] == 2  # 40% of 5
    
    # Check if patient_id was dropped
    assert 'patient_id' not in train_df.columns
    assert 'patient_id' not in test_df.columns