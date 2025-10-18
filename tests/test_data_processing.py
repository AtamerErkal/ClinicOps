# tests/test_data_processing.py
import pandas as pd
import os
import pytest
from scripts.data_processing import process_data

# Create a fixture for dummy data
@pytest.fixture
def dummy_raw_data(tmp_path):
    """Create a dummy raw CSV file in a temporary directory."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    raw_file = raw_dir / "Patient_Stay_Data.csv"
    
    dummy_df = pd.DataFrame({
        'patient_id': [1, 2, 3, 4, 5],
        'age': [50, 60, 70, 80, 90],
        'gender': ['M', 'F', 'M', 'F', 'M'],
        'insurance': ['A', 'B', 'A', 'C', 'B'],
        'stay_days': [5, 10, 3, 12, 8]
    })
    dummy_df.to_csv(raw_file, index=False)
    return str(raw_file)

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