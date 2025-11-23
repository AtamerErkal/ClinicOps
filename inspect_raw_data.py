"""
Script to inspect raw data and show categorical value formats
"""
import pandas as pd

# Load raw data
df = pd.read_csv("data/raw/Patient_Stay_Data.csv")

print("=" * 80)
print("CATEGORICAL COLUMNS - EXACT VALUES")
print("=" * 80)

# Find categorical columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

for col in cat_cols:
    unique_vals = df[col].unique()
    print(f"\n{col}:")
    print(f"  Unique values: {unique_vals}")

print("\n" + "=" * 80)
print("SAMPLE ROW (First row as JSON)")
print("=" * 80)
print(df.iloc[0].to_dict())

print("\n" + "=" * 80)
print("AFTER pd.get_dummies(drop_first=True)")
print("=" * 80)
df_encoded = pd.get_dummies(df.iloc[:1], drop_first=True)
print(f"Column names ({len(df_encoded.columns)}):")
for col in df_encoded.columns:
    print(f"  - {col}")