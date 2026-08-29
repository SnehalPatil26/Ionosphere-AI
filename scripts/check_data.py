from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "tec_omni_ml_dataset.csv"
)

print("=" * 70)
print("TEC DATA QUALITY CHECK")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print("\nDataset shape:")
print(df.shape)

print("\nTEC statistics:")
print(df["tec"].describe())

print("\nTEC missing values:")
print(df["tec"].isna().sum())

print("\nTEC unique values:")
print(df["tec"].nunique())

print("\nTEC first 20 values:")
print(df["tec"].head(20).to_list())

print("\nOMNI missing values:")

omni_columns = [
    col for col in df.columns
    if col.startswith("omni_")
]

missing = df[omni_columns].isna().sum()

print(
    missing.sort_values(ascending=False).head(15)
)

print("\nOMNI columns:")
print(len(omni_columns))

print("\nLatitude range:")
print(df["latitude"].min(), "to", df["latitude"].max())

print("\nLongitude range:")
print(df["longitude"].min(), "to", df["longitude"].max())

print("\nDate range:")
print(df["datetime"].min(), "to", df["datetime"].max())

print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)