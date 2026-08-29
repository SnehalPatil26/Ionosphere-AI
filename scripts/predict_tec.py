from pathlib import Path
import pandas as pd
import numpy as np
import joblib

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_DIR / "data" / "processed" / "tec_omni_ml_dataset.csv"
MODEL_FILE = PROJECT_DIR / "models" / "tec_prediction_model.pkl"
OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "tec_predictions.csv"

print("=" * 70)
print("TEC PREDICTION SYSTEM")
print("=" * 70)

print("\n[1/5] Loading trained model...")

package = joblib.load(MODEL_FILE)

model = package["model"]
feature_columns = package["features"]

print("Model loaded successfully!")
print(f"Features required: {len(feature_columns)}")

print("\n[2/5] Loading ML dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Original dataset rows: {len(df):,}")

print("\n[3/5] Cleaning prediction data...")

df["datetime"] = pd.to_datetime(
    df["datetime"],
    errors="coerce"
)

# Invalid TEC values
df["tec"] = pd.to_numeric(
    df["tec"],
    errors="coerce"
)

df = df.replace(
    [
        -1e31,
        1e31,
        999999.99,
        99999.99,
        9999.99,
        -9999.99
    ],
    np.nan
)

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

# Keep only valid TEC
df = df[
    df["tec"].notna()
]

df = df[
    (df["tec"] > 0) &
    (df["tec"] < 100)
].copy()

df = df.dropna(
    subset=[
        "datetime",
        "latitude",
        "longitude"
    ]
)

print(f"Valid prediction rows: {len(df):,}")

print("\n[4/5] Preparing prediction features...")

X = df[feature_columns].copy()

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

for column in X.columns:

    if X[column].isna().any():

        median_value = X[column].median()

        if pd.isna(median_value):
            median_value = 0

        X[column] = X[column].fillna(
            median_value
        )

X = X.fillna(0)

print(f"Prediction rows: {len(X):,}")

print("\n[5/5] Generating TEC predictions...")

predictions = model.predict(X)

predictions = np.maximum(
    predictions,
    0
)

df["predicted_tec"] = predictions

df["prediction_error"] = (
    df["predicted_tec"] - df["tec"]
)

df["absolute_error"] = (
    df["prediction_error"].abs()
)

df[
    [
        "datetime",
        "latitude",
        "longitude",
        "tec",
        "predicted_tec",
        "prediction_error",
        "absolute_error"
    ]
].to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("PREDICTION COMPLETE")
print("=" * 70)

print(f"Rows predicted : {len(df):,}")
print(f"Output file    : {OUTPUT_FILE}")

print("\nFirst 10 predictions:")

print(
    df[
        [
            "datetime",
            "latitude",
            "longitude",
            "tec",
            "predicted_tec",
            "prediction_error"
        ]
    ].head(10).to_string(index=False)
)

mean_actual = df["tec"].mean()
mean_predicted = df["predicted_tec"].mean()
mae = df["absolute_error"].mean()

print("\nPrediction statistics:")

print(f"Mean actual TEC    : {mean_actual:.4f}")
print(f"Mean predicted TEC : {mean_predicted:.4f}")
print(f"Mean absolute error: {mae:.4f}")

print("\n" + "=" * 70)
print("SUCCESS!")
print("=" * 70)