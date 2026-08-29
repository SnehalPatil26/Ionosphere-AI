from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# TEC + OMNI PREDICTION MODEL
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "tec_omni_ml_dataset.csv"
)

MODEL_FOLDER = PROJECT_DIR / "models"
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_FOLDER / "tec_prediction_model.pkl"


print("=" * 70)
print("TEC PREDICTION MODEL TRAINING")
print("=" * 70)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\n[1/8] Loading dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Original rows    : {len(df):,}")
print(f"Original columns : {len(df.columns):,}")


# ============================================================
# 2. CONVERT DATETIME
# ============================================================

print("\n[2/8] Processing datetime...")

df["datetime"] = pd.to_datetime(
    df["datetime"],
    errors="coerce"
)

df = df.dropna(
    subset=["datetime"]
)

df = df.sort_values(
    "datetime"
).reset_index(drop=True)


# ============================================================
# 3. CLEAN TEC DATA
# ============================================================

print("\n[3/8] Cleaning TEC and OMNI data...")

# TEC invalid values
df["tec"] = pd.to_numeric(
    df["tec"],
    errors="coerce"
)

# Remove known invalid TEC fill value
df.loc[
    df["tec"] <= -100,
    "tec"
] = np.nan

# TEC should be positive
df.loc[
    df["tec"] <= 0,
    "tec"
] = np.nan


# ============================================================
# CLEAN OMNI VALUES
# ============================================================

omni_columns = [
    col for col in df.columns
    if str(col).startswith("omni_")
]

print(f"OMNI features found: {len(omni_columns)}")

for col in omni_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    # Remove common OMNI fill values
    df.loc[
        df[col].abs() >= 9999,
        col
    ] = np.nan


# ============================================================
# CLEAN LATITUDE / LONGITUDE
# ============================================================

df["latitude"] = pd.to_numeric(
    df["latitude"],
    errors="coerce"
)

df["longitude"] = pd.to_numeric(
    df["longitude"],
    errors="coerce"
)


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

df = df.dropna(
    subset=[
        "tec",
        "latitude",
        "longitude"
    ]
).copy()


print(
    f"Rows after cleaning: {len(df):,}"
)


# ============================================================
# 4. SELECT FEATURES
# ============================================================

print("\n[4/8] Preparing features...")


base_features = [
    "latitude",
    "longitude",
    "year",
    "month",
    "day",
    "hour",
    "day_of_year",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos"
]


feature_columns = []

for col in base_features:

    if col in df.columns:
        feature_columns.append(col)


# Add OMNI features
feature_columns.extend(
    omni_columns
)


print(
    f"Features selected: {len(feature_columns)}"
)


# ============================================================
# CREATE X AND y
# ============================================================

X = df[
    feature_columns
].copy()

y = df[
    "tec"
].copy()


# ============================================================
# NUMERIC CONVERSION
# ============================================================

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# REMOVE INFINITE VALUES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# MEDIAN IMPUTATION
# ============================================================

for col in X.columns:

    median_value = X[col].median()

    if pd.isna(median_value):

        median_value = 0

    X[col] = X[col].fillna(
        median_value
    )


# ============================================================
# FINAL SAFETY CHECK
# ============================================================

X = X.fillna(0)


print(
    f"Final training samples: {len(X):,}"
)


# ============================================================
# 5. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

print("\n[5/8] Creating time-based train/test split...")


# IMPORTANT:
# We do NOT randomly shuffle the data.
# Earlier random split could make the evaluation unrealistic.


split_index = int(
    len(X) * 0.80
)


X_train = X.iloc[
    :split_index
].copy()

X_test = X.iloc[
    split_index:
].copy()


y_train = y.iloc[
    :split_index
].copy()

y_test = y.iloc[
    split_index:
].copy()


print(
    f"Training rows: {len(X_train):,}"
)

print(
    f"Testing rows : {len(X_test):,}"
)

print(
    f"Training period: "
    f"{df['datetime'].iloc[0]} "
    f"to "
    f"{df['datetime'].iloc[split_index - 1]}"
)

print(
    f"Testing period : "
    f"{df['datetime'].iloc[split_index]} "
    f"to "
    f"{df['datetime'].iloc[-1]}"
)


# ============================================================
# 6. TRAIN RANDOM FOREST
# ============================================================

print("\n[6/8] Training Random Forest...")


model = RandomForestRegressor(
    n_estimators=200,
    max_depth=18,
    min_samples_leaf=3,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


print(
    "Model training completed!"
)


# ============================================================
# 7. EVALUATE MODEL
# ============================================================

print("\n[7/8] Evaluating model...")


predictions = model.predict(
    X_test
)


mae = mean_absolute_error(
    y_test,
    predictions
)


rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)


r2 = r2_score(
    y_test,
    predictions
)


print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"MAE  : {mae:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"R²   : {r2:.4f}"
)

print("=" * 70)


# ============================================================
# 8. SAVE MODEL
# ============================================================

print("\n[8/8] Saving model...")


model_package = {

    "model": model,

    "features": feature_columns,

    "mae": mae,

    "rmse": rmse,

    "r2": r2

}


joblib.dump(
    model_package,
    MODEL_FILE
)


print("\nModel saved successfully:")

print(
    MODEL_FILE
)


print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)