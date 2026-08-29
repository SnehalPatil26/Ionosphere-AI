from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# TEC + OMNI -> ML READY DATASET
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

TEC_FILE = PROJECT_DIR / "data" / "processed" / "tec_all_years.csv"
OMNI_FOLDER = PROJECT_DIR / "data" / "omni"
OUTPUT_FOLDER = PROJECT_DIR / "data" / "processed"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PREPARING FINAL ML DATASET")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load TEC data
# ------------------------------------------------------------

print("\n[1/5] Loading TEC data...")

tec = pd.read_csv(TEC_FILE)

tec["datetime"] = pd.to_datetime(
    tec["datetime"],
    errors="coerce"
)

tec["latitude"] = pd.to_numeric(
    tec["latitude"],
    errors="coerce"
)

tec["longitude"] = pd.to_numeric(
    tec["longitude"],
    errors="coerce"
)

tec["tec"] = pd.to_numeric(
    tec["tec"],
    errors="coerce"
)

tec = tec.dropna(
    subset=["datetime", "latitude", "longitude", "tec"]
)

print(f"TEC rows: {len(tec):,}")

# ------------------------------------------------------------
# 2. Load OMNI data
# ------------------------------------------------------------

print("\n[2/5] Loading OMNI data...")

omni_files = sorted(
    OMNI_FOLDER.glob("omni2_*.txt")
)

omni_list = []

for file in omni_files:

    print(f"Reading {file.name}...")

    df = pd.read_csv(
        file,
        sep=r"\s+",
        header=None,
        engine="python"
    )

    # --------------------------------------------------------
    # OMNI2 hourly data standard first columns:
    #
    # 0 = year
    # 1 = day of year
    # 2 = hour
    # --------------------------------------------------------

    df = df.rename(
        columns={
            0: "year",
            1: "day_of_year",
            2: "hour"
        }
    )

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce"
    )

    df["day_of_year"] = pd.to_numeric(
        df["day_of_year"],
        errors="coerce"
    )

    df["hour"] = pd.to_numeric(
        df["hour"],
        errors="coerce"
    )

    # Create datetime
    df["datetime"] = (
        pd.to_datetime(
            df["year"].astype("Int64").astype(str)
            + "-01-01",
            errors="coerce"
        )
        + pd.to_timedelta(
            df["day_of_year"] - 1,
            unit="D"
        )
        + pd.to_timedelta(
            df["hour"],
            unit="h"
        )
    )

    omni_list.append(df)

omni = pd.concat(
    omni_list,
    ignore_index=True
)

print(f"OMNI rows: {len(omni):,}")

# ------------------------------------------------------------
# 3. Select useful OMNI columns
# ------------------------------------------------------------

print("\n[3/5] Selecting OMNI features...")

# Common OMNI2 column positions
# We keep only numerical columns that can be useful
# for TEC prediction.

candidate_columns = [
    3,   # IMF / solar wind related
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54
]

available_columns = [
    col for col in candidate_columns
    if col in omni.columns
]

omni_features = omni[
    ["datetime"] + available_columns
].copy()

# Give safe feature names
rename_map = {
    col: f"omni_{i + 1}"
    for i, col in enumerate(available_columns)
}

omni_features = omni_features.rename(
    columns=rename_map
)

# Convert features to numeric
for col in omni_features.columns:
    if col != "datetime":
        omni_features[col] = pd.to_numeric(
            omni_features[col],
            errors="coerce"
        )

print(
    f"OMNI features selected: "
    f"{len(available_columns)}"
)

# ------------------------------------------------------------
# 4. Merge TEC + OMNI
# ------------------------------------------------------------

print("\n[4/5] Merging TEC and OMNI...")

# TEC data contains multiple locations.
# OMNI is hourly, so merge using nearest timestamp.

tec = tec.sort_values("datetime")

omni_features = omni_features.sort_values(
    "datetime"
)

merged = pd.merge_asof(
    tec,
    omni_features,
    on="datetime",
    direction="nearest",
    tolerance=pd.Timedelta("1 hour")
)

print(
    f"Merged rows before cleaning: "
    f"{len(merged):,}"
)

# ------------------------------------------------------------
# 5. Clean and create ML features
# ------------------------------------------------------------

print("\n[5/5] Cleaning and creating features...")

# Remove rows without OMNI values
omni_columns = [
    col for col in merged.columns
    if col.startswith("omni_")
]

merged = merged.dropna(
    subset=omni_columns,
    how="all"
)

# Time features
merged["year"] = merged["datetime"].dt.year
merged["month"] = merged["datetime"].dt.month
merged["day"] = merged["datetime"].dt.day
merged["hour"] = merged["datetime"].dt.hour
merged["day_of_year"] = merged["datetime"].dt.dayofyear

# Cyclic time features
merged["hour_sin"] = np.sin(
    2 * np.pi * merged["hour"] / 24
)

merged["hour_cos"] = np.cos(
    2 * np.pi * merged["hour"] / 24
)

merged["month_sin"] = np.sin(
    2 * np.pi * merged["month"] / 12
)

merged["month_cos"] = np.cos(
    2 * np.pi * merged["month"] / 12
)

# Sort
merged = merged.sort_values(
    ["datetime", "latitude", "longitude"]
)

# Remove duplicate rows
merged = merged.drop_duplicates()

# Reset index
merged = merged.reset_index(drop=True)

# ------------------------------------------------------------
# Save final dataset
# ------------------------------------------------------------

output_file = (
    OUTPUT_FOLDER /
    "tec_omni_ml_dataset.csv"
)

merged.to_csv(
    output_file,
    index=False
)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL ML DATASET CREATED")
print("=" * 70)

print(f"Rows       : {len(merged):,}")
print(f"Columns    : {len(merged.columns):,}")
print(f"Output     : {output_file}")

print("\nColumns:")
for col in merged.columns:
    print(f"  - {col}")

print("\nFirst 5 rows:")
print(
    merged.head().to_string()
)

print("\n" + "=" * 70)
print("SUCCESS!")
print("=" * 70)