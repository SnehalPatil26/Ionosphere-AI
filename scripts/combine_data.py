from pathlib import Path
import pandas as pd

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent

TEC_FOLDER = PROJECT_DIR / "data" / "processed"
OMNI_FOLDER = PROJECT_DIR / "data" / "omni"
OUTPUT_FOLDER = PROJECT_DIR / "data" / "processed"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("TEC + OMNI DATA COMBINATION")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load all TEC files
# ---------------------------------------------------------

tec_files = sorted(TEC_FOLDER.glob("tec_*.csv"))

if not tec_files:
    print("ERROR: No TEC CSV files found.")
    print(f"Expected folder: {TEC_FOLDER}")
    raise SystemExit

print(f"\nTEC files found: {len(tec_files)}")

tec_list = []

for file in tec_files:
    print(f"Loading: {file.name}")

    try:
        df = pd.read_csv(file)

        if "datetime" not in df.columns:
            print(f"  SKIPPED: datetime column missing")
            continue

        df["datetime"] = pd.to_datetime(
            df["datetime"],
            errors="coerce"
        )

        df["tec"] = pd.to_numeric(
            df["tec"],
            errors="coerce"
        )

        df = df.dropna(subset=["datetime", "tec"])

        tec_list.append(df)

        print(f"  OK -> {len(df)} rows")

    except Exception as e:
        print(f"  FAILED -> {e}")


if not tec_list:
    print("\nERROR: TEC data could not be loaded.")
    raise SystemExit


tec = pd.concat(
    tec_list,
    ignore_index=True
)

print(f"\nTotal TEC rows: {len(tec):,}")


# ---------------------------------------------------------
# 2. Load OMNI files
# ---------------------------------------------------------

omni_files = sorted(OMNI_FOLDER.glob("omni2_*.txt"))

if not omni_files:
    print("\nERROR: No OMNI TXT files found.")
    print(f"Expected folder: {OMNI_FOLDER}")
    raise SystemExit

print(f"\nOMNI files found: {len(omni_files)}")

omni_list = []

for file in omni_files:
    print(f"Loading: {file.name}")

    try:
        # OMNI files are whitespace separated.
        # Automatically detect columns.
        df = pd.read_csv(
            file,
            sep=r"\s+",
            header=None,
            engine="python"
        )

        print(f"  Columns detected: {df.shape[1]}")
        print(f"  Rows: {len(df):,}")

        omni_list.append(df)

    except Exception as e:
        print(f"  FAILED -> {e}")


if not omni_list:
    print("\nERROR: OMNI data could not be loaded.")
    raise SystemExit


omni = pd.concat(
    omni_list,
    ignore_index=True
)

print(f"\nTotal OMNI rows: {len(omni):,}")


# ---------------------------------------------------------
# 3. Save combined raw dataset
# ---------------------------------------------------------

tec_output = OUTPUT_FOLDER / "tec_all_years.csv"

tec.to_csv(
    tec_output,
    index=False
)

print("\nTEC combined file created:")
print(tec_output)

print("\n" + "=" * 70)
print("DATA COMBINATION STEP COMPLETE")
print("=" * 70)

print(f"TEC rows  : {len(tec):,}")
print(f"OMNI rows : {len(omni):,}")

print("\nTEC columns:")
print(list(tec.columns))

print("\nOMNI columns:")
print(list(omni.columns))

print("\nNext step will create the final ML-ready dataset.")