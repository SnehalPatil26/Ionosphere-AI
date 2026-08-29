import pandas as pd
from pathlib import Path

folder = Path("data/processed")

print()
print("=" * 75)
print("PROCESSED TEC DATA VERIFICATION")
print("=" * 75)

print()
print(f"{'YEAR':<8}{'ROWS':>15}{'MIN':>12}{'MAX':>12}{'MEAN':>12}")
print("-" * 60)

for year in range(2020, 2025):

    file = folder / f"tec_{year}.csv"

    if not file.exists():
        print(f"{year:<8} FILE NOT FOUND")
        continue

    print(f"Reading {file.name} ...")

    df = pd.read_csv(file)

    print(
        f"{year:<8}"
        f"{len(df):>15,}"
        f"{df['tec'].min():>12.2f}"
        f"{df['tec'].max():>12.2f}"
        f"{df['tec'].mean():>12.2f}"
    )

print()
print("=" * 75)
print("VERIFICATION COMPLETE")
print("=" * 75)