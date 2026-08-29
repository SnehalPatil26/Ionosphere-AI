from cdflib import CDF
from pathlib import Path
import numpy as np


TEC_VARIABLES = [
    "tecIGS",
    "tecESA",
    "tecJPL",
    "tecUPC",
    "tecIGR",
    "tecESR",
    "tecJPR",
    "tecUPR"
]


for year in [2022, 2023, 2024]:

    files = sorted(
        Path("data/tec").glob(
            f"gps_tec2hr_igs_{year}*.cdf"
        )
    )

    if not files:
        print(f"\nNO FILE FOUND FOR {year}")
        continue

    file = files[0]

    print()
    print("=" * 70)
    print(f"YEAR {year}")
    print("=" * 70)
    print("FILE:", file.name)

    cdf = CDF(str(file))

    epoch = np.asarray(cdf.varget("Epoch"))
    lat = np.asarray(cdf.varget("lat"))
    lon = np.asarray(cdf.varget("lon"))

    print()
    print("SHAPES")
    print("Epoch:", epoch.shape)
    print("Latitude:", lat.shape)
    print("Longitude:", lon.shape)

    print()
    print("TEC VARIABLES")

    for variable in TEC_VARIABLES:

        try:
            data = np.asarray(
                cdf.varget(variable)
            )

            valid = np.isfinite(data) & (data > -1e30)

            print(
                f"{variable:<10}"
                f"shape={str(data.shape):<18}"
                f"valid={int(valid.sum()):>8}"
                f"min={data[valid].min() if valid.any() else 'NONE'}"
                f" max={data[valid].max() if valid.any() else 'NONE'}"
            )

        except Exception as error:

            print(
                f"{variable:<10} ERROR: {error}"
            )

    print()
    print("Expected grid rows:")
    print(
        len(epoch),
        "x",
        len(lat),
        "x",
        len(lon),
        "=",
        len(epoch) * len(lat) * len(lon)
    )

print()
print("=" * 70)
print("CHECK COMPLETE")
print("=" * 70)