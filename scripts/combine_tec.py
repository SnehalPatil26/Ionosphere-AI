import os
import pandas as pd
import cdflib

TEC_FOLDER = "data/tec"
OUTPUT_FILE = "data/processed/tec_2020_2024.csv"

os.makedirs("data/processed", exist_ok=True)

all_data = []

files = sorted(
    file for file in os.listdir(TEC_FOLDER)
    if file.lower().endswith(".cdf")
)

print(f"Found {len(files)} CDF files.")

for index, filename in enumerate(files, start=1):

    filepath = os.path.join(TEC_FOLDER, filename)

    try:
        cdf = cdflib.CDF(filepath)
        info = cdf.cdf_info()

        variables = info.zVariables

        # Find important variables
        epoch_var = None
        lat_var = None
        lon_var = None
        tec_var = None

        for var in variables:
            name = var.lower()

            if "epoch" in name or "time" in name:
                epoch_var = var

            elif "lat" in name:
                lat_var = var

            elif "lon" in name:
                lon_var = var

            elif "tec" in name:
                tec_var = var

        if not all([epoch_var, lat_var, lon_var, tec_var]):
            print(f"[SKIP] Variables not found: {filename}")
            continue

        epoch = cdf.varget(epoch_var)
        latitude = cdf.varget(lat_var)
        longitude = cdf.varget(lon_var)
        tec = cdf.varget(tec_var)

        dates = cdflib.cdfepoch.to_datetime(epoch)

        df = pd.DataFrame({
            "datetime": dates,
            "latitude": latitude,
            "longitude": longitude,
            "tec": tec
        })

        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        df = df.dropna(subset=["datetime", "latitude", "longitude", "tec"])

        all_data.append(df)

        print(f"[{index}/{len(files)}] OK - {filename}")

    except Exception as e:
        print(f"[ERROR] {filename} -> {e}")


if all_data:

    final_df = pd.concat(all_data, ignore_index=True)

    final_df = final_df.sort_values("datetime")

    final_df = final_df.drop_duplicates()

    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\n==============================")
    print("TEC CSV CREATION COMPLETE")
    print("==============================")
    print(f"CDF files processed : {len(files)}")
    print(f"Rows                : {len(final_df)}")
    print(f"Output              : {OUTPUT_FILE}")
    print("==============================")

else:
    print("No TEC data was successfully processed.")