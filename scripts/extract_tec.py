from cdflib import CDF, cdfepoch
from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEC_FOLDER = BASE_DIR / "data" / "tec"
OUTPUT_FOLDER = BASE_DIR / "data" / "processed"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEC SOURCE PRIORITY
# ============================================================

TEC_VARIABLES = [
    "tecIGS",
    "tecESA",
    "tecJPL",
    "tecUPC",
    "tecIGR",
    "tecCOR",
    "tecESR",
    "tecJPR",
    "tecUPR"
]


# ============================================================
# FIND BEST TEC VARIABLE
# ============================================================

def find_best_tec_variable(cdf):

    print()
    print("Available TEC variables:")

    best_variable = None
    best_valid_count = 0

    for variable in TEC_VARIABLES:

        try:

            data = np.asarray(
                cdf.varget(variable)
            )

            data = data.astype(float)

            valid = (
                np.isfinite(data)
                &
                (data > -1e30)
            )

            valid_count = int(
                np.sum(valid)
            )

            print(
                f"  {variable:<10} "
                f"valid values: {valid_count:,}"
            )

            if valid_count > best_valid_count:

                best_valid_count = valid_count
                best_variable = variable

        except Exception:

            pass

    if best_variable is None:

        raise ValueError(
            "No valid TEC variable found"
        )

    print()
    print(
        f"Selected TEC source: "
        f"{best_variable}"
    )

    print(
        f"Valid TEC values: "
        f"{best_valid_count:,}"
    )

    return best_variable


# ============================================================
# PROCESS ONE CDF FILE
# ============================================================

def process_cdf_file(cdf_path):

    print()
    print(
        f"Opening: {cdf_path.name}"
    )

    cdf = CDF(str(cdf_path))

    # --------------------------------------------------------
    # Read Epoch
    # --------------------------------------------------------

    epoch_data = np.asarray(
        cdf.varget("Epoch")
    )

    # --------------------------------------------------------
    # Read latitude / longitude
    # --------------------------------------------------------

    latitude_data = np.asarray(
        cdf.varget("lat")
    )

    longitude_data = np.asarray(
        cdf.varget("lon")
    )

    # --------------------------------------------------------
    # Select BEST TEC source
    # --------------------------------------------------------

    tec_variable = find_best_tec_variable(
        cdf
    )

    tec_data = np.asarray(
        cdf.varget(tec_variable)
    )

    # --------------------------------------------------------
    # Print shapes
    # --------------------------------------------------------

    print()
    print("Shapes:")
    print(
        f"Epoch     : {epoch_data.shape}"
    )
    print(
        f"Latitude  : {latitude_data.shape}"
    )
    print(
        f"Longitude : {longitude_data.shape}"
    )
    print(
        f"TEC       : {tec_data.shape}"
    )

    # --------------------------------------------------------
    # Validate expected dimensions
    # --------------------------------------------------------

    if tec_data.ndim != 3:

        raise ValueError(
            f"Unexpected TEC dimensions: "
            f"{tec_data.shape}"
        )

    time_count = len(epoch_data)
    lat_count = len(latitude_data)
    lon_count = len(longitude_data)

    expected_shape = (
        time_count,
        lat_count,
        lon_count
    )

    if tec_data.shape != expected_shape:

        raise ValueError(
            f"TEC shape {tec_data.shape} "
            f"does not match expected "
            f"{expected_shape}"
        )

    # --------------------------------------------------------
    # Convert Epoch
    # --------------------------------------------------------

    try:

        datetime_data = cdfepoch.to_datetime(
            epoch_data
        )

    except Exception:

        datetime_data = pd.to_datetime(
            epoch_data,
            errors="coerce"
        )

    datetime_data = pd.to_datetime(
        datetime_data,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Create mesh/grid
    # --------------------------------------------------------

    time_grid = np.repeat(
        datetime_data,
        lat_count * lon_count
    )

    lat_grid = np.tile(
        np.repeat(
            latitude_data,
            lon_count
        ),
        time_count
    )

    lon_grid = np.tile(
        longitude_data,
        time_count * lat_count
    )

    tec_grid = tec_data.reshape(
        -1
    )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        {
            "datetime": time_grid,
            "latitude": lat_grid,
            "longitude": lon_grid,
            "tec": tec_grid
        }
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    df["tec"] = pd.to_numeric(
        df["tec"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Replace CDF fill values
    # --------------------------------------------------------

    df.loc[
        df["tec"] <= -1e30,
        "tec"
    ] = np.nan

    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    before = len(df)

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=[
            "datetime",
            "latitude",
            "longitude",
            "tec"
        ]
    )

    removed = before - len(df)

    print(
        f"Invalid rows removed: "
        f"{removed:,}"
    )

    print(
        f"Valid rows: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Delete CDF object
    # --------------------------------------------------------

    del cdf

    return df


# ============================================================
# PROCESS YEAR
# ============================================================

def process_year(year):

    print()
    print("=" * 70)
    print(f"YEAR {year}")
    print("=" * 70)

    year_files = sorted(
        TEC_FOLDER.glob(
            f"gps_tec2hr_igs_{year}*.cdf"
        )
    )

    print(
        f"CDF files found: "
        f"{len(year_files)}"
    )

    if not year_files:

        print(
            f"No CDF files found for {year}"
        )

        return

    all_data = []

    successful = 0
    failed = 0

    # --------------------------------------------------------
    # Process files
    # --------------------------------------------------------

    for index, cdf_file in enumerate(
        year_files,
        start=1
    ):

        print()
        print(
            f"[{index}/{len(year_files)}] "
            f"{cdf_file.name}"
        )

        try:

            df = process_cdf_file(
                cdf_file
            )

            if len(df) > 0:

                all_data.append(df)

                successful += 1

                print(
                    f"SUCCESS | "
                    f"{len(df):,} rows"
                )

            else:

                failed += 1

                print(
                    "FAILED | "
                    "No valid rows"
                )

        except Exception as error:

            failed += 1

            print(
                f"FAILED | "
                f"{type(error).__name__}: "
                f"{error}"
            )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    output_file = (
        OUTPUT_FOLDER /
        f"tec_{year}.csv"
    )

    if all_data:

        print()
        print(
            "Combining yearly data..."
        )

        final_df = pd.concat(
            all_data,
            ignore_index=True
        )

        final_df = final_df.sort_values(
            by="datetime"
        )

        final_df = final_df.reset_index(
            drop=True
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        final_df.to_csv(
            output_file,
            index=False
        )

        total_rows = len(final_df)

    else:

        total_rows = 0

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print(f"YEAR {year} COMPLETE")
    print(
        f"Successful files : {successful}"
    )
    print(
        f"Failed files     : {failed}"
    )
    print(
        f"Total rows       : {total_rows:,}"
    )
    print(
        f"Output file      : {output_file}"
    )
    print("-" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    years = [
        2020,
        2021,
        2022,
        2023,
        2024
    ]

    print()
    print("=" * 70)
    print("FINAL TEC DATA EXTRACTION")
    print("=" * 70)

    print()
    print(
        f"TEC folder    : {TEC_FOLDER}"
    )

    print(
        f"Output folder : {OUTPUT_FOLDER}"
    )

    for year in years:

        process_year(year)

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ALL YEARS PROCESSED")
    print("=" * 70)

    for year in years:

        output_file = (
            OUTPUT_FOLDER /
            f"tec_{year}.csv"
        )

        if output_file.exists():

            try:

                df = pd.read_csv(
                    output_file
                )

                print(
                    f"{output_file.name}: "
                    f"{len(df):,} rows"
                )

            except Exception:

                print(
                    f"{output_file.name}: "
                    "created"
                )

    print()
    print(
        "TEC extraction finished."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()