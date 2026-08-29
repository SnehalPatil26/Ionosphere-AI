import os
import requests
from datetime import date, timedelta

OUTPUT_FOLDER = "data/tec"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

START_YEAR = 2020
END_YEAR = 2024

session = requests.Session()

current = date(START_YEAR, 1, 1)
end = date(END_YEAR, 12, 31)

total = 0
downloaded = 0
skipped = 0
failed = 0

while current <= end:

    year = current.year
    date_str = current.strftime("%Y%m%d")

    filename = f"gps_tec2hr_igs_{date_str}_v01.cdf"

    url = (
        f"https://spdf.gsfc.nasa.gov/pub/data/gps/"
        f"tec2hr_igs/{year}/{filename}"
    )

    output_path = os.path.join(OUTPUT_FOLDER, filename)

    total += 1

    # Already downloaded
    if os.path.exists(output_path):
        skipped += 1
        print(f"[SKIP] {filename}")
        current += timedelta(days=1)
        continue

    try:
        response = session.get(url, timeout=60)

        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)

            downloaded += 1
            print(f"[OK] {filename}")

        else:
            failed += 1
            print(f"[NOT FOUND] {filename}")

    except Exception as e:
        failed += 1
        print(f"[ERROR] {filename} -> {e}")

    current += timedelta(days=1)

print("\n==============================")
print("TEC DOWNLOAD COMPLETE")
print("==============================")
print(f"Total dates checked : {total}")
print(f"Downloaded          : {downloaded}")
print(f"Already existed     : {skipped}")
print(f"Failed / unavailable: {failed}")
print("==============================")