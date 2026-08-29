from cdflib import CDF
from pathlib import Path

tec_folder = Path("data/tec")
files = list(tec_folder.glob("*.cdf"))

print("TEC files found:", len(files))

for file in files:
    print("\nFile:", file.name)

    try:
        cdf = CDF(str(file))
        info = cdf.cdf_info()

        print("\nCDF Information:")
        print(info)

        print("\nVariables:")
        print(info.zVariables)

        print("\nGlobal Attributes:")
        print(cdf.globalattsget())

        cdf.close()

    except Exception as e:
        print("ERROR:", e)