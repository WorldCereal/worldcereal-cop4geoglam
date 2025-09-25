import os

import geopandas as gpd
import pandas as pd

folder = "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique/refdata/original/"

gpkg = gpd.read_file(os.path.join(folder, "moz_results_2025.gpkg"))

gpkg_agroforestry = gpkg[gpkg["trees_in_cropfield"]== 'trees_yes']

print(gpkg_agroforestry["id_ssu"])

outfolder = "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique/refdata/harmonized/"

pd.DataFrame(gpkg_agroforestry[["id_ssu"]]).to_csv(os.path.join(outfolder,
                                                                "potential_agroforestry_ssu.csv"))
