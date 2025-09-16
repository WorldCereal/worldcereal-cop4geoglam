import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

folder = "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique/refdata/original/"

file = os.path.join(folder,"MOZ_desk.parquet")

gdf = gpd.read_parquet(file)
gdf["id_psu"] = [ssu.split("_")[0] for ssu in gdf["id_ssu"]]

PSU = gdf["id_psu"].unique()

psu_box = gpd.GeoDataFrame(crs=gdf.crs, columns=["id_psu", "geometry"])

#add_factor determined by crs
if gdf.crs.to_string() == "EPSG:4326":
    add_factor = 0.0005
else:
    add_factor = 50

for psu in PSU:
    temp = gdf[gdf["id_psu"]==psu]
    if psu == "254562":
        temp = temp[temp["id_ssu"]!="254562_24"]
    minx, miny, maxx, maxy = temp.total_bounds
    minx = minx - add_factor
    miny = miny - add_factor
    maxx = maxx + add_factor
    maxy = maxy + add_factor
    geom = box(minx, miny, maxx, maxy)
    psu_box = pd.concat([psu_box, gpd.GeoDataFrame({"id_psu":[psu],
                                                    "geometry":[geom]})],
                                                    ignore_index=True)

#save to file
outfolder = "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique/refdata/"
outname = os.path.join(outfolder, "MOZ_PSU.parquet")
psu_box.to_parquet(outname, index=False)
