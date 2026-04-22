
import glob
import json
import os
from typing import Any, Dict, cast

import geopandas as gpd
import pandas as pd

activation_folder = "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique_pm"
extractions_folder = os.path.join(activation_folder, "extractions")

merged_extractions = glob.glob(os.path.join(extractions_folder, "merged_extractions","*","*.parquet"))

out_folder = os.path.join(activation_folder,"trainingdata")
out_name = "2025_MOZ_COPERNICUS4GEOGLAM_ITC_POINT_EXP_POLY_MERGED.parquet"

if not os.path.exists(out_folder):
    os.makedirs(out_folder)
    os.chmod(out_folder, 0o777)

if not os.path.exists(os.path.join(out_folder,out_name)):
    merged_file = []

    for extraction_file in merged_extractions:
        file = gpd.read_parquet(extraction_file)
        file["source_file"] = os.path.basename(extraction_file)
        file.to_crs(epsg=4326, inplace=True)
        merged_file.append(file)

    merged_gdf = gpd.GeoDataFrame(pd.concat(merged_file, ignore_index=True))

    merged_gdf.to_parquet(os.path.join(out_folder,out_name),index=False)
else:
    merged_gdf = gpd.read_parquet(os.path.join(out_folder,out_name))

ITC_points = merged_gdf[merged_gdf["sample_id"].str.contains("ITC")]
C4G_points = merged_gdf[~merged_gdf["sample_id"].str.contains("ITC")]

ITC_points["ssu_id"] = ITC_points["sample_id"].str.split("_").str[5] + "_" + ITC_points["sample_id"].str.split("_").str[6]
C4G_points["ssu_id"] = C4G_points["sample_id"].str.split("_").str[5] + "_" + C4G_points["sample_id"].str.split("_").str[6]

merged_gdf = gpd.GeoDataFrame(pd.concat([ITC_points,C4G_points], ignore_index=True))

ITC_lookup_path = os.path.join(activation_folder,"refdata","harmonized","lookup","2025_MOZ_ITC_POINT_110_harmonized_lookup.parquet")
ITC_lookup = pd.read_parquet(ITC_lookup_path)

#add cropping pattern info to ITC_points from ITC_lookup based on sample_id
ITC_points = ITC_points.merge(ITC_lookup[["sample_id","cropping_pattern",'trees_in_cropfield']], on="sample_id", how="left")

ITC_points = ITC_points[ITC_points["trees_in_cropfield"]!="trees_yes"]

#only monocropping
ITC_mono = ITC_points[ITC_points["cropping_pattern"] == "mono_culture"]
merged_mono = gpd.GeoDataFrame(pd.concat([ITC_mono,C4G_points], ignore_index=True))

class_mappings_path = os.path.join(activation_folder, "class_mappings_mozambique.json")

with open(class_mappings_path) as f:
    loaded = json.load(f)
    if isinstance(loaded, list):
        # Convert list of mappings to a dict
        class_mappings: Dict[str, Any] = {mapping["name"]: mapping["mapping"] for mapping in loaded}
    else:
        class_mappings = cast(Dict[str, Any], loaded)

landcover_mapping = class_mappings["LANDCOVER10"]
croptype_mapping = class_mappings["CROPTYPE_Mozambique"]

merged_mono["ewoc_code"] = merged_mono["ewoc_code"].astype(str)
merged_mono["landcover"] = merged_mono["ewoc_code"].map(landcover_mapping)
merged_mono["croptype"] = merged_mono["ewoc_code"].map(croptype_mapping)

#provide a table for the number of samples and unique ssu_id's per landcover class
summary_table_landcover = merged_mono.groupby(["landcover","source_file"]).agg({"sample_id":"nunique","ssu_id":"nunique"}).reset_index()
summary_table_croptype = merged_mono.groupby(["croptype","source_file"]).agg({"sample_id":"nunique","ssu_id":"nunique"}).reset_index()

#save them as .txt files
summary_table_landcover.to_csv(os.path.join(out_folder,"summary_table_landcover_monocropping.txt"), index=False, sep="\t")
summary_table_croptype.to_csv(os.path.join(out_folder,"summary_table_croptype_monocropping.txt"), index=False, sep="\t")
