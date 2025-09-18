'''
Main Code to duplicate extractions for mixed cropping points.
#►Copernicus4GEOGLAM
'''

import os
from json import dump, load

import geopandas as gpd
import pandas as pd

activation = "mozambique"

base_folder = "/vitodata/worldcereal/data/COP4GEOGLAM/"
activation_folder = os.path.join(base_folder, activation)
extractions = os.path.join(activation_folder,
                           "trainingdata","worldcereal_merged_extractions.parquet")
#get the duplicate mapping json from the same folder as this script
duplicate_mapping_file = os.path.join(activation_folder,"mixed_cropping",
                                      f"duplicate_mapping_{activation.capitalize()}.json")
#open json file
with open(duplicate_mapping_file, "r") as f:
    duplicate_mapping = load(f)
duplicate_mapping_final = duplicate_mapping["DUPLICATE_CROPTYPE_FINAL"]
duplicate_mapping_original = duplicate_mapping["DUPLICATE_CROPTYPE_ORIGINAL"]

class_mapping_file = os.path.join(activation_folder,"refdata","harmonized",
                                  f"class_mapping_{activation.capitalize()}.json")
with open(class_mapping_file, "r") as f:
    class_mapping = load(f)
croptype_mapping = class_mapping[f"CROPTYPE_{activation.capitalize()}"]

duplicate_croptype_mapping = {}
for key, value in croptype_mapping.items():
    if value not in duplicate_croptype_mapping:
        duplicate_croptype_mapping[value] = key

original_gpkg = os.path.join(activation_folder,
                             "refdata","original","moz_results_2025.gpkg")

extractions_dataset = gpd.read_parquet(extractions)
original_dataset = gpd.read_file(original_gpkg)

extractions_dataset["round_x"] = extractions_dataset["geometry"].x.round(4)
extractions_dataset["round_y"] = extractions_dataset["geometry"].y.round(4)

original_dataset["round_x"] = original_dataset["geometry"].x.round(4)
original_dataset["round_y"] = original_dataset["geometry"].y.round(4)

merged = extractions_dataset.merge(original_dataset, on=["round_x", "round_y"],
                                   how="left", suffixes=("", "_original"))

croptypes = merged["croptype"].unique()

duplicated_list = []
duplicate_ids = []

for croptype in croptypes:
    merged_c = merged[merged["croptype"] == croptype]
    if croptype in list(duplicate_mapping_final.keys()):
        print("duplicating croptype:", croptype)
        sample_ids = merged_c["sample_id"].unique()
        duplicates = duplicate_mapping_final[croptype]
        duplicate_ids.append(sample_ids)
        for duplic in duplicates:
            merged_d = merged_c.copy()
            merged_d["croptype"] = duplic
            merged_d["ewoc_code"] = int(duplicate_croptype_mapping[duplic])
            duplicated_list.append(merged_d)
    else:
        duplicated_list.append(merged_c)

duplicate_ids = [item for sublist in duplicate_ids for item in sublist]

result = gpd.GeoDataFrame(pd.concat(duplicated_list, ignore_index=True),
                          geometry="geometry",crs=extractions_dataset.crs)
result = result.loc[:,list(extractions_dataset.columns)]
result = result.drop(columns=["round_x", "round_y"])

#write duplicate_ids to a json file
duplicate_ids_file = os.path.join(activation_folder, "mixed_cropping",
                                  f"duplicate_ids_{activation}.json")
output_file = os.path.join(activation_folder, "mixed_cropping",
                           f"worldcereal_merged_extractions_{activation}_duplicated.parquet")

result.to_parquet(output_file, index=False)
print(f"Duplicated extractions saved to {output_file}")

with open(duplicate_ids_file, "w") as f:
    dump(duplicate_ids, f)
print(f"Duplicate IDs saved to {duplicate_ids_file}")
