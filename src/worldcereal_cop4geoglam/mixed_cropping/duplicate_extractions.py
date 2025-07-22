'''
Main Code to duplicate extractions for mixed cropping points.
#►Copernicus4GEOGLAM
'''

import os
import geopandas as gpd

activation = "moldova"

base_folder = "/vitodata/worldcereal/data/COP4GEOGLAM/"
activation_folder = os.path.join(base_folder, activation)
extractions = os.path.join(activation_folder, "trainingdata","worldcereal_merged_extractions.parquet")

#Note that the location of this geopackage still needs to be adapted to also exist on the worldcereal mount.
original_gpkg = "/data/users/Public/koendevos/Copernicus4GEOGLAM/Moldova/mda_results_2025_RDM.gpkg"

extractions_dataset = gpd.read_parquet(extractions)
original_dataset = gpd.read_file(original_gpkg)

extractions_dataset["round_x"] = extractions_dataset["geometry"].x.round(4)
extractions_dataset["round_y"] = extractions_dataset["geometry"].y.round(4)

original_dataset["round_x"] = original_dataset["geometry"].x.round(4)
original_dataset["round_y"] = original_dataset["geometry"].y.round(4)

merged = extractions_dataset.merge(original_dataset, on=["round_x", "round_y"], how="left", suffixes=("", "_original"))





print("t")

