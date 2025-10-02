
import glob
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

pd.options.mode.chained_assignment = None

def getPixelIndices(ds, points):
    x_indices = []
    y_indices = []

    x_coords = ds["x"].values
    y_coords = ds["y"].values

    for idx, row in points.iterrows():
        x = row["geometry"].x
        y = row["geometry"].y


        # Extract the raster indices of the nearest pixel in terms of array indices
        x_index = np.abs(x_coords - x).argmin()
        y_index = np.abs(y_coords - y).argmin()

        x_indices.append(x_index)
        y_indices.append(y_index)

    return x_indices, y_indices


def getPSU_file(activation_folder,psu,test_set):
    local_debug_folder = os.path.join(activation_folder,
                                      "production","local_no_mixed_no_agroforestry")

    croptype_files = glob.glob(os.path.join(local_debug_folder,
                                            f"*{psu}_croptype_masked.nc"))
    original_crs = test_set.crs
    psu_test = test_set[test_set["id_psu"]==psu]

    psu_test["probability_cassava"] = None
    psu_test["probability_maize"] = None
    psu_test["probability_other_crops"] = None
    psu_test["probability_rice"] = None
    psu_test["probability_sweet_potato"] = None
    psu_test["probability_pigeon_pea"] = None
    psu_test["probability_soybean"] = None
    psu_test["probability_sesame"] = None

    if len(croptype_files)==0:
        print(f"No croptype files found for {psu}")
    else:
        #open croptype netcdf
        croptype_ds = xr.open_dataset(croptype_files[0])

        #reproject to the croptype ds crs
        psu_test = psu_test.to_crs(croptype_ds["spatial_ref"].attrs["spatial_ref"])

        prob_cassava = croptype_ds["probability_cassava"].values
        prob_maize = croptype_ds["probability_maize"].values
        prob_other_crops = croptype_ds["probability_other_crops"].values
        prob_rice = croptype_ds["probability_rice"].values
        prob_sweet_potato = croptype_ds["probability_sweet_potato"].values
        prob_pigeon_pea = croptype_ds["probability_pigeon_pea"].values
        prob_soybean = croptype_ds["probability_soybean"].values
        prob_sesame = croptype_ds["probability_sesame"].values

        # Get pixel indices for all points
        x_indices, y_indices = getPixelIndices(croptype_ds, psu_test)

        probability_cassava = []
        probability_maize = []
        probability_other_crops = []
        probability_rice = []
        probability_sweet_potato = []
        probability_pigeon_pea = []
        probability_soybean = []
        probability_sesame = []

        #extract the croptype array for the points
        for idx, x in enumerate(x_indices):
            y = y_indices[idx]

            probability_cassava.append(prob_cassava[x,y])
            probability_maize.append(prob_maize[x,y])
            probability_other_crops.append(prob_other_crops[x,y])
            probability_rice.append(prob_rice[x,y])
            probability_sweet_potato.append(prob_sweet_potato[x,y])
            probability_pigeon_pea.append(prob_pigeon_pea[x,y])
            probability_soybean.append(prob_soybean[x,y])
            probability_sesame.append(prob_sesame[x,y])

        psu_test["probability_cassava"] = probability_cassava
        psu_test["probability_maize"] = probability_maize
        psu_test["probability_other_crops"] = probability_other_crops
        psu_test["probability_rice"] = probability_rice
        psu_test["probability_sweet_potato"] = probability_sweet_potato
        psu_test["probability_pigeon_pea"] = probability_pigeon_pea
        psu_test["probability_soybean"] = probability_soybean
        psu_test["probability_sesame"] = probability_sesame
        croptype_ds.close()

        psu_test = psu_test.to_crs(original_crs)

    return psu_test

activation = "mozambique"

activation_folder = f"/vitodata/worldcereal/data/COP4GEOGLAM/{activation}"

data_dir = os.path.join(activation_folder,"trainingdata","data_split")
test_data_sample_id = pd.read_csv(os.path.join(data_dir,f"test_ids_{activation}.csv"))

extractions = glob.glob(os.path.join(activation_folder,
                           "trainingdata","worldcereal_merged_extractions.parquet",
                           "*EXP*",
                           "*.parquet"))[0]

original_dataset = gpd.read_parquet(
    glob.glob(os.path.join(activation_folder,
                           "refdata","harmonized","*no_agroforestry.parquet"))[0])

extractions_dataset = gpd.read_parquet(extractions)
extractions_dataset["id_ssu"] = [
    f"{sample_id.split('_')[5]}_{sample_id.split('_')[6]}"
    for sample_id in extractions_dataset["sample_id"]
]


mixed_points = extractions_dataset.loc[extractions_dataset["ewoc_code"]>=1114000000,]
mixed_points = mixed_points.loc[mixed_points["ewoc_code"]<1115000000,]

test_points = extractions_dataset.loc[extractions_dataset["sample_id"].isin(
    test_data_sample_id["sample_id"]),]
test_points = pd.concat([test_points,test_points],ignore_index=True)
test_points["id_psu"] = [
    f"{sample_id.split('_')[5]}"
    for sample_id in test_points["sample_id"]
]
print(f"test points share: {len(test_points)/len(extractions_dataset)} ")

test_points = test_points.loc[test_points["ewoc_code"]<2000000000,]
test_points = test_points.loc[test_points["ewoc_code"]>1000000000,]

test_points = test_points.reset_index(drop=True)

test_psus = test_points["id_psu"].unique()

PSU_updated = []

for psu in tqdm(test_psus):
    psu_file = getPSU_file(activation_folder,psu,test_points)
    PSU_updated.append(psu_file)

test_points_updated = pd.concat(PSU_updated, ignore_index=True)

joined = test_points_updated.merge(
    original_dataset[["id_ssu","croptype"]],
    on="id_ssu",
    how="left"
)

joined_no_na = joined.dropna(subset=["probability_sesame"])
joined_no_na = joined_no_na.loc[joined_no_na["probability_sesame"]!=65535,]

write_path = os.path.join(activation_folder,"mixed_cropping",
                          f"test_points_probabilities_{activation}_duplicates.parquet")
joined_no_na.to_parquet(write_path)

crops = joined_no_na["croptype"].unique()
#remove none
crops = [crop for crop in crops if crop is not None]

pigeon_pea = [crop for crop in crops if "pigeon_pea" in crop]



print("t")
