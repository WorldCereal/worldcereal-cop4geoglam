
import glob
import os

import geopandas as gpd
import pandas as pd
import xarray as xr
from tqdm import tqdm

pd.options.mode.chained_assignment = None

def getPSU_file(activation_folder,psu,test_set):
    local_debug_folder = os.path.join(activation_folder,
                                      "production","local_debug_duplicates")

    croptype_files = glob.glob(os.path.join(local_debug_folder,f"*{psu}_croptype.nc"))
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

        #extract the croptype array for the points
        for idx, row in psu_test.iterrows():
            x = row["geometry"].x
            y = row["geometry"].y

            psu_test.loc[idx, "probability_cassava"] = croptype_ds.sel(
                x=x, y=y, method="nearest")["probability_cassava"].values.item()
            psu_test.loc[idx, "probability_maize"] = croptype_ds.sel(
                x=x, y=y, method="nearest")["probability_maize"].values.item()
            psu_test.loc[idx, "probability_other_crops"] = croptype_ds.sel(
                x=x, y=y, method="nearest")["probability_other_crops"].values.item()
            psu_test.loc[idx, "probability_rice"] = croptype_ds.sel(
                x=x, y=y, method="nearest")["probability_rice"].values.item()
            psu_test.loc[idx, "probability_sweet_potato"] = croptype_ds.sel(
                x=x, y=y, method="nearest")["probability_sweet_potato"].values.item()
            psu_test.loc[idx, "probability_pigeon_pea"] = croptype_ds.sel(
                x=x, y=y, method="nearest")["probability_pigeon_pea"].values.item()
            psu_test.loc[idx, "probability_soybean"] = croptype_ds.sel(
                x=x, y=y, method="nearest")["probability_soybean"].values.item()
            psu_test.loc[idx, "probability_sesame"] = croptype_ds.sel(
                x=x, y=y, method="nearest")["probability_sesame"].values.item()

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

test_points = extractions_dataset.loc[extractions_dataset["sample_id"].isin(
    test_data_sample_id["sample_id"]),]
test_points["id_psu"] = [
    f"{sample_id.split('_')[5]}"
    for sample_id in test_points["sample_id"]
]
print(f"test points share: {len(test_points)/len(extractions_dataset)} ")

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
write_path = os.path.join(activation_folder,"mixed_cropping",
                          f"test_points_probabilities_{activation}_duplicates.parquet")
joined_no_na.to_parquet(write_path)

crops = joined_no_na["croptype"].unique()
#remove none
crops = [crop for crop in crops if crop is not None]

pigeon_pea = [crop for crop in crops if "pigeon_pea" in crop]



print("t")
