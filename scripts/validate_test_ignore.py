import os
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
)

THRESHOLD_DEFAULT = 0.5

THRESHOLDS = {
    "maize": 0.4, #default=0.35
    "rice": 0.24, #default =0.24
    "soybean": 0.2, #default=0.16
    "sesame": 0.6, #default=0.45
    "cassava": 0.25, #default=0.30
    "sweet_potato": 0.2, #default=0.25
    "pigeon_pea": 0.2, #default=0.24
}

CLASSES_DICT = {
    "band_names": {
        "1": "maize",
        "2": "rice",
        "3": "pigeon_pea",
        "4": "soybean",
        "5": "sesame",
        "6": "sweet_potato",
        "7": "cassava",
    },
    "single_crop_classes": {
        1: "maize",
        2: "rice",
        3: "soybean",
        4: "sesame",
        5: "cassava",
        6: "sweet_potato",
        7: "pigeon_pea",
    },
    "mixed_crops_classes": {
        15: "maize-cassava",
        17: "maize-pigeon_pea",
        57: "cassava-pigeon_pea",
        157: "maize-cassava-pigeon_pea",
        200: "other_crop/mixtures",
    },
}

band_names = cast(dict[str, str], CLASSES_DICT["band_names"])

test_samples_file = "2025_MOZ_COPERNICUS4GEOGLAM_ITC_POINT_EXP_POLY_MERGED_PGP_remove_45_withMaize_Maize_remove_20_test_sample_ids"
ignore_samples_file = "2025_MOZ_COPERNICUS4GEOGLAM_ITC_POINT_EXP_POLY_MERGED_PGP_remove_45_withMaize_Maize_remove_20_ignore_sample_ids"

# Load the parquet file
val_dir = "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique_pm/validation/"
parquet_file_path = '/vitodata/worldcereal/data/COP4GEOGLAM/mozambique_pm/validation/MOZ_PM_predictions_all_points_masked.parquet'
data = gpd.read_parquet(parquet_file_path)

# Load the CSV files for train/val/test sample_ids
csv_folder = "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique_pm/trainingdata/data_split/"
test_samples_file = os.path.join(csv_folder, test_samples_file + ".csv")
ignore_samples_file = os.path.join(csv_folder, ignore_samples_file + ".csv")

sample_ids_test = pd.read_csv(test_samples_file)["sample_id"].tolist()
sample_ids_ignore = pd.read_csv(ignore_samples_file)["sample_id"].tolist()

test_samples = data[data['sample_id'].isin(sample_ids_test)]
ignore_samples = data[data['sample_id'].isin(sample_ids_ignore)]

test_samples = test_samples.drop(columns=['geometry']).groupby('sample_id').first().reset_index()
ignore_samples = ignore_samples.drop(columns=['geometry']).groupby('sample_id').first().reset_index()

#rename columns with name 1,2,3,4,5,6,7 to single crop_classes
test_samples = test_samples.rename(columns=CLASSES_DICT["band_names"])
ignore_samples = ignore_samples.rename(columns=CLASSES_DICT["band_names"])

crops = list(band_names.values())
keep_cols = ["sample_id", "ewoc_code"] + crops

test_samples = test_samples[keep_cols]
ignore_samples = ignore_samples[keep_cols]

harm_folder = "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique_pm/refdata/harmonized"
harm_C4G_file = os.path.join(harm_folder,"2025_MOZ_COPERNICUS4GEOGLAM_POINT_110_harmonized_with_EXP_POINTS_POLY.geoparquet")
harm_ITC_file = os.path.join(harm_folder,"2025_MOZ_ITC_POINT_110_harmonized.geoparquet")

harm_C4G = gpd.read_parquet(harm_C4G_file)
harm_ITC = gpd.read_parquet(harm_ITC_file)

test_C4G = pd.merge(test_samples, harm_C4G[["sample_id","ewoc_code","landuse","cropping_pattern","croptype","trees_in_cropfield"]], on=["sample_id","ewoc_code"], how="inner")
test_ITC = pd.merge(test_samples, harm_ITC[["sample_id","ewoc_code","landuse","cropping_pattern","croptype","trees_in_cropfield"]], on=["sample_id","ewoc_code"], how="inner")
ignore_C4G = pd.merge(ignore_samples, harm_C4G[["sample_id","ewoc_code","landuse","cropping_pattern","croptype","trees_in_cropfield"]], on=["sample_id","ewoc_code"], how="inner")
ignore_ITC = pd.merge(ignore_samples, harm_ITC[["sample_id","ewoc_code","landuse","cropping_pattern","croptype","trees_in_cropfield"]], on=["sample_id","ewoc_code"], how="inner")

test_set = pd.concat([test_C4G, test_ITC], ignore_index=True)
ignore_set = pd.concat([ignore_C4G, ignore_ITC], ignore_index=True)

test_set["predicted_crop"] = ""
ignore_set["predicted_crop"] = ""

#if all zeroes, set to "254"
def assign_predicted_crop(row):
    #croplands check
    cropland_sum = np.sum(row[crops])
    if cropland_sum == 0:
        return "254"
    else:
        crop_value = ""
        if row["maize"] >= THRESHOLDS["maize"]*100:
            crop_value += "1"
        if row["rice"] >= THRESHOLDS["rice"]*100:
            crop_value += "2"
        if row["soybean"] >= THRESHOLDS["soybean"]*100:
            crop_value += "3"
        if row["sesame"] >= THRESHOLDS["sesame"]*100:
            crop_value += "4"
        if row["cassava"] >= THRESHOLDS["cassava"]*100:
            crop_value += "5"
        if row["sweet_potato"] >= THRESHOLDS["sweet_potato"]*100:
            crop_value += "6"
        if row["pigeon_pea"] >= THRESHOLDS["pigeon_pea"]*100:
            crop_value += "7"
        if crop_value == "":

            #fall back to argmax if no crop meets the threshold
            crop_max = str(row[crops].idxmax())
            single_crops_keys = list(CLASSES_DICT["single_crop_classes"].keys())
            single_crops_values = list(CLASSES_DICT["single_crop_classes"].values())
            crop_value = str(single_crops_keys[single_crops_values.index(crop_max)])

            #also check if there are crops with probabilities close to the argmax crop and assign that also.

            crop_prob = row[crop_max]
            #identify all crops where the prob is closer than 5% to the crop_prob of the argmax crop and assign them to the same class
            close_crops = [crop for crop in crops if abs(row[crop] - crop_prob) <= 5 and crop != crop_max]
            for close_crop in close_crops:
                close_crop_value = str(single_crops_keys[single_crops_values.index(close_crop)])
                crop_value += close_crop_value

            #crop_value = "200"
        mixed_crops = list(CLASSES_DICT["mixed_crops_classes"].keys())
        mixed_crops = [str(c) for c in mixed_crops]
        if len(crop_value) > 1:
            if crop_value not in mixed_crops:
                crop_value = crop_value
        return crop_value

test_set["predicted_crop"] = test_set.apply(assign_predicted_crop, axis=1)
ignore_set["predicted_crop"] = ignore_set.apply(assign_predicted_crop, axis=1)

#set croptype to "NA" if landuse is not "cropland"
test_set.loc[test_set["landuse"] != "agriculture", "croptype"] = "NA"
ignore_set.loc[ignore_set["landuse"] != "agriculture", "croptype"] = "NA"

set_to_200 = ["12","13","14","25","34","37","47","67"]
test_set.loc[test_set["predicted_crop"].isin(set_to_200), "predicted_crop"] = "200"
ignore_set.loc[ignore_set["predicted_crop"].isin(set_to_200), "predicted_crop"] = "200"

#remove samples with ewoc code "1000000000" from test set and assign them to "253"
test_set = test_set[~(test_set["ewoc_code"] == "1000000000")]
ignore_set = ignore_set[~(ignore_set["ewoc_code"] == "1000000000")]

#if all zeroes, set to "254"
def assign_true_crop(row):
    #croplands check
    other_crops = ["sugarcane","sorghum","sunflower","groundnut","cowpea","other","gram"]
    pred_crops = test_set["predicted_crop"].unique().tolist() + ignore_set["predicted_crop"].unique().tolist()
    croptype_label = row["croptype"]
    label = ""
    if "maize" in croptype_label:
        label += "1"
    if "rice" in croptype_label:
        label += "2"
    if "soy" in croptype_label:
        label += "3"
    if "sesame" in croptype_label:
        label += "4"
    if "cassav" in croptype_label:
        label += "5"
    if "sweet" in croptype_label:
        label += "6"
    if "pigeon" in croptype_label:
        label += "7"
    for crop in other_crops:
        if crop in croptype_label:
            label = "200"
    if label == "":
        if croptype_label == "NA":
            label = "254"
        else:
            label = "200"
    if label not in pred_crops:
        label = "200"
    return label

def assign_crop(row):
    #croplands check
    other_crops = ["sugarcane","sorghum","sunflower","groundnut","cowpea","other","gram"]
    croptype_label = row["croptype"]
    label = ""
    if "maize" in croptype_label:
        label += "1"
    if "rice" in croptype_label:
        label += "2"
    if "soy" in croptype_label:
        label += "3"
    if "sesame" in croptype_label:
        label += "4"
    if "cassav" in croptype_label:
        label += "5"
    if "sweet" in croptype_label:
        label += "6"
    if "pigeon" in croptype_label:
        label += "7"
    for crop in other_crops:
        if crop in croptype_label:
            label += "X"
    if label == "":
        if croptype_label == "NA":
            label = "254"
    return label


test_set["true_crop"] = test_set.apply(assign_true_crop, axis=1)
ignore_set["true_crop"] = ignore_set.apply(assign_true_crop, axis=1)

test_set["true_label"] = test_set.apply(assign_crop, axis=1)

#get class-specific accuracy, precision, recall, f1-score
f1_scores = classification_report(test_set["true_crop"], test_set["predicted_crop"], output_dict=True)
print("Classification Report for Test Set:")
print(classification_report(test_set["true_crop"], test_set["predicted_crop"]))

f1_scores = classification_report(ignore_set["true_crop"], ignore_set["predicted_crop"], output_dict=True)
print("Classification Report for Test Set:")
print(classification_report(ignore_set["true_crop"], ignore_set["predicted_crop"]))

count_test = test_set.groupby(["true_label","predicted_crop"]).size().unstack(fill_value=0)
count_ignore = ignore_set.groupby(["ewoc_code","predicted_crop"]).size().unstack(fill_value=0)

count_test.to_csv(os.path.join(val_dir,"test_set_croptype_vs_predicted_crop.csv"))
count_ignore.to_csv(os.path.join(val_dir,"ignore_set_croptype_vs_predicted_crop.csv"))
