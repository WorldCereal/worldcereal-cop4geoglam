
import os

import geopandas as gpd
import numpy as np
import pandas as pd


def identifyCrops(row, crop_dict):
    #make a list of the crops mentioned in the crop type column (can be more than one)
    croptypes = row['croptype'].split(" ")
    croptypes = [ct.strip() for ct in croptypes if ct.strip() != ""]
    croptypes = list(set(croptypes))
    return(croptypes)

def mapCropList(croplist, crop_dict):
    crops_mapped = []
    for yc in croplist:
        found = False
        for crop_key, crop_values in crop_dict.items():
            if yc in crop_values:
                crops_mapped.append(crop_key)
                found = True
                break
        if not found:
            crops_mapped.append("other")
    return crops_mapped

def calculateMembership(row,crop_dict,dominant_membership = 0.8):

    croptypes = row['croptype_list']
    #create empty membership array
    membership = np.zeros(len(crop_dict)+1, dtype=np.float32)

    yes_crops = []
    allcrops = []
    for crop in croptypes:
        if "yes" in crop:
            yes_crops.append(crop.replace("yes_",""))
        else:
            allcrops.append(crop)

    yes_crops_mapped = mapCropList(yes_crops, crop_dict)
    allcrops_mapped = mapCropList(allcrops, crop_dict)

    if len(yes_crops_mapped) > 0:
        for yc in yes_crops_mapped:
            if yc in crop_dict.keys():
                idx = list(crop_dict.keys()).index(yc)
                membership[idx] = np.round(dominant_membership,2)
            else:
                membership[-1] = np.round(dominant_membership,2)

    membership = np.round(membership,2)

    rem_membership = np.round(1 - np.sum(membership),2)
    #remove already assigned crops from allcrops_mapped, but not if other
    #check if other in both yes_crops_mapped and allcrops_mapped
    if "other" in yes_crops_mapped and "other" in allcrops_mapped:
        allcrops_mapped = allcrops_mapped.copy()
    else:
        allcrops_mapped = [ac for ac in allcrops_mapped if ac not in yes_crops_mapped]
    if len(allcrops_mapped) > 0:
        for ac in allcrops_mapped:
            if ac in crop_dict.keys():
                idx = list(crop_dict.keys()).index(ac)
                membership[idx] += np.round(rem_membership / len(allcrops_mapped),3)
            else:
                membership[-1] += np.round(rem_membership / len(allcrops_mapped), 3)

    #normalize membership to sum to 1
    membership = membership / np.sum(membership)
    membership = np.round(membership, 2)

    return membership


if __name__ == "__main__":

    activation = "mozambique"

    crop_dict = {
        "maize": ["maize"],
        "rice": ["rice"],
        "soybean": ["soya_beans"],
        "sesame": ["sesame","sesame_1"],
        "cassava": ["cassave"],
        "cowpea": ["cow_peas","cowpeas"],
        "sweet_potato": ["sweet_potatoes"],
        "pigeon_pea": ["pigeon_pea","Feijão-boer"],
        "sugarcane": ["sugarcane"]
    }

    base_folder = "/vitodata/worldcereal/data/COP4GEOGLAM/"
    activation_folder = os.path.join(base_folder, activation)

    #Note that the location of this geopackage still needs to be adapted to also exist on the worldcereal mount.
    original_gpkg = os.path.join(activation_folder,"refdata","original","moz_results_2025.gpkg")

    originals = gpd.read_file(original_gpkg)
    #drop rows with no crop type
    originals = originals[~originals['croptype'].isna()]

    #remove fallows
    originals = originals[originals['croptype']!="fallow_yes"]

    #remove agroforestry
    originals = originals[originals["trees_in_cropfield"] != "trees_yes"]

    originals['croptype_list'] = originals.apply(lambda row: identifyCrops(row, crop_dict), axis=1)
    originals['membership'] = originals.apply(lambda row: calculateMembership(row, crop_dict), axis=1)

    #open extractions file
    extractions = os.path.join(activation_folder, "trainingdata","worldcereal_merged_extractions_v1.parquet",
                               "ref_id=2025_MOZ_COPERNICUS4GEOGLAM_POINT_110_harmonized_with_EXP_POINTS",
                               "2025_MOZ_COPERNICUS4GEOGLAM_POINT_110_harmonized_with_EXP_POINTS_0.parquet")
    extractions_file = pd.read_parquet(extractions)
    extractions_file["id_ssu"] = [sample_id.split("_")[5] + "_" + sample_id.split("_")[6] for sample_id in extractions_file["sample_id"]]

    #merge membership info into extractions
    extractions_file = extractions_file.merge(originals[['id_ssu','landuse','croptype','membership']], on='id_ssu', how='left')

    originals['membership'] = originals.apply(lambda row: np.round(row["membership"],2), axis=1)

    #save to file
    extractions_file.to_parquet(os.path.join(activation_folder, "trainingdata","worldcereal_merged_extractions_with_membership.parquet"), index=False)

    extractions_file = extractions_file.dropna(subset=['membership'])

    summed_membership = np.sum(np.stack(originals['membership'].values), axis=0)

    #add names to summed membership
    crop_names = list(crop_dict.keys()) + ["other"]
    summed_membership_dict = dict(zip(crop_names, summed_membership))
    print("Summed membership across all samples:")
    print(summed_membership_dict)
