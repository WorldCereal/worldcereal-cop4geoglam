
import numpy as np
import pandas as pd


def identifyCrops(row, crop_dict):
    #make a list of the crops mentioned in the crop type column (can be more than one)
    if pd.isna(row['croptype']):
        return []
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

    if len(croptypes) == 0:
        return None
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

def updateEWOC(row,landcover_dict):
    landuse = row['landuse']
    if landuse in landcover_dict.keys():
        return landcover_dict[landuse]
    else:
        return row["ewoc_code"]

def updateEWOCname(row,landcover_dict):
    landuse = row['landuse']
    if landuse in landcover_dict.keys():
        return landuse
    else:
        return row["ewoc_name"]


if __name__ == "__main__":


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

    landcover_dict = {
        "natural_shrubs": 3000000000,
        "forest": 4000000000,
        "natural_grassland": 2000000000,
        "rocks": 5000000000,
        "baresoil_sand": 5000000000,
        "waterway_ponds": 7000000000,
        "build_up": 6000000000,
        "swamp_reeds": 2002000000
    }


    data_file = "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique/refdata/harmonized/2025_MOZ_COPERNICUS4GEOGLAM_POINT_110_harmonized_with_EXP_POINTS.parquet"

    #open data file
    df = pd.read_parquet(data_file)

    df['croptype_list'] = df.apply(lambda row: identifyCrops(row, crop_dict), axis=1)
    df['membership'] = df.apply(lambda row: calculateMembership(row, crop_dict), axis=1)

    unique_landcover = df['landuse'].unique()

    df["ewoc_updated"] = df.apply(lambda row: updateEWOC(row, landcover_dict), axis=1)
    df["ewoc_name_updated"] = df.apply(lambda row: updateEWOCname(row, landcover_dict), axis=1)

    df["ewoc_code"] = df["ewoc_updated"]
    df["ewoc_name"] = df["ewoc_name_updated"]

    df = df.drop(columns=["ewoc_updated","ewoc_name_updated"])

    for crop in df["croptype"].unique():
        if crop is not None:
            unique_memberships = df[df["croptype"]==crop]["membership"].reset_index(drop=True)
            sum_membership = np.sum(unique_memberships)/len(unique_memberships)
            print(f"Crop: {crop}, unique memberships: {sum_membership}")

    total_sum_membership = np.sum(df["membership"].dropna().reset_index(drop=True))/len(df["membership"].dropna().reset_index(drop=True))

    share_ewoc_big = [str(ewoc)[0] for ewoc in df["ewoc_code"] if not pd.isna(ewoc)]
    df["ewoc_big"] = share_ewoc_big
    share_ewoc = df.groupby("ewoc_big").size() / len(df)

    #save updated file
    df.to_parquet(data_file.replace(".parquet","_with_membership_updatedEWOC.parquet"), index=False)

    print('t')
