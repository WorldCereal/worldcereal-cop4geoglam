'''
Script that creates the initial json for a use case when the 
ewocMapped.csv file is provided by the RDM
--> Copernicus4GEOGLAM

This drafts an initial json, but make sure to check for its validity for the process at hand.
Also make sure to remove all options that are not used in the actual mapping.
'''

import os
from json import dumps, load

import pandas as pd


def getLevel(row):
    # Check for the most detailed level (from level_5 to level_1)
    if pd.notna(row["level_5"]):
        return "level_5"
    elif pd.notna(row["level_4"]):
        return "level_4"
    elif pd.notna(row["level_3"]):
        return "level_3"
    elif pd.notna(row["level_2"]):
        return "level_2"
    elif pd.notna(row["level_1"]):
        return "level_1"
    else:
        raise ValueError("Row does not have a valid level")

def findKey(dict, value):
    # Find the key in the dictionary that corresponds to a given value
    for key, val in dict.items():
        if value in val:
            return key
    return None

def CROPTYPE_EWOC(joined,crop_dict):

    #link each Label in joined to the corresponding key from dict
    CROPTYPE_EWOC_dict = {}
    for i, row in joined.iterrows():
        croptype_ewoc = row["ewoc"]
        crop_key = findKey(crop_dict, row["Label"])
        if crop_key is not None:
            CROPTYPE_EWOC_dict[str(croptype_ewoc)] = crop_key

    #Sort dict on keys
    CROPTYPE_EWOC_dict = dict(sorted(CROPTYPE_EWOC_dict.items(), key=lambda item: int(item[0])))
    return(CROPTYPE_EWOC_dict)

def LANDCOVER(joined,example,remove_lc):
    #for each croptype, get the landcover label
    LANDCOVER_dict = {}

    for i, row in joined.iterrows():
        if row["Label"] not in remove_lc:
            croptype_ewoc = row["ewoc"]
            if croptype_ewoc in example["LANDCOVER10"].keys():
                LU_label = example["LANDCOVER10"][croptype_ewoc]
            else:
                LU_label = row["LC_label"]
                print(f"Warning: {row['Label']} not found in example json, using LC_label: {LU_label}, Remove?")
                #Ask user if this is ok or to remove
                user_input = input(f"   Is it ok to use {LU_label} for {row['Label']}- remove? (y/remove): ")
                if user_input.lower() == "remove":
                    continue                

            LANDCOVER_dict[str(croptype_ewoc)] = LU_label

    #Sort dict on keys
    LANDCOVER_dict = dict(sorted(LANDCOVER_dict.items(), key=lambda item: item[0]))

    return(LANDCOVER_dict)

folder = "/data/users/Public/koendevos/Copernicus4GEOGLAM/"
activation = "Moldova"

#Define the crop type classes for your final json and of which they consist.
#--> all the ones that are not included in this list will be removed from the final croptype mapping.
crop_dict = {
    "wheat": ["wheat_winter","wheat_spring"],
    "maize": ["maize"],
    "barley": ["winter_barley"],
    "sunflower": ["sunflower"],
    "rapeseed": ["rapeseed"],
    "soybean": ["soya_beans"],
    "vineyard": ["grapes"],
    "orchard": ["apple","no_dominant walnut plum none","pear","plum",
                 "cherry","Sour cherry orchard","walnut","Wax Cherry Orchard",
                 "Wax Cherry Orchard ","Wild Cherry Orchard"],
    "other_temporary_crops": ["no_dominant potatoes pumpkin none", "oats",
                              "peas","potatoes","Watermelon"],
    "other_permanent_crops": ["berries","Clary sage"],    
    }

#Identify which will rows will be removed for final landuse mapping
remove_lc = ["agriculture","fallow","no_dominant fodder_grass walnut none","no_dominant grapes fallow",
             "no_dominant grapes fodder_grass none","no_dominant grapes maize none none",
             "no_dominant grapes wheat_winter none none", "no_dominant maize apple none",
             "no_dominant maize fodder_grass none", "no_dominant maize grapes fodder_grass none",
             "no_dominant maize grapes none none", "no_dominant maize soya_beans fodder_grass none",
             "no_dominant maize sunflower fodder_grass none","no_dominant sunflower fodder_grass none",
             "no_dominant sunflower fodder_grass onion none", "yes_ maize walnut none", "yes_ wheat_winter maize fodder_grass none",
             "yes_maize fodder_grass none", "yes_maize grapes none","yes_wheat walnut none",
             "no_dominant sunflower walnut none","no_dominant wheat_winter apple none",
             "no_dominant wheat_winter walnut none", "yes_grapes maize sunflower none",
             "yes_grapes sunflower none","yes_maize grapes none none","yes_sunflower wheat_winter grapes none none"]

act_folder = os.path.join(folder, activation)

ewocMapped = pd.read_csv(os.path.join(act_folder, f"ewocMapped_{activation}.csv"))
ewocMapped["Code"] = ewocMapped["Code"].astype(str)
wcLegend =     wcLegend = pd.read_csv(
        os.path.join(act_folder, "WorldCereal_LC_CT_legend_latest.csv"),
        sep=";",  # Adjust if the delimiter is not a comma
        on_bad_lines="skip"  # Skip problematic rows
    )
wcLegend["ewoc"] = [ewoc.replace("-","") for ewoc in wcLegend["ewoc_code"]]

joined = ewocMapped.merge(wcLegend, left_on="Code", right_on="ewoc", how="left")

CROPTYPE_EWOC_dict = CROPTYPE_EWOC(joined,crop_dict)

json_example = load(open(os.path.join(folder, "class_mappings_example.json"), "r"))

LANDCOVER_dict = LANDCOVER(joined,json_example,remove_lc)

superdict = {
    "LANDCOVER10": LANDCOVER_dict,
    f"CROPTYPE_{activation}": CROPTYPE_EWOC_dict
}

json_path = os.path.join(act_folder, f"class_mapping_{activation}_prelim.json")
with open(json_path, "w") as f:
    f.write(dumps(superdict, indent=4))
    
#Also write this to the current folder (= script folder)
current_json_path = os.path.join(os.path.dirname(__file__), f"class_mapping_{activation}_prelim.json")
with open(current_json_path, "w") as f:
    f.write(dumps(superdict, indent=4))
print(f"JSON file created at: {json_path}")