'''
Script to create the duplicate mapping file
'''

import os
from json import dumps

import pandas as pd

folder = "/data/users/Public/koendevos/Copernicus4GEOGLAM/"
activation = "Mozambique"

ewoc = os.path.join(folder,activation,f"ewocMapped_{activation}.csv")
ewoc = pd.read_csv(ewoc)

crop_dict = {
    "maize": ["maize"],
    "rice": ["rice"],
    "soybean": ["soya_beans"],
    "sesame": ["sesame","sesame_1"],
    "cassava": ["cassave", "yes_cassave cassave"],
    "cowpea": ["cow_peas","cowpeas"],
    "sweet_potato": ["sweet_potatoes"],
    "pigeon_pea": ["pigeon_pea","Feijão-boer","yes_pigeon_pea pigeon_pea"],
    "sugarcane": ["sugarcane"]
}

def duplicate_original(ewoc):
    unique_labels = ewoc["Label"].unique()

    croptype_labels = {}

    for lab in unique_labels:
        lablist = [lab.replace("yes_","") for lab in lab.split(" ")]
        #remove duplicates from list
        lablist = list(set(lablist))
        croptype_labels[lab] = lablist

    return croptype_labels

def duplicate_final(ewoc,crop_dict):
    unique_labels = ewoc["Label"].unique()

    croptype_labels = {}

    for lab in unique_labels:
        lablist = [lab.replace("yes_","") for lab in lab.split(" ")]

        lablist_ad = []

        for labs in lablist:
            for crop_key, crop_values in crop_dict.items():
                if labs in crop_values:
                    lablist_ad.append(crop_key)
                    break
                else:
                    lablist_ad.append("other_crops")

        #remove duplicates from list
        lablist = list(set(lablist_ad))

        croptype_labels[lab] = lablist

    return croptype_labels


duplicate_original_file = duplicate_original(ewoc)
duplicate_final_file = duplicate_final(ewoc,crop_dict)

superdict = {
    "DUPLICATE_CROPTYPE_ORIGINAL": duplicate_original_file,
    "DUPLICATE_CROPTYPE_FINAL": duplicate_final_file
}

#save json file in folder of this script
json_path = os.path.join(os.path.dirname(__file__),
                         f"duplicate_mapping_{activation}_prelim.json")

with open(json_path, "w") as f:
    f.write(dumps(superdict, indent=4))

print("t")
