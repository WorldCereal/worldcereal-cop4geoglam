
import gc
import glob
import json
import os

import numpy as np
import rasterio
from tqdm import tqdm


def preprocess_data(data):
        if isinstance(data, dict):
            return {k: preprocess_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [preprocess_data(v) for v in data]
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data

def createLCDistribution(cropland_files, landcover, out_folder):

    bare_distribution = {i: 0 for i in range(101)}
    built_up_distribution = {i: 0 for i in range(101)}
    grasslands_distribution = {i: 0 for i in range(101)}
    permanent_crops_distribution = {i: 0 for i in range(101)}
    shrublands_distribution = {i: 0 for i in range(101)}
    temporary_crops_distribution = {i: 0 for i in range(101)}
    trees_distribution = {i: 0 for i in range(101)}
    water_distribution = {i: 0 for i in range(101)}
    wetlands_distribution = {i: 0 for i in range(101)}

    lc_index = {
        "bare_sparsely_vegetated": 0,
        "built_up": 1,
        "grasslands": 2,
        "permanent_crops": 3,
        "shrublands": 4,
        "temporary_crops": 5,
        "trees": 6,
        "water": 7,
        "wetlands": 8
    }

    for cf in tqdm(cropland_files,desc="Processing cropland files"):
        with rasterio.open(cf) as src:
            data = src.read()

            #for which band is the max value
            max_band = data.argmax(axis=0)
            bare_mask = (max_band == 0)
            build_up_mask = (max_band == 1)
            grasslands_mask = (max_band == 2)
            permanent_crops_mask = (max_band == 3)
            shrublands_mask = (max_band == 4)
            temporary_crops_mask = (max_band == 5)
            trees_mask = (max_band == 6)
            water_mask = (max_band == 7)
            wetlands_mask = (max_band == 8)


            # Update distributions using np.unique with counts
            def update_distribution(mask, distribution, index_lc):
                #which index of keys of all_cropland_distr corresponds to the lc?
                values, counts = np.unique((data[index_lc][mask] * 100).astype(int), return_counts=True)
                for value, count in zip(values, counts):
                    distribution[value] += count

            update_distribution(bare_mask, bare_distribution,lc_index[landcover])
            update_distribution(build_up_mask, built_up_distribution,lc_index[landcover])
            update_distribution(grasslands_mask, grasslands_distribution,lc_index[landcover])
            update_distribution(permanent_crops_mask, permanent_crops_distribution,lc_index[landcover])
            update_distribution(shrublands_mask, shrublands_distribution,lc_index[landcover])
            update_distribution(temporary_crops_mask, temporary_crops_distribution,lc_index[landcover])
            update_distribution(trees_mask, trees_distribution,lc_index[landcover])
            update_distribution(water_mask, water_distribution,lc_index[landcover])
            update_distribution(wetlands_mask, wetlands_distribution,lc_index[landcover])


    all_cropland_distr = {
        "bare_sparsely_vegetated": bare_distribution,
        "built_up": built_up_distribution,
        "grasslands": grasslands_distribution,
        "permanent_crops": permanent_crops_distribution,
        "shrublands": shrublands_distribution,
        "temporary_crops": temporary_crops_distribution,
        "trees": trees_distribution,
        "water": water_distribution,
        "wetlands": wetlands_distribution
    }

    processed_data = preprocess_data(all_cropland_distr)

    #save as json
    with open(os.path.join(out_folder,f"{landcover}_distributions.json"),"w") as f:
        json.dump(processed_data,f)


if __name__ == "__main__":

    activation = "mozambique"
    main_folder = "/vitodata/worldcereal/data/COP4GEOGLAM/"

    production = "v3_landcover"

    threshold = 0.5

    threshold_dict = {
        "bare_sparsely_vegetated" : 0.5,
        "built_up": 0.5,
        "grasslands": 0.5,
        "permanent_crops": 0.5,
        "shrublands": 0.5,
        "temporary_crops": 0.5,
        "trees": 0.5,
        "water": 0.5,
        "wetlands": 0.5
    }

    ## -----
    act_folder = os.path.join(main_folder, activation)
    production_folder = os.path.join(act_folder,"production",production,"raw")
    out_folder = os.path.join(act_folder,"fuzzy_test")

    cropland_files = glob.glob(os.path.join(production_folder,"MOZ*","*.tif"))

    lc_count_dict = {
        "bare_sparsely_vegetated": 0,
        "built_up": 0,
        "grasslands": 0,
        "permanent_crops": 0,
        "shrublands": 0,
        "temporary_crops": 0,
        "trees": 0,
        "water": 0,
        "wetlands": 0
    }

    lc_count_array = np.zeros(len(lc_count_dict),dtype=np.int64)

    for cf in tqdm(cropland_files,desc="Processing cropland files"):
        if cf == cropland_files[19]:
            print("t")
        with rasterio.open(cf) as src:
            data = src.read()

            #for which band is the max value
            max_band = data.argmax(axis=0)
            unique, counts = np.unique(max_band, return_counts=True)
            for u, c in zip(unique, counts):
                lc_name = list(lc_count_dict.keys())[u]
                lc_count_dict[lc_name] += c
            del data
        gc.collect()

    print(lc_count_dict)
    #save count_dict
    with open(os.path.join(out_folder,"lc_count_dict.json"),"w") as f:
        json.dump(preprocess_data(lc_count_dict),f)
