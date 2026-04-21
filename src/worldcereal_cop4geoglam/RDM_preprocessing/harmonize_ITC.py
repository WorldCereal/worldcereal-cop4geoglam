#file to harmonize ITC file with COP4GEOGLAM data

import os

import geopandas as gpd
import numpy as np
import pandas as pd


def harmonize_ITC(itc_file_name,output_name,activation = "mozambique_pm",overwrite=False):

    ## do not change this--
    base_dir = "/vitodata/worldcereal/data/COP4GEOGLAM/"

    ref_data_dir = os.path.join(base_dir,activation,"refdata")
    original_ref_data_dir = os.path.join(ref_data_dir,"original")
    harmonized_ref_data_dir = os.path.join(ref_data_dir,"harmonized")

    out_file = os.path.join(harmonized_ref_data_dir,output_name)

    if not os.path.exists(out_file) or overwrite:

        #Load ITC data
        itc_data = gpd.read_file(os.path.join(original_ref_data_dir,itc_file_name))

        itc_harm = itc_data.copy()
        #adjust to the same CRS as the other datasets (4326)
        itc_harm = itc_harm.to_crs(epsg=4326)

        #convert multipoint geometries to point geometries, keeping the original attributes
        itc_harm = itc_harm.explode(index_parts=False).reset_index(drop=True)

        itc_harm["id_psu"] = "itc_"+itc_harm["plotid"].astype(str)
        itc_harm["id_ssu"] = "itc_"+itc_harm["gid"].astype(str)
        itc_harm["sample_id"] = "2025_MOZ_ITC_POINT_110_" + itc_harm["id_ssu"]+ "_" + itc_harm.index.astype(str)

        #assuming only cropland points in the dataset
        itc_harm["lu_code"] = 3
        itc_harm["landuse"] = "agriculture"

        #set value for box_id, not required at any later stage, but is considered to allow to merge the datasets
        itc_harm["box_id"] = 42

        #set valid time to peak of the season, also an assumption- we are using the same date as for the Cop4GEOGLAM settings.
        itc_harm["valid_time"] = "2025-04-01"

        #setting extract to 1
        itc_harm["extract"] = 1

        itc_harm["croptype"] = None
        itc_harm["cropping_pattern"] = "mixed_cropping"
        itc_harm["dominant_crop"] = None
        itc_harm["dominant_percentage"] = None
        itc_harm["trees_in_cropfield"] = None

        itc_geoglam_dict_crops = {
            "corn": "maize",
            "beans": "pigeon_pea",
            "cassava": "cassave",
            "large peanut": "groundnuts",
            "sugarcane": "sugarcane",
            "sorghum": "sorghum",
            "other vegetable": "other_vegetable",
            "soybean": "soya_beans",
            "rice": "rice",
            "small peanut": "groundnuts",
            "sesame":"sesame",
            "okra": "okra",
            "orange-fleshed sweet potato": "sweet_potatoes",
            "squash": "pumpkin",
            "yam or amadumbe": "yam",
            "non-orange-fleshed sweet potato": "sweet_potatoes",
            "pepper": "pepper"
        }

        #now harmonizing the crops and adding trees_in_cropfield using notes
        for i, row in itc_harm.iterrows():
            croplist = row["croplist_en"]
            crops_row = croplist.split(",")
            crops_row = [itc_geoglam_dict_crops.get(c.strip(), "other") for c in crops_row]
            percentagelist = row["percentagelist"].replace("{","").replace("}","").split(",")
            percentages_row = [float(p) for p in percentagelist]
            #check if a crop is twice in the croplist, if so, add the percentages together
            if len(np.unique(crops_row)) != len(percentages_row):
                crop_percentages = {}
                for c, p in zip(crops_row, percentages_row):
                    if c in crop_percentages:
                        crop_percentages[c] += p
                    else:
                        crop_percentages[c] = p
                crops_row = list(crop_percentages.keys())
                percentages_row = list(crop_percentages.values())

            row_crops = list(zip(crops_row, percentages_row))
            if len(row_crops) == 1:
                if percentages_row[0] == 100:
                    itc_harm.at[i, "cropping_pattern"] = "mono_culture"
                    itc_harm.at[i,"croptype"] = crops_row[0]
                else:
                    itc_harm.at[i,"dominant_percentage"] = percentages_row[0]
                    itc_harm.at[i,"dominant_crop"] = crops_row[0]
                    itc_harm.at[i,"croptype"] = crops_row[0]
            else:
                #identify dominant crop
                dominant_crop = row_crops[np.argmax(percentages_row)][0]
                #order the other crops by alphabetic order
                other_crops = sorted([c for c in crops_row if c != dominant_crop])
                itc_harm.at[i,"dominant_crop"] = dominant_crop
                itc_harm.at[i,"dominant_percentage"] = row_crops[np.argmax(percentages_row)][1]
                itc_harm.at[i,"croptype"] = f"yes_{dominant_crop}"
                for other_crop in other_crops:
                    itc_harm.at[i,"croptype"] += f" {other_crop}"

            #adding information on trees in cropfield using notes
            if row["notes"] is not None:
                if "oil palm" in row["notes"].lower():
                    itc_harm.at[i,"trees_in_cropfield"] = "trees_yes"

        #link to ewoc mapping
        ewoc_file = os.path.join(harmonized_ref_data_dir,"ewoc_mapping_moz_results_2025.csv")
        itc_ewoc_file = ewoc_file.replace(".csv","_itc.csv")
        ewoc_file = pd.read_csv(ewoc_file)
        itc_ewoc_file = pd.read_csv(itc_ewoc_file, delimiter = ";")

        ewoc = pd.concat([ewoc_file.drop(["count"],axis=1),itc_ewoc_file], ignore_index=True)

        itc_harm = itc_harm.merge(ewoc[["croptype","ewoc_code","ewoc_name"]], left_on="croptype", right_on="croptype", how="left")
        #identify for which rows there is no match in the already-existing ewoc mapping
        no_match = itc_harm[itc_harm["ewoc_code"].isna()]["croptype"].unique()
        if len(no_match)>0:
            print("The following croptypes from the ITC dataset do not have a match in the existing ewoc mapping and are deleted from the dataset:")
            print(no_match)
            itc_harm = itc_harm[~itc_harm["croptype"].isin(no_match)]

        keep_cols = [
            "id_psu",
            "ewoc_name",
            "ewoc_code",
            "lu_code",
            "id_ssu",
            "box_id",
            "valid_time",
            "sample_id",
            "extract",
            "landuse",
            "cropping_pattern",
            "croptype",
            "trees_in_cropfield",
            "dominant_crop",
            "dominant_percentage",
            "geometry"
        ]

        itc_harm = itc_harm[keep_cols]

        merged_points = itc_harm.copy()

        merged_points["sampling_ewoc_code"] = merged_points["ewoc_code"]
        merged_points["h3_l3_cell"]="unspecified"
        merged_points["irrigation_status"]=0
        merged_points["quality_score_lc"] = 1
        merged_points["quality_score_ct"] = 1

        #save the harmonized dataset as a parquet file
        merged_points.to_parquet(os.path.join(harmonized_ref_data_dir,output_name), index=False)


if __name__ == "__main__":

    itc_file_name = "ITC_InSitu.gpkg"
    output_name = "2025_MOZ_ITC_POINT_110_harmonized.geoparquet"

    # Define the path to the input shapefile
    harmonize_ITC(itc_file_name=itc_file_name,
                    output_name=output_name,
                    activation = "mozambique_pm")
