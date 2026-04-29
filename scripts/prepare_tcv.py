
import json
import os
from typing import Any, Dict, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def createSummaryTable(df,group_cols,agg_cols):

    agg_dict = {col:"nunique" for col in agg_cols}
    df[agg_cols] = df[agg_cols].fillna("NA")
    df[group_cols] = df[group_cols].fillna("NA")
    summary_table = df.groupby(group_cols).agg(agg_dict).reset_index()

    return(summary_table)

def identifyDifficultPGP(difficult_percentage = 0.4):
    feature_distance_folder = "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique_pm/feature_distance"
    pgp_to_maize = pd.read_parquet(os.path.join(feature_distance_folder,"pigeon_pea_distance_to_maize.parquet"))

    threshold = pgp_to_maize["distance_to_maize"].quantile(difficult_percentage)
    #select the sample_ids of the pigeon pea points that are below this threshold, as these are the ones closest to maize in feature space and thus likely more difficult to classify
    difficult_pgp_samples = pgp_to_maize[pgp_to_maize["distance_to_maize"] <= threshold]["ssu_id"].unique().tolist()
    return(difficult_pgp_samples)

def identifySamplesWithTrees(activation,ref_id):

    activation_folder = os.path.join("/vitodata/worldcereal/data/COP4GEOGLAM/",activation)
    original_file = os.path.join(activation_folder,"refdata","harmonized")

    harm_file = os.path.join(original_file,f"{ref_id}.geoparquet")

    #identify for which samples the "trees_in_cropfield" column is set to "trees_yes", and save the sample ids to a csv file
    if os.path.exists(harm_file):
        harm_df = pd.read_parquet(harm_file)

        tree_samples = harm_df[harm_df["trees_in_cropfield"]=="trees_yes"]["sample_id"].unique()

        tree_samples_df = pd.DataFrame(tree_samples, columns=["sample_id"])

        tree_samples = tree_samples_df["sample_id"].tolist()

        #identify agroforestry where ewoc_code starts with '14' and add those sample ids to the list of tree_samples
        agroforestry_samples = harm_df[harm_df["ewoc_code"].astype(str).str.startswith("14")]["sample_id"].unique()
        agroforestry_samples_df = pd.DataFrame(agroforestry_samples, columns=["sample_id"])
        agroforestry_samples = agroforestry_samples_df["sample_id"].tolist()

        tree_samples = list(set(tree_samples + agroforestry_samples))

    return tree_samples

def identifyMaizeSamples(activation,ref_id):
    activation_folder = os.path.join("/vitodata/worldcereal/data/COP4GEOGLAM/",activation)
    original_file = os.path.join(activation_folder,"refdata","harmonized")

    harm_file = os.path.join(original_file,f"{ref_id}.geoparquet")

    if os.path.exists(harm_file):
        harm_df = pd.read_parquet(harm_file)

        ewoc_maize = 1101060000
        maize_samples = harm_df[harm_df["ewoc_code"]==ewoc_maize]["sample_id"].unique()

        maize_samples_df = pd.DataFrame(maize_samples, columns=["sample_id"])

        maize_samples = maize_samples_df["sample_id"].tolist()

    return maize_samples

def ignoreSamples(tcv_folder,run_prefix,datafile,ignore_samples=[],overwrite=False,ignoreMaize=False):
    ignore_samples_csv = os.path.join(tcv_folder,f"{run_prefix}_ignore_sample_ids.csv")
    if not os.path.exists(ignore_samples_csv) or overwrite:
        #load existing train, test, and val sample_ids

        all_sample_ids = datafile["sample_id"].unique().tolist()
        matching_sample_ids = set()
        for ignore_sample in ignore_samples:
            for sample_id in all_sample_ids:
                if ignore_sample in sample_id:
                    matching_sample_ids.add(sample_id)

        matching_sample_ids = list(matching_sample_ids)

        trees_samples_ids = identifySamplesWithTrees(activation,"2025_MOZ_COPERNICUS4GEOGLAM_POINT_110_harmonized_with_EXP_POINTS_POLY")
        trees_samples_ids_ITC = identifySamplesWithTrees(activation,"2025_MOZ_ITC_POINT_110_harmonized")

        matching_sample_ids = set(matching_sample_ids + trees_samples_ids + trees_samples_ids_ITC)

        if ignoreMaize:
            maize_samples_ids = identifyMaizeSamples(activation,"2025_MOZ_COPERNICUS4GEOGLAM_POINT_110_harmonized_with_EXP_POINTS_POLY")
            matching_sample_ids = set(list(matching_sample_ids) + maize_samples_ids)


        #save the matching sample ids to a csv file
        pd.DataFrame(matching_sample_ids, columns=["sample_id"]).to_csv(os.path.join(tcv_folder,f"{ref_id}_ignore_sample_ids.csv"), index=False)
        ignore_ids = matching_sample_ids
    else:
        ignore_ids = pd.read_csv(ignore_samples_csv)["sample_id"].tolist()

    return ignore_ids

def makeSplit(activation,ref_id,test_size=0.15,cal_size=0.15,random_state=42,overwrite=False,ignore_samples = [],ignoreMaize=False,removePGP_percentage=0.4,
              addPGP_to_ignore=False,output_name=None):

    activation_folder = os.path.join("/vitodata/worldcereal/data/COP4GEOGLAM/",activation)
    trainingdata_folder = os.path.join(activation_folder,"trainingdata")

    tcv_folder = os.path.join(trainingdata_folder,"data_split")
    if not os.path.exists(tcv_folder):
        os.makedirs(tcv_folder)
        os.chmod(tcv_folder, 0o777)

    train_out = os.path.join(tcv_folder,f"{ref_id}_train.parquet")
    if output_name is not None:
        train_out = train_out.replace(ref_id,output_name)

    run_prefix = ref_id
    if output_name is not None:
        run_prefix = output_name

    if not os.path.exists(train_out) or overwrite:

        merged_gdf = gpd.read_parquet(os.path.join(trainingdata_folder,f"{ref_id}.parquet"))

        ITC_points = merged_gdf[merged_gdf["sample_id"].str.contains("ITC")]
        C4G_points = merged_gdf[~merged_gdf["sample_id"].str.contains("ITC")]

        ITC_points["ssu_id"] = ITC_points["sample_id"].str.split("_").str[5] + "_" + ITC_points["sample_id"].str.split("_").str[6]
        C4G_points["ssu_id"] = C4G_points["sample_id"].str.split("_").str[5] + "_" + C4G_points["sample_id"].str.split("_").str[6]

        merged_gdf = gpd.GeoDataFrame(pd.concat([ITC_points,C4G_points], ignore_index=True))
        all_samples = merged_gdf["sample_id"].unique().tolist()

        ITC_lookup_path = os.path.join(activation_folder,"refdata","harmonized","lookup","2025_MOZ_ITC_POINT_110_harmonized_lookup.parquet")
        ITC_lookup = pd.read_parquet(ITC_lookup_path)

        #add cropping pattern info to ITC_points from ITC_lookup based on sample_id
        ITC_points = ITC_points.merge(ITC_lookup[["sample_id","cropping_pattern",'trees_in_cropfield',"dominant_crop","dominant_percentage"]], on="sample_id", how="left")

        ITC_points = ITC_points[ITC_points["trees_in_cropfield"]!="trees_yes"]

        #only monocropping
        ITC_mono = ITC_points[ITC_points["cropping_pattern"] == "mono_culture"]
        merged_mono = gpd.GeoDataFrame(pd.concat([ITC_mono,C4G_points], ignore_index=True))

        if ignoreMaize:
            ITC_maize = ITC_points[ITC_points["dominant_crop"].str.contains("maize", case=False, na=False)]
            #select on dominant crop percentage being larger than 80%
            ITC_maize = ITC_maize[ITC_maize["dominant_percentage"] >= 80]
            #set ewoc_code to 1101060000 for these samples
            ITC_maize["ewoc_code"] = 1101060000
            ITC_maize["cropping_pattern"] = "mono_culture"
            #add these points to the merged_mono dataframe
            merged_mono = gpd.GeoDataFrame(pd.concat([merged_mono,ITC_maize], ignore_index=True))

        class_mappings_path = os.path.join(activation_folder, "class_mappings_mozambique.json")

        with open(class_mappings_path) as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                # Convert list of mappings to a dict
                class_mappings: Dict[str, Any] = {mapping["name"]: mapping["mapping"] for mapping in loaded}
            else:
                class_mappings = cast(Dict[str, Any], loaded)

        landcover_mapping = class_mappings["LANDCOVER10"]
        croptype_mapping = class_mappings["CROPTYPE_Mozambique"]

        merged_mono["ewoc_code"] = merged_mono["ewoc_code"].astype(str)
        merged_mono["landcover"] = merged_mono["ewoc_code"].map(landcover_mapping)
        merged_mono["croptype"] = merged_mono["ewoc_code"].map(croptype_mapping)

        ignore_samples = ignoreSamples(tcv_folder,run_prefix,merged_mono,ignore_samples=ignore_samples,overwrite=overwrite,ignoreMaize=ignoreMaize)
        #remove sample_id's that are in the ignore_samples list
        merged_mono = merged_mono[~merged_mono["sample_id"].isin(ignore_samples)]

        #make a stratified split of ssu_id's into train
        stratify_cols = ["landcover","croptype"]
        merged_mono_ssu = merged_mono[["ssu_id","ewoc_code"] + stratify_cols].drop_duplicates()

        #also remove the ones with missing values for landcover (= likely fallows)
        merged_mono_ssu = merged_mono_ssu[~merged_mono_ssu["landcover"].isna()]

        #add finetune_class, if croptype available, use that, if NA, use landcover
        merged_mono_ssu["finetune_class"] = merged_mono_ssu["croptype"]
        merged_mono_ssu["finetune_class"] = merged_mono_ssu["finetune_class"].fillna(merged_mono_ssu["landcover"])

        train_val_ssu, test_ssu = train_test_split(merged_mono_ssu[["ssu_id","ewoc_code","finetune_class"]], test_size=test_size, random_state=random_state, stratify=merged_mono_ssu["finetune_class"])
        train_val_ssu_ids = train_val_ssu["ssu_id"].unique()
        train_val_df = merged_mono_ssu[merged_mono_ssu["ssu_id"].isin(train_val_ssu_ids)]
        train_ssu, val_ssu = train_test_split(train_val_df[["ssu_id","ewoc_code","finetune_class"]], test_size=cal_size/(1-test_size), random_state=random_state, stratify=train_val_df["finetune_class"])

        if removePGP_percentage > 0:
            difficultPGP = identifyDifficultPGP(difficult_percentage=removePGP_percentage)
            dPGP_SSU = set(difficultPGP)

        merged_mono["finetune_class"] = merged_mono["croptype"]
        merged_mono["finetune_class"] = merged_mono["finetune_class"].fillna(merged_mono["landcover"])

        train_df = merged_mono[merged_mono["ssu_id"].isin(train_ssu["ssu_id"])].copy()
        val_df = merged_mono[merged_mono["ssu_id"].isin(val_ssu["ssu_id"])].copy()
        test_df = merged_mono[merged_mono["ssu_id"].isin(test_ssu["ssu_id"])].copy()

        # ---------------------------------------------------------------------
        # Move difficult PGP SSUs from train into val and test, equally split
        # ---------------------------------------------------------------------

        # Difficult SSUs that are actually present in the current training split
        if removePGP_percentage > 0:
            dpgp_in_train = np.array(
                train_df.loc[train_df["ssu_id"].isin(dPGP_SSU), "ssu_id"].drop_duplicates()
            )

            if len(dpgp_in_train) > 0:
                rng = np.random.default_rng(random_state)
                rng.shuffle(dpgp_in_train)

                # Split difficult SSUs approximately 50/50 between val and test
                half = len(dpgp_in_train) // 2

                dpgp_to_val = set(dpgp_in_train[:half])
                dpgp_to_test = set(dpgp_in_train[half:])

                # Select rows to move
                move_to_val_df = train_df[train_df["ssu_id"].isin(dpgp_to_val)].copy()
                move_to_test_df = train_df[train_df["ssu_id"].isin(dpgp_to_test)].copy()

                # Remove these SSUs from train
                train_df = train_df[
                    ~train_df["ssu_id"].isin(dpgp_to_val | dpgp_to_test)
                ].copy()

                if addPGP_to_ignore:
                    # Add the difficult PGP SSUs to the ignore list
                    new_ignore_samples = set(move_to_val_df["sample_id"].unique()) | set(move_to_test_df["sample_id"].unique())
                    ignore_samples = set(ignore_samples) | new_ignore_samples

                else:

                    # Add them to val and test
                    val_df = pd.concat([val_df, move_to_val_df], ignore_index=True)
                    test_df = pd.concat([test_df, move_to_test_df], ignore_index=True)

        train_sample_id = train_df["sample_id"].unique()
        val_sample_id = val_df["sample_id"].unique()
        test_sample_id = test_df["sample_id"].unique()

        removed_samples = set(all_samples) - set(train_sample_id) - set(val_sample_id) - set(test_sample_id)
        #write to ignore samples csv file
        ignore_samples_df = pd.DataFrame(list(removed_samples), columns=["sample_id"])
        ignore_samples_df.to_csv(os.path.join(tcv_folder,f"{run_prefix}_ignore_sample_ids.csv"), index=False)

        train_df.to_parquet(os.path.join(tcv_folder,f"{run_prefix}_train.parquet"), index=False)
        val_df.to_parquet(os.path.join(tcv_folder,f"{run_prefix}_val.parquet"), index=False)
        test_df.to_parquet(os.path.join(tcv_folder,f"{run_prefix}_test.parquet"), index=False)

        #save the sample ids in separate csv files
        pd.DataFrame(train_sample_id, columns=["sample_id"]).to_csv(os.path.join(tcv_folder,f"{run_prefix}_train_sample_ids.csv"), index=False)
        pd.DataFrame(val_sample_id, columns=["sample_id"]).to_csv(os.path.join(tcv_folder,f"{run_prefix}_val_sample_ids.csv"), index=False)
        pd.DataFrame(test_sample_id, columns=["sample_id"]).to_csv(os.path.join(tcv_folder,f"{run_prefix}_test_sample_ids.csv"), index=False)

        #create summary tables for train, val and test splits
        summary_train = createSummaryTable(train_df,group_cols=["landcover","croptype","source_file"],agg_cols=["sample_id","ssu_id"])
        summary_val = createSummaryTable(val_df,group_cols=["landcover","croptype","source_file"],agg_cols=["sample_id","ssu_id"])
        summary_test = createSummaryTable(test_df,group_cols=["landcover","croptype","source_file"],agg_cols=["sample_id","ssu_id"])

        summary_train.to_csv(os.path.join(tcv_folder,f"{run_prefix}_train_summary.txt"), index=False, sep="\t")
        summary_val.to_csv(os.path.join(tcv_folder,f"{run_prefix}_val_summary.txt"), index=False, sep="\t")
        summary_test.to_csv(os.path.join(tcv_folder,f"{run_prefix}_test_summary.txt"), index=False, sep="\t")

if __name__ == "__main__":

    activation = "mozambique_pm"
    ref_id = "2025_MOZ_COPERNICUS4GEOGLAM_ITC_POINT_EXP_POLY_MERGED"

    cal_size = 0.1
    test_size = 0.2

    ignore_samples = [
        "253659_35_EXP_16357",
        "253659_43_EXP_16381",
        "253659_51_EXP_16408",
        "253659_53",
        "261126_21",
        "336406_15_EXP_22368",
        "12959_32_EXP_3447",
        "237888_25_EXP_38014",
        "237888_33",
        "261126_32_EXP_63396",
        "324810_13_EXP_42487",
        "34547_15_EXP_23121",
        "34547_15_EXP_23128",
        "34547_15_EXP_23124",
        "34517_34_EXP_23198",
        "34517_34_EXP_23194",
        "34517_34_EXP_23199",
        "34517_34_EXP_23196",
        "34517_44",
        "34517_45",
        "361380_13_EXP_44846",
        "361380_13_EXP_44842",
        "361380_13_EXP_44847",
        "361380_13_EXP_44844",
        "361380_13_EXP_44848",
        "361380_14",
        "361380_15",
        "361380_21",
        "361380_41"
    ]

    overwrite = True
    ignoreMaize = True
    removePGP_percentage = 0.4
    addPGP_to_ignore = True
    output_name = ref_id + "_PGP_remove"

    makeSplit(activation,ref_id,test_size=test_size,cal_size=cal_size,
              overwrite=overwrite,ignore_samples = ignore_samples,ignoreMaize=ignoreMaize,
              removePGP_percentage=removePGP_percentage,addPGP_to_ignore=addPGP_to_ignore,
              output_name=output_name)
