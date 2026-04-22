
import json
import os
from typing import Any, Dict, cast

import geopandas as gpd
import pandas as pd
from sklearn.model_selection import train_test_split


def createSummaryTable(df,group_cols,agg_cols):

    agg_dict = {col:"nunique" for col in agg_cols}
    df[agg_cols] = df[agg_cols].fillna("NA")
    df[group_cols] = df[group_cols].fillna("NA")
    summary_table = df.groupby(group_cols).agg(agg_dict).reset_index()

    return(summary_table)

def makeSplit(activation,ref_id,test_size=0.15,cal_size=0.15,random_state=42,overwrite=False):

    activation_folder = os.path.join("/vitodata/worldcereal/data/COP4GEOGLAM/",activation)
    trainingdata_folder = os.path.join(activation_folder,"trainingdata")

    tcv_folder = os.path.join(trainingdata_folder,"data_split")
    if not os.path.exists(tcv_folder):
        os.makedirs(tcv_folder)
        os.chmod(tcv_folder, 0o777)

    train_out = os.path.join(tcv_folder,f"{ref_id}_train.parquet")

    if not os.path.exists(train_out) or overwrite:

        merged_gdf = gpd.read_parquet(os.path.join(trainingdata_folder,f"{ref_id}.parquet"))

        ITC_points = merged_gdf[merged_gdf["sample_id"].str.contains("ITC")]
        C4G_points = merged_gdf[~merged_gdf["sample_id"].str.contains("ITC")]

        ITC_points["ssu_id"] = ITC_points["sample_id"].str.split("_").str[5] + "_" + ITC_points["sample_id"].str.split("_").str[6]
        C4G_points["ssu_id"] = C4G_points["sample_id"].str.split("_").str[5] + "_" + C4G_points["sample_id"].str.split("_").str[6]

        merged_gdf = gpd.GeoDataFrame(pd.concat([ITC_points,C4G_points], ignore_index=True))

        ITC_lookup_path = os.path.join(activation_folder,"refdata","harmonized","lookup","2025_MOZ_ITC_POINT_110_harmonized_lookup.parquet")
        ITC_lookup = pd.read_parquet(ITC_lookup_path)

        #add cropping pattern info to ITC_points from ITC_lookup based on sample_id
        ITC_points = ITC_points.merge(ITC_lookup[["sample_id","cropping_pattern",'trees_in_cropfield']], on="sample_id", how="left")

        ITC_points = ITC_points[ITC_points["trees_in_cropfield"]!="trees_yes"]

        #only monocropping
        ITC_mono = ITC_points[ITC_points["cropping_pattern"] == "mono_culture"]
        merged_mono = gpd.GeoDataFrame(pd.concat([ITC_mono,C4G_points], ignore_index=True))

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

        train_df = merged_mono[merged_mono["ssu_id"].isin(train_ssu["ssu_id"])]
        val_df = merged_mono[merged_mono["ssu_id"].isin(val_ssu["ssu_id"])]
        test_df = merged_mono[merged_mono["ssu_id"].isin(test_ssu["ssu_id"])]

        train_sample_id = train_df["sample_id"].unique()
        val_sample_id = val_df["sample_id"].unique()
        test_sample_id = test_df["sample_id"].unique()

        train_df.to_parquet(os.path.join(tcv_folder,f"{ref_id}_train.parquet"), index=False)
        val_df.to_parquet(os.path.join(tcv_folder,f"{ref_id}_val.parquet"), index=False)
        test_df.to_parquet(os.path.join(tcv_folder,f"{ref_id}_test.parquet"), index=False)

        #save the sample ids in separate csv files
        pd.DataFrame(train_sample_id, columns=["sample_id"]).to_csv(os.path.join(tcv_folder,f"{ref_id}_train_sample_ids.csv"), index=False)
        pd.DataFrame(val_sample_id, columns=["sample_id"]).to_csv(os.path.join(tcv_folder,f"{ref_id}_val_sample_ids.csv"), index=False)
        pd.DataFrame(test_sample_id, columns=["sample_id"]).to_csv(os.path.join(tcv_folder,f"{ref_id}_test_sample_ids.csv"), index=False)

        #create summary tables for train, val and test splits
        summary_train = createSummaryTable(train_df,group_cols=["landcover","croptype","source_file"],agg_cols=["sample_id","ssu_id"])
        summary_val = createSummaryTable(val_df,group_cols=["landcover","croptype","source_file"],agg_cols=["sample_id","ssu_id"])
        summary_test = createSummaryTable(test_df,group_cols=["landcover","croptype","source_file"],agg_cols=["sample_id","ssu_id"])

        summary_train.to_csv(os.path.join(tcv_folder,f"{ref_id}_train_summary.txt"), index=False, sep="\t")
        summary_val.to_csv(os.path.join(tcv_folder,f"{ref_id}_val_summary.txt"), index=False, sep="\t")
        summary_test.to_csv(os.path.join(tcv_folder,f"{ref_id}_test_summary.txt"), index=False, sep="\t")


if __name__ == "__main__":

    activation = "mozambique_pm"
    ref_id = "2025_MOZ_COPERNICUS4GEOGLAM_ITC_POINT_EXP_POLY_MERGED"

    cal_size = 0.15
    test_size = 0.15

    makeSplit(activation,ref_id,test_size=test_size,cal_size=cal_size)
