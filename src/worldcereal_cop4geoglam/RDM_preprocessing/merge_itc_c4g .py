#script that merges updated COP4GEOGLAM data with new ITC data

import os

import geopandas as gpd
import pandas as pd
from harmonize_ITC import harmonize_ITC
from merge_polygons_and_points import merge_points_and_polygons


def merge_cop4geoglam_with_itc(activation,
                               file_in_cop4geoglam = "2025_MOZ_COPERNICUS4GEOGLAM_POINT_110_harmonized_with_EXP_POINTS_merged.parquet",
                               file_in_itc = "2025_MOZ_ITC_POINT_110_harmonized.parquet",
                               file_out = "2025_MOZ_COPERNICUS4GEOGLAM_ITC_POINT_110_harmonized_with_EXP_POINTS_merged.parquet",
                               overwrite=False):

    ## do not change this--
    base_dir = "/vitodata/worldcereal/data/COP4GEOGLAM/"

    ref_data_dir = os.path.join(base_dir,activation,"refdata")
    harmonized_ref_data_dir = os.path.join(ref_data_dir,"harmonized")

    outfile = os.path.join(harmonized_ref_data_dir,file_out)

    if not os.path.exists(outfile) or overwrite:

        cop4geoglam_data =os.path.join(harmonized_ref_data_dir,file_in_cop4geoglam)
        if not os.path.exists(cop4geoglam_data):
            merge_points_and_polygons(activation,
                                    original_point_file=file_in_cop4geoglam.replace("_merged.parquet",".parquet"),
                                    polygon_to_point_file="Polygon_to_points.gpkg",
                                    overwrite=overwrite)
        cop4geoglam_data = gpd.read_parquet(os.path.join(harmonized_ref_data_dir,file_in_cop4geoglam))

        #add columns on dominant crop and percentage
        cop4geoglam_data["dominant_crop"] = None
        cop4geoglam_data["dominant_percentage"] = None

        for i, row in cop4geoglam_data.iterrows():
            if row["croptype"] is not None:
                if "yes" in row["croptype"]:
                    crops = row["croptype"].split(" ")
                    dominant_crops = crops[0].replace("yes_","")
                    cop4geoglam_data.at[i,"dominant_crop"] = dominant_crops
        cop4geoglam_data["id_psu"] = cop4geoglam_data["id_psu"].astype(str)

        itc_data = os.path.join(harmonized_ref_data_dir,file_in_itc)
        if not os.path.exists(itc_data):
            harmonize_ITC(itc_file_name = "itc_fc_crop_point_selection_checked.gpkg",
                          output_name =  file_in_itc,
                          activation=activation)
        itc_data = gpd.read_parquet(os.path.join(harmonized_ref_data_dir,file_in_itc))
        itc_data["id_psu"] = itc_data["id_psu"].astype(str)

        #ensure same CRS
        if cop4geoglam_data.crs != itc_data.crs:
            itc_data = itc_data.to_crs(cop4geoglam_data.crs)

        #check whether all columns are compatible in terms of data types, if not, adjust
        for column in cop4geoglam_data.columns:
            if column in itc_data.columns:
                if cop4geoglam_data[column].dtype != itc_data[column].dtype:
                    itc_data[column] = itc_data[column].astype(cop4geoglam_data[column].dtype)

        #remove box_id from both
        cop4geoglam_data = cop4geoglam_data.drop(columns=["box_id"], errors="ignore")
        itc_data = itc_data.drop(columns=["box_id"], errors="ignore")

        merged_data = gpd.GeoDataFrame(pd.concat([cop4geoglam_data, itc_data], ignore_index=True), crs=cop4geoglam_data.crs)

        merged_data.to_parquet(outfile)

if __name__ == "__main__":

    activation = "mozambique"
    file_in_cop4geoglam = "2025_MOZ_COPERNICUS4GEOGLAM_POINT_110_harmonized_with_EXP_POINTS_merged.parquet"
    file_in_itc = "2025_MOZ_ITC_POINT_110_harmonized.parquet"
    file_out = "2025_MOZ_COPERNICUS4GEOGLAM_ITC_POINT_110_harmonized_with_EXP_POINTS_merged.parquet"
    merge_cop4geoglam_with_itc(activation,
                               file_in_cop4geoglam,
                               file_in_itc,
                               file_out,
                               overwrite=True)
