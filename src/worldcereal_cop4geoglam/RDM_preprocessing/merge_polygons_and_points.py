#Script that merges the superresolution points with in-situ points

import os
from pathlib import Path

import geopandas as gpd
from polygon_to_point import polygon_to_point
from tqdm import tqdm


def merge_points_and_polygons(activation, original_point_file, polygon_to_point_file,overwrite=False):

    ## do not change this--
    base_dir = "/vitodata/worldcereal/data/COP4GEOGLAM/"

    ref_data_dir = os.path.join(base_dir,activation,"refdata")
    original_ref_data_dir = os.path.join(ref_data_dir,"original")
    harmonized_ref_data_dir = os.path.join(ref_data_dir,"harmonized")

    merged_out = os.path.join(harmonized_ref_data_dir,original_point_file.replace(".parquet","_merged.parquet"))

    if not os.path.exists(merged_out) or overwrite:

        #original points file
        original_points = gpd.read_parquet(os.path.join(harmonized_ref_data_dir,original_point_file))
        new_points = gpd.read_file(os.path.join(original_ref_data_dir,polygon_to_point_file))
        if not os.path.exists(os.path.join(original_ref_data_dir,polygon_to_point_file)):
            #if the polygon to point file does not exist, create it
            polygon_to_point(activation, output_file=Path(polygon_to_point_file).stem)

        #first, ensure that they are in the same CRS
        if original_points.crs != new_points.crs:
            new_points = new_points.to_crs(original_points.crs)

        bar = tqdm(total=len(original_points), desc="Comparing original points to new points")
        indices_to_drop = []
        merged_points = original_points.copy()
        points_to_add = []

        #loop over original points, check for SSU_ID
        for i,point_o in original_points.iterrows():
            bar.update(1)
            point_o_id = point_o['id_ssu']
            #check if point id is in new points
            if point_o_id in new_points['id_ssu'].values:
                points_new = new_points[new_points['id_ssu'] == point_o_id]
                for n, point_n in points_new.iterrows():
                    updated_point = point_o.copy()
                    updated_point['geometry'] = point_n['geometry']
                    #add the updated point to the original points dataframe
                    #add row to original points dataframe
                    points_to_add.append(updated_point)
                #remove the original point from the original points dataframe
                indices_to_drop.append(i)

        merged_points = merged_points.drop(indices_to_drop)
        #add the new points to the original points dataframe
        add_points = gpd.GeoDataFrame(points_to_add)
        add_points = add_points.drop_duplicates()

        merged_points = gpd.pd.concat([merged_points, add_points], ignore_index=True)

        #store the merged file in the harmonized folder
        merged_points.to_parquet(merged_out, index=False)

if __name__ == "__main__":
    ##Change this--
    activation = "mozambique"
    original_point_file = "2025_MOZ_COPERNICUS4GEOGLAM_POINT_110_harmonized_with_EXP_POINTS.parquet"
    polygon_to_point_file = "Polygon_to_points.gpkg"
    merge_points_and_polygons(activation, original_point_file, polygon_to_point_file)
