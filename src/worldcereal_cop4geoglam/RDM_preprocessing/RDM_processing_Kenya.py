
import os

import fiona
import geopandas as gpd
import pandas as pd

main_folder = "/vitodata/FOODTURE/cop4geo/Kenya/"

def open_geopackage(file_path):

    layers = fiona.listlayers(file_path)

    dict_out = {}

    for layer in layers:
        try:
            gdf = gpd.read_file(file_path,layer=layer)
            dict_out[layer] = gdf
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return dict_out


def set_location(geopackage):
    polygon_layer = list(geopackage.keys())[0]
    field_layer = list(geopackage.keys())[2]

    #set field location as the point representation of the polygon centroid
    field_location = geopackage[field_layer]
    polygons = geopackage[polygon_layer]

    join_col_polygons = "polygon_idbysegment"
    join_col_points = "parcel_id"

    joined = polygons.merge(field_location[[join_col_points, "geometry"]],
                            left_on=join_col_polygons, right_on=join_col_points,
                            how='left')
    joined = joined.set_geometry("geometry_y")
    #drop redundant geometry column
    joined = joined.drop(columns=["geometry_x"])

    geopackage[polygon_layer] = joined

    return geopackage

def correct_mixed_systems(geopackage):
    polygon_layer = list(geopackage.keys())[0]
    polygons = geopackage[polygon_layer]

    #if there is a value for croptype_mixed, use this as landuse value instead.
    polygons.loc[polygons["croptype_mixed"].notna(),
                 "landuse"] = polygons.loc[polygons["croptype_mixed"].notna(),
                                           "croptype_mixed"]

    geopackage[polygon_layer] = polygons

    return geopackage

def setTiming(geopackage,year,season = "short"):
    polygon_layer = list(geopackage.keys())[0]
    polygons = geopackage[polygon_layer]

    if season == "short":
        mid_season = f"{year}-12-01"
    elif season == "long":
        mid_season = f"{year}-06-01"

    #check the harvest status:
    non_harvested = polygons["harvest_info"] == "no"

    survey_date = pd.to_datetime(polygons["date_survey"])

    #difference between survye date and mid season
    date_diff = (survey_date - pd.to_datetime(mid_season)).dt.days

    #remove fields where the survey date is more than 90 days before or after mid
    to_remove = non_harvested & ((date_diff < -90) | (date_diff > 90))
    polygons = polygons.loc[~to_remove,:].copy()

    #set valid time as mid season date
    polygons["valid_time"] = mid_season

    geopackage[polygon_layer] = polygons

    return geopackage

def remove_invalid_data(geopackage):
    polygon_layer = list(geopackage.keys())[0]
    polygons = geopackage[polygon_layer]

    #remove fields with z_valid_for_rs_use not set to true
    valid_fields = polygons["z_valid_for_rs_use"]
    polygons = polygons.loc[valid_fields,:].copy()

    return polygons

def process_gkpg(geopackage,year,season,file_name):
    geopackage = set_location(geopackage)
    geopackage = correct_mixed_systems(geopackage)
    geopackage = setTiming(geopackage,year,season=season)
    geopackage = remove_invalid_data(geopackage)

    keep_cols = ['polygon_idbysegment', 'date_survey', 'valid_time', 'landuse',
                 'croptype_mixed', 'harvest_info', 'z_valid_for_rs_use', 'water_supply',
                 'geometry']
    geopackage = geopackage.rename(columns={"geometry_y":"geometry"})
    geopackage = geopackage.loc[:, keep_cols]

    #set geometry column
    geopackage = gpd.GeoDataFrame(geopackage, crs="EPSG:4326", geometry="geometry")

    #save final shapefile
    if '.shp' not in file_name:
        file_name = file_name + ".shp"
    geopackage.to_file(os.path.join(main_folder, file_name))

#observations with bare soil are not removed
shortrain = open_geopackage(os.path.join(main_folder,
                                         "kenya_shortrain2022_polygons.gpkg"))
longrain = open_geopackage(os.path.join(main_folder,
                                        "kenya_longrain2023_polygons.gpkg"))

process_gkpg(longrain,2023,season="long",
             file_name = "kenya_longrain2023_processed.shp")
process_gkpg(shortrain,2022,season="short",
             file_name="kenya_shortrain2022_processed.shp")
