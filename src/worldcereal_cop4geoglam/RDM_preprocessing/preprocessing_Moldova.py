
'''
Script used to preprocess the data for Moldova before uploading to the RDM.
@Copernicus4GEOGLAM
'''

#Loading the necessary libraries
import os
import fiona

import geopandas as gpd

# Define the path to the directory containing the files
path = "/data/users/Public/koendevos/Copernicus4GEOGLAM/Moldova/"
crop_points_file = "mda_results_2025.gpkg"

os.makedirs(path, exist_ok=True)
os.chmod(path, 0o777)

#Functions
#Open Geopackage

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
    
geopackage = os.path.join(path, crop_points_file)
geopackage_gdf = open_geopackage(geopackage)

activation = crop_points_file.replace('.gpkg',"")

croptype_points = geopackage_gdf[f'{activation}']
lu_points = geopackage_gdf[f'{activation}_desk_lu_classification']

#Check for CRS inconsistency
activation_crs = croptype_points.crs
if lu_points.crs != activation_crs:
    print(f"CRS mismatch: {lu_points.crs} vs {activation_crs}")
    lu_points = lu_points.to_crs(activation_crs)

#join datasets
dataset = croptype_points.merge(lu_points, on = ["id_psu","box_id","id_ssu","lu_code"], how='right')

print("t")
