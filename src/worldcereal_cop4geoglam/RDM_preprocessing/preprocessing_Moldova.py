
'''
Script used to preprocess the data for Moldova before uploading to the RDM.
@Copernicus4GEOGLAM
'''

#Loading the necessary libraries
import os
import fiona

import geopandas as gpd

from datetime import datetime

# Define the path to the directory containing the files
path = "/data/users/Public/koendevos/Copernicus4GEOGLAM/Moldova/" #Change this to the worldcereal path
crop_points_file = "mda_results_2025.gpkg"

keep_columns = ["serial_id","_id","id_psu","box_id","id_ssu","landuse","croptype","cropping_pattern","date","irrigation","geometry"]

#LU Dict is retrieved from Moldova feasibility study
lu_dict = {
        "1": "forest",
        "2": "natural_grassland",
        "3": "agriculture",
        "4": "bare",
        "5": "build_up",
        "6": "natural_shrubs",
        "7": "water",
        "8": "wetland",
        "9": "undecided"
}
lu_landuse_list = ["1.0","2.0","3.0","4.0","5.0","6.0","7.0","8.0","9.0"]

os.makedirs(path, exist_ok=True)
os.chmod(path, 0o777)

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

#fill in NA landuse values
dataset['lu_code'] = dataset['lu_code'].fillna('9')
dataset['lu_code'] = dataset['lu_code'].astype(int).astype(str)
#replace lu_code with landuse name
dataset.loc[dataset["landuse"].isna(),'landuse'] = dataset.loc[dataset["landuse"].isna(),'lu_code'].replace(lu_dict) 
dataset.loc[dataset["landuse"].isin(lu_landuse_list), 'landuse'] = dataset.loc[dataset["landuse"].isin(lu_landuse_list), 'landuse'].astype(int).astype(str).replace(lu_dict)

#Remove duplicate geometry column
dataset["geometry"] = dataset["geometry_y"]

#Set date of desk research to earliest date 
dataset["date"] = [date[0:10] for date in dataset["start"].astype(str)]
dates = [datetime.strptime(date,"%Y-%m-%d") for date in dataset.loc[dataset["date"] != "nan","date"].unique()]
dataset.loc[dataset["date"].isna(),'date'] = datetime.strftime(min(dates),"%Y-%m-%d")
dataset.loc[dataset["date"] == "nan",'date'] = datetime.strftime(min(dates),"%Y-%m-%d")

#Fill in NA croptype with Landuse?
dataset.loc[dataset["croptype"].isna(),"croptype"] = dataset.loc[dataset["croptype"].isna(),"landuse"]

#First estimate on number of points
landuse_est = dataset.groupby(["landuse"])["geometry"].count().reset_index().rename(columns={"geometry":"count"}).sort_values(by="count", ascending=False)
croptype_est = dataset.groupby(["croptype"])["geometry"].count().reset_index().rename(columns={"geometry":"count"}).sort_values(by="count", ascending=False)

print(landuse_est.head())
print(croptype_est.head())

#Remove unnecessary columns
dataset = dataset.loc[:, keep_columns]

dataset = gpd.GeoDataFrame(dataset, crs=activation_crs, geometry="geometry")

#Convert to single point geometries
dataset = dataset.set_geometry(dataset.geometry.centroid)

#Save dataset
dataset.to_file(os.path.join(path, f"{activation}_RDM.gpkg"), driver="GPKG")