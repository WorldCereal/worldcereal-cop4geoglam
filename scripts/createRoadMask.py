import glob
import os

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rio_cogeo import cog_profiles, cog_translate
from shapely.geometry import box
from tqdm import tqdm


def createRoadsMask(raster_file,roads_file,roads_folder,data=None):

        raster_filename = os.path.basename(raster_file)
        roads_raster_file = os.path.join(roads_folder, raster_filename).replace("croptype","cropland")
        if raster_filename == "croptype_2024-10-01_2025-09-30_2025-09-28_MOZ_1522.tif":
            print("t")


        if not os.path.exists(roads_raster_file):

            if data is not None:
                roads = data
            else:
                roads = gpd.read_file(roads_file)

            with rasterio.open(raster_file) as src:
                raster_crs = src.crs
                bounds = src.bounds
                height = src.height
                width = src.width
                transform = src.transform

            #Reprojecting roads to raster CRS
            roads_proj = roads.to_crs(raster_crs)

            #clip the roads with the raster extent
            bbox = gpd.GeoDataFrame({'geometry': [box(*bounds)]}, crs=raster_crs)
            roads_clip = gpd.clip(roads_proj, bbox)

            #rasterize the clipped roads
            roads_clip["value"] = 1
            shapes = ((geom, 1) for geom in roads_clip.geometry)

            roads_raster = rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                all_touched=True,
                dtype='uint8'
            )

            #Save the rasterized roads
            raster_filename = os.path.basename(raster_file)
            roads_raster_file = os.path.join(roads_folder, raster_filename)

            with rasterio.open(
                roads_raster_file,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=roads_raster.dtype,
                crs=raster_crs,
                transform=transform,
            ) as dst:
                dst.write(roads_raster, 1)

            #Convert to COG
            cog_profile = cog_profiles.get("deflate")
            cog_translate(
                roads_raster_file,
                roads_raster_file,
                cog_profile,
                in_memory=True,
                quiet=True
            )

if __name__ == "__main__":

    activation = "mozambique"
    production_name = "v4_landcover"

    main_folder = "/vitodata/worldcereal/data/COP4GEOGLAM/"

    #Loading the roads shapefile
    act_folder = os.path.join(main_folder, activation)
    aux_folder = os.path.join(act_folder,"auxdata")
    shapefile_roads = os.path.join(aux_folder,"gis_osm_roads_free_1.shp")
    shapefile_buildings = os.path.join(aux_folder,"gis_osm_buildings_a_free_1.shp")
    roads_folder = os.path.join(aux_folder,"osm_roads_rasterized")
    buildings_folder = os.path.join(aux_folder,"osm_buildings_rasterized")
    os.makedirs(roads_folder, exist_ok=True)
    os.makedirs(buildings_folder, exist_ok=True)

    #Rasters
    raster_folder = os.path.join(act_folder,"production",production_name,"raw","*")
    raster_files = glob.glob(os.path.join(raster_folder,"croptype*.tif"))

    #buildings = gpd.read_file(shapefile_buildings)
    roads = gpd.read_file(shapefile_roads)

    for raster_file in tqdm(raster_files):
        createRoadsMask(raster_file,shapefile_roads,roads_folder,data=roads)
        #createRoadsMask(raster_file,shapefile_buildings,buildings_folder,data=buildings)
