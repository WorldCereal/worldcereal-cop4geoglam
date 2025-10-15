import glob
import os

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rio_cogeo import cog_profiles, cog_translate
from shapely.geometry import box
from tqdm import tqdm

if __name__ == "__main__":

    activation = "mozambique"
    production_name = "v3_landcover"

    main_folder = "/vitodata/worldcereal/data/COP4GEOGLAM/"

    #Loading the roads shapefile
    act_folder = os.path.join(main_folder, activation)
    aux_folder = os.path.join(act_folder,"auxdata")
    shapefile = os.path.join(aux_folder,"gis_osm_roads_free_1.shp")
    roads = gpd.read_file(shapefile)
    roads_folder = os.path.join(aux_folder,"osm_roads_rasterized")
    os.makedirs(roads_folder, exist_ok=True)

    #Rasters
    raster_folder = os.path.join(act_folder,"production",production_name,"raw","cropland")
    raster_files = glob.glob(os.path.join(raster_folder,"*.tif"))

    for raster_file in tqdm(raster_files, desc="Processing rasters"):
        raster_filename = os.path.basename(raster_file)
        roads_raster_file = os.path.join(roads_folder, raster_filename)
        if not os.path.exists(roads_raster_file):
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
            cog_path = roads_raster_file.replace(".tif", "_cog.tif")
            cog_translate(
                roads_raster_file,
                roads_raster_file,
                cog_profile,
                in_memory=True,
                quiet=True
            )
