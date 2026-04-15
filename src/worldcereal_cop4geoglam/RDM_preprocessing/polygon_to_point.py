import os

import geopandas as gpd
from shapely.geometry import Point

##Change this--
activation = "mozambique"
field_polygons = "view_results_2025_polygons_v2_1.gpkg"
inward_buffer = 10 #in meters
point_distance = 10

## do not change this--
base_dir = "/vitodata/worldcereal/data/COP4GEOGLAM/"

ref_data_dir = os.path.join(base_dir,activation,"refdata")
original_ref_data_dir = os.path.join(ref_data_dir,"original")

#Load in the polygons file
polygons = gpd.read_file(os.path.join(original_ref_data_dir,field_polygons))
polygons_original = polygons.copy()

#apply the buffer
polygons["geometry"] = polygons["geometry"].buffer(-inward_buffer)

#remove any invalid and empty geometries
polygons = polygons[polygons.is_valid]
polygons = polygons[~polygons.is_empty]

#remove points
polygons = polygons[~(polygons.geometry.type == "Point")]

points = []
for idx, row in polygons.iterrows():
    polygon = row["geometry"]
    minx, miny, maxx, maxy = polygon.bounds
    x_coords = list(range(int(minx), int(maxx), point_distance))
    y_coords = list(range(int(miny), int(maxy), point_distance))
    point_count = 0
    for x in x_coords:
        for y in y_coords:
            point = Point(x, y)
            if polygon.contains(point):
                point_count += 1
                # Optionally, copy attributes from the polygon
                point_attrs = row.to_dict()
                point_attrs["geometry"] = point
                points.append(point_attrs)
    #if there are no points within the polygon, add a point at the centroid
    if point_count == 0:
        centroid = polygon.centroid
        #additional check to see if centroid is within the polygon, if not, skip adding a point for this polygon
        if not polygon.contains(centroid):
            continue
        point_attrs = row.to_dict()
        point_attrs["geometry"] = centroid
        points.append(point_attrs)

#which polygons were removed due to buffering?
removed_polygons = polygons_original[~polygons_original.index.isin(polygons.index)]
#create a centroid for them and add them to the points list
for idx, row in removed_polygons.iterrows():
    polygon = row["geometry"]
    centroid = polygon.centroid
    if not polygon.contains(centroid):
        continue
    point_attrs = row.to_dict()
    point_attrs["geometry"] = centroid
    points.append(point_attrs)

# Convert to GeoDataFrame
points_gdf = gpd.GeoDataFrame(points, geometry="geometry", crs=polygons.crs)
#save the points file
points_gdf.to_file(os.path.join(original_ref_data_dir,"Polygon_to_points.gpkg"), driver="GPKG")
