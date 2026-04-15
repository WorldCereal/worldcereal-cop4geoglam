"""
Script used to preprocess the data for Moldova before uploading to the RDM.
@Copernicus4GEOGLAM
"""

# Loading the necessary libraries
import os
from datetime import datetime

import fiona
import geopandas as gpd
import pandas as pd

# Define the path to the directory containing the files
# Change this to the worldcereal path
path = "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique/refdata/original/"
crop_points_file = "moz_results_2025.gpkg"
dataset_name = "2025_moz_copernicus4geoglam1_point_110"

keep_columns = [
    "_id",
    "id_psu",
    "box_id",
    "id_ssu",
    "landuse",
    "croptype",
    "cropping_pattern",
    "date",
    "irrigation",
    "overview_photo_of_field",
    "detail_photo_of_field",
    "geometry",
]

# LU Dict is retrieved from Moldova feasibility study
lu_dict = {
    "1": "forest",
    "2": "natural_grassland",
    "3": "agriculture",
    "4": "bare",
    "5": "build_up",
    "6": "natural_shrubs",
    "7": "water",
    "8": "wetland",
    "9": "undecided",
}
lu_landuse_list = ["1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0"]

os.makedirs(path, exist_ok=True)


def open_geopackage(file_path):

    layers = fiona.listlayers(file_path)

    dict_out = {}

    for layer in layers:
        try:
            gdf = gpd.read_file(file_path, layer=layer)
            dict_out[layer] = gdf
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return dict_out


def mimic_RDM(dataset, dataset_name):
    dataset["sample_id"] = [dataset_name + "_" + str(row) for row in dataset.index]
    dataset["valid_time"] = [
        date.replace("-", "/") + " 00:00:00 (CET)"
        for date in dataset["date"].astype(str)
    ]

    return dataset


geopackage = os.path.join(path, crop_points_file)
geopackage_gdf = open_geopackage(geopackage)

activation = crop_points_file.replace(".gpkg", "")

croptype_points = geopackage_gdf[f"{activation}"]
lu_points = geopackage_gdf[f"{activation}_upfront_desk_classification"]

# Check for CRS inconsistency
activation_crs = croptype_points.crs
if lu_points.crs != activation_crs:
    print(f"CRS mismatch: {lu_points.crs} vs {activation_crs}")
    lu_points = lu_points.to_crs(activation_crs)

# join datasets on id_ssu
dataset = croptype_points.merge(
    lu_points[["id_ssu", "box_id", "lu_code"]], on=["id_ssu", "box_id"], how="left"
)

# for some points, there is no landuse value,
# use the lu_code and lu_dict to map out the values
dataset["lu_code"] = dataset["lu_code"].astype(int).astype(str)
dataset.loc[~dataset["landuse"].isin(lu_landuse_list), "landuse"] = dataset.loc[
    ~dataset["landuse"].isin(lu_landuse_list), "lu_code"
].map(lu_dict)

# Set date of desk research to earliest date
dataset["date"] = [date[0:10] for date in dataset["start"].astype(str)]
dates = [
    datetime.strptime(date, "%Y-%m-%d")
    for date in dataset.loc[dataset["date"] != "nan", "date"].unique()
]
dataset.loc[dataset["date"].isna(), "date"] = datetime.strftime(min(dates), "%Y-%m-%d")
dataset.loc[dataset["date"] == "nan", "date"] = datetime.strftime(
    min(dates), "%Y-%m-%d"
)

# Fill in NA croptype with Landuse?
dataset.loc[dataset["croptype"].isna(), "croptype"] = dataset.loc[
    dataset["croptype"].isna(), "landuse"
]

# First estimate on number of points
landuse_est = (
    dataset.groupby(["landuse"])["geometry"]
    .count()
    .reset_index()
    .rename(columns={"geometry": "count"})
    .sort_values(by="count", ascending=False)
)
croptype_est = (
    dataset.groupby(["croptype"])["geometry"]
    .count()
    .reset_index()
    .rename(columns={"geometry": "count"})
    .sort_values(by="count", ascending=False)
)

print(landuse_est.head())
print(croptype_est.head())

# Remove unnecessary columns
dataset = dataset.loc[:, keep_columns]

dataset = gpd.GeoDataFrame(dataset, crs=activation_crs, geometry="geometry")

# Convert to single point geometries
dataset = dataset.set_geometry(dataset.geometry.centroid)


# Drop false SSUs
remove_PSU = [269801]
dataset = dataset[~dataset["id_psu"].isin(remove_PSU)]

# Drop undecided landuse
dataset = dataset[~dataset["landuse"].isin(["undecided", "9.0"])]

# Order by ssu
dataset = dataset.sort_values(by=["id_ssu"]).reset_index(drop=True)

# Mimic RDM structure
dataset = mimic_RDM(dataset, dataset_name)

if not os.path.exists(
    os.path.join(
        path.replace("original", "harmonized"), f"ewoc_mapping_{activation}.csv"
    )
):
    croptype_count = dataset.groupby("croptype")["valid_time"].count()
    croptype_count = pd.DataFrame(croptype_count)
    croptype_count = croptype_count.rename(columns={"valid_time": "count"}).sort_values(
        by="count", ascending=False
    )
    # write it to csv
    croptype_count.to_csv(
        os.path.join(
            path.replace("original", "harmonized"), f"ewoc_mapping_{activation}.csv"
        )
    )
else:
    # open csv
    croptype_count = pd.read_csv(
        os.path.join(
            path.replace("original", "harmonized"), f"ewoc_mapping_{activation}.csv"
        )
    )
    # Sanity check for double labels per ewoc_code
    if croptype_count["ewoc_code"].nunique() != croptype_count["ewoc_name"].nunique():
        croptype_count_check = (
            croptype_count.groupby("ewoc_code")["ewoc_name"].nunique().reset_index()
        )
        croptype_count_check = croptype_count_check[
            croptype_count_check["ewoc_name"] > 1
        ]
        # add column with the unique labels
        croptype_count_check["ewoc_names"] = [
            ", ".join(
                croptype_count[croptype_count["ewoc_code"] == code][
                    "ewoc_name"
                ].unique()
            )
            for code in croptype_count_check["ewoc_code"]
        ]
        croptype_count_check.reset_index()

    croptype_count_check = (
        croptype_count.groupby("ewoc_code")["ewoc_name"].nunique().reset_index()
    )
    # add column with the unique labels
    croptype_count_check["ewoc_names"] = [
        ", ".join(
            croptype_count[croptype_count["ewoc_code"] == code]["ewoc_name"].unique()
        )
        for code in croptype_count_check["ewoc_code"]
    ]
    croptype_count_check.reset_index()

    # Provide estimate of number of points per ewoc_name
    dataset_RDM = dataset.merge(
        croptype_count[["ewoc_name", "ewoc_code", "croptype"]],
        on="croptype",
        how="left",
    )
    ewoc_est = (
        dataset_RDM.groupby(["ewoc_name"])["geometry"]
        .count()
        .reset_index()
        .rename(columns={"geometry": "count"})
        .sort_values(by="count", ascending=False)
    )
    dataset_RDM["ewoc_code"] = dataset_RDM["ewoc_code"].astype(int)

    print(ewoc_est.head())
    # Save dataset
    dataset_RDM.to_file(
        os.path.join(path.replace("original", "harmonized"), f"{activation}_RDM.gpkg"),
        driver="GPKG",
    )

    # save dataset as geoparquet
    dataset_RDM.to_parquet(
        os.path.join(
            path.replace("original", "harmonized"),
            "2025_moz_copernicus4geoglam1_point_110.parquet",
        )
    )
