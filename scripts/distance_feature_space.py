import pandas as pd

df_loc = "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique_pm/trainingdata/combined_extractions_0701-1031_with-anomalies_WIDE.parquet"

df = pd.read_parquet(df_loc)
col_prefixes = ["OPTICAL","SAR","DEM","METEO"]
pc_cols = []
for col in df.columns:
    print(col)
    for prefix in col_prefixes:
        if col.startswith(prefix):
            pc_cols.append(col)

maize_points = df[df["ewoc_code"] == 1101060000]
maize_points[maize_points == 65535] = pd.NA

pigeon_pea_points = df[df["ewoc_code"] == 1105010060]
pigeon_pea_points[pigeon_pea_points == 65535] = pd.NA

maize_mean = maize_points[pc_cols].median()

#calculate the euclidean distance of each pigeon pea point to the maize mean point
def euclidean_distance(row, mean):
    return ((row - mean) ** 2).sum() ** 0.5

pigeon_pea_points["distance_to_maize"] = pigeon_pea_points[pc_cols].apply(lambda row: euclidean_distance(row, maize_mean), axis=1)
pigeon_pea_points["ssu_id"] = pigeon_pea_points["sample_id"].str.split("_").str[5] + "_" + pigeon_pea_points["sample_id"].str.split("_").str[6]
#group_by ssu_id
pigeon_pea_ssu = pigeon_pea_points.groupby("ssu_id")["distance_to_maize"].mean().reset_index()

outfile = "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique_pm/feature_distance/pigeon_pea_distance_to_maize.parquet"
pigeon_pea_ssu.to_parquet(outfile)
