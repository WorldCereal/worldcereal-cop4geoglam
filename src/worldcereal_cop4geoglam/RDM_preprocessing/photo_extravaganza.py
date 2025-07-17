'''
This script makes a compilation of all the funny pictures (those that are no standard crop type classes)
for visual reference
'''

import os
import fiona
import geopandas as gpd
import paramiko

from tqdm import tqdm


path = "/data/users/Public/koendevos/Copernicus4GEOGLAM/Moldova/" #Change this to the worldcereal path
crop_points_file = "mda_results_2025.gpkg"
activation = crop_points_file.replace(".gpkg", "")

## Terrasphere host

hostname = "81.169.253.229"
port = 22
username = f"{activation[0:3]}_user"
password = f"pwd4G30Gl4m@{activation[0:3].upper()}"

# Create an SSH client
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Automatically add host keys

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

gpkg = open_geopackage(os.path.join(path,crop_points_file))
croptype_points = gpkg[f'{activation}']

#Identify the croptype points that are not in the standard classes
croptype_points.loc[:,"croptype"] = croptype_points["croptype"].astype(str)
no_dominant_photos_detail = []
no_dominant_photos_detail_label = []
mixed_dominant_detail = []
mixed_dominant_photos_label = []
no_dominant_photos_overview = []
mixed_dominant_photos_overview = []

for i, row in croptype_points.iterrows():
    croptype = row["croptype"]
    if "no_dominant" in croptype:
        no_dominant_photos_detail.append(row["detail_photo_of_field"])
        no_dominant_photos_overview.append(row["overview_photo_of_field"])
        no_dominant_photos_detail_label.append(row["croptype"])
    elif "yes" in croptype:
        mixed_dominant_detail.append(row["detail_photo_of_field"])
        mixed_dominant_photos_overview.append(row["overview_photo_of_field"])
        mixed_dominant_photos_label.append(row["croptype"])
    else:
        continue

#Make connection to the terrasphere server
client.connect(hostname, port, username, password)
sftp = client.open_sftp()

#Photos are located in server root/detail_pix
photos_path_detail = "/detail_pix/"
photos_path_overview = "/overview_pix/"
no_dominant_photos_detail = [photos_path_detail + photo for photo in no_dominant_photos_detail]
mixed_dominant_detail = [photos_path_detail + photo for photo in mixed_dominant_detail]
no_dominant_photos_overview = [photos_path_overview + photo for photo in no_dominant_photos_overview]
mixed_dominant_photos_overview = [photos_path_overview + photo for photo in mixed_dominant_photos_overview]


os.makedirs(os.path.join(path, "no_dominant_photos"), exist_ok=True)
os.chmod(os.path.join(path, "no_dominant_photos"), 0o777)
os.makedirs(os.path.join(path, "mixed_dominant_photos"), exist_ok=True)
os.chmod(os.path.join(path, "mixed_dominant_photos"), 0o777)
os.makedirs(os.path.join(path, "overview_pictures"), exist_ok=True)
os.chmod(os.path.join(path, "overview_pictures"), 0o777)

try:
    # Connect to the server
    client.connect(hostname, port, username, password)

    # Open an SFTP session
    sftp = client.open_sftp()

    for photo in tqdm(no_dominant_photos_detail, desc="Taking pictures- Say Cheese! 🧀", unit="photo"):
        i = no_dominant_photos_detail.index(photo)
        local_path = os.path.join(path, "no_dominant_photos", os.path.basename(photo)+f"_{no_dominant_photos_detail_label[i]}.jpg")
        if not os.path.exists(local_path):
            sftp.get(photo, local_path)

    for photo in tqdm(mixed_dominant_detail, desc="Taking pictures- Say Cheese! 🧀", unit="photo"):
        i = mixed_dominant_detail.index(photo)
        local_path = os.path.join(path, "mixed_dominant_photos", os.path.basename(photo)+f"_{mixed_dominant_photos_label[i]}.jpg")
        if not os.path.exists(local_path):
            sftp.get(photo, local_path)

    for photo in tqdm(no_dominant_photos_overview, desc="Taking overview pictures- Say Cheese! 🧀", unit="photo"):
        i = no_dominant_photos_overview.index(photo)
        local_path = os.path.join(path, "overview_pictures", os.path.basename(photo)+f"_{no_dominant_photos_detail_label[i]}.jpg")
        if not os.path.exists(local_path):
            sftp.get(photo, local_path)

    for photo in tqdm(mixed_dominant_photos_overview, desc="Taking overview pictures- Say Cheese! 🧀", unit="photo"):
        i = mixed_dominant_photos_overview.index(photo)
        local_path = os.path.join(path, "overview_pictures", os.path.basename(photo)+f"_{mixed_dominant_photos_label[i]}.jpg")
        if not os.path.exists(local_path):
            sftp.get(photo, local_path)

    sftp.close()
finally:
    client.close()
