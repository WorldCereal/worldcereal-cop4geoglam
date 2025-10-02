
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm import tqdm

folder = "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique/"

photo_folder = "/data/sigma/cop4geo/moz/"

detail_folders = ["all_crops_photos","mixed_dominant_photos","no_dominant_photos"]

validation_ids = os.path.join(folder,"models","catboost","v1","croptype",
                              "Presto_run=202510011045_DownstreamCatBoost_croptype_v1_balance=True",
                              "val_embeddings.parquet")
validation_file = pd.read_parquet(validation_ids)

validation_file["id_ssu"] = [sample_id.split("_")[5]+"_"+ sample_id.split("_")[6] for
                             sample_id in validation_file["sample_id"]]

unique_ids = sorted(validation_file["id_ssu"].unique())

yn_photos = []

for id_ssu in tqdm(unique_ids, desc="Processing IDs"):
    print(id_ssu)
    photo = glob.glob(os.path.join(photo_folder,"**",f"*{id_ssu}*.jpg"),recursive=True)
    photo = [photo for photo in photo if
             any(detail in photo for detail in detail_folders)]
    id_label = validation_file[validation_file["id_ssu"]==id_ssu].reset_index()

    if len(photo)>0:
        #if there are multiple photos, create a loop to display them all
        for phot in photo:
            img = Image.open(phot)
            # Display the image using matplotlib
            plt.imshow(img)
            plt.axis("off")  # Turn off axes for better visualization
            plt.title(f"{id_label['finetune_class'][0]}?")
            plt.show()

            # Ask user for input
            yn = input(f"Is this a {id_label['finetune_class'][0]} system? y/n: ")
            yn_photos.append((id_ssu,yn))
            if yn.lower() == 'y':
                continue
            elif yn.lower() == 'n':
                continue
            else:
                print("Invalid input, please enter 'y' or 'n'.")


validation_file["user_validation"] = validation_file["id_ssu"].map(dict(yn_photos))






print(validation_file.head(5))
