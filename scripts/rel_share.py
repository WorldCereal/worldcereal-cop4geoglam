
import os

import numpy as np
import rasterio

folder_v4 = "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique/production/v4_landcover/raw"

v4_or = os.path.join(folder_v4,"Copernicus4GEOGLAM_Zambézia_CropMask_2025.tif")
with rasterio.open(v4_or) as src:
    profile = src.profile
    data = src.read(1)

#count unique values
unique, counts = np.unique(data, return_counts=True)
