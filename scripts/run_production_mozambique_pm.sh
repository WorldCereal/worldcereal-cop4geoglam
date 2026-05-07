#!/bin/bash
set -euo pipefail

export OPENEO_AUTH_METHOD="client_credentials"
export OPENEO_AUTH_CLIENT_ID_CDSE="openeo-worldcereal-service-account"
export OPENEO_AUTH_CLIENT_SECRET_CDSE="kBCYAeh50XZPWBZfCBwcr1jYyg93pYMg"
export OPENEO_AUTH_PROVIDER_ID_CDSE="CDSE" 

# Make sure your current work directory is the root of the worldcereal repository
# ("worldcereal-classification")
# The next line should NOT be touched.
PROCESS_CMD="scripts/inference/run_worldcereal_task_openeo.py"

# Set organization IDs for billing (one will be randomly selected per job to spread credit billing)
OPENEO_ORGANIZATION_IDS="10523 12968"


# Make sure you select the path to your WorldCereal Python environment
PYTHONPATH="/home/kristofvt/miniconda3/envs/worldcereal/bin/python"

# Path to seasonal model zip file
SEASONAL_MODEL_ZIP="https://s3.waw3-1.cloudferro.com/project_dependencies/worldcereal/cases/WorldCerealPresto-Mozambique-PerfectiveMaintenance-PGPremove40-month-augment=True-balance=True-timeexplicit=True-masking=enabled-ema0.4-clamp=0.01-2.0-run=202605051657.zip"

# Parameters for spatial extent
GRID_PATH="/vitodata/worldcereal/data/COP4GEOGLAM/mozambique/auxdata/zambezia_blocks_20k_utm.gpkg"

# Parameter specifying output folder
OUTPUT_FOLDER="/vitodata/worldcereal/data/COP4GEOGLAM/mozambique_pm/production"

## Alternatively, you can specify the season specifications directly as a JSON string,
## and provide an explicit start and end date of the processing window encompassing both seasons.
SEASON_SPECIFICATIONS='{"s1": ["2024-10-01", "2025-07-31"]}'
START_DATE="2024-09-01"
END_DATE="2025-08-31"

## Product to generate: cropland or croptype
PRODUCT="croptype"

## Note below we have set the following flags:
# --enable-cropland-head \  --> meaning that the model will produce a cropland map.
# --enable-croptype-head \  --> meaning that the model will produce a croptype map.
# --enforce-cropland-gate \  --> meaning that the croptype classification will be masked using the cropland product.
# --merge-classification-products \  --> this will merge the cropland and croptype classification products into a single output.
# --class-probabilities \  --> this will output class probabilities in addition to the final classification map.

##  Postprocessing options
## (note that we activate cropland post-processing by setting 
# --enable-cropland-postprocess and
# --enable-croptype-postprocess flags below.
POSTPROCESS_METHOD="smooth_probabilities" # options are "majority_vote" or "smooth_probabilities"
POSTPROCESS_KERNEL_SIZE=3 # only used if method is "majority_vote"

##  Additionally export embeddings and NDVI time series
##  Add the following flags to the function call below:
# --export-embeddings \
# --export-ndvi \
# --driver_memory "12g"

# note below we set restart_failed to True, meaning that failed jobs
# will be restarted if you run the script again.


# Run mapping
"${PYTHONPATH}" "${PROCESS_CMD}" \
--task "classification" \
--bbox 35.1433568 -18.9023309 39.1410669 -14.99999809 \
--start_date "${START_DATE}" \
--end_date "${END_DATE}" \
--season-specifications-json "${SEASON_SPECIFICATIONS}" \
--product "${PRODUCT}" \
--output_folder "${OUTPUT_FOLDER}" \
--restart_failed \
--parallel_jobs 10 \
--enable-cropland-postprocess \
--enable-croptype-postprocess \
--cropland-postprocess-method "${POSTPROCESS_METHOD}" \
--cropland-postprocess-kernel-size "${POSTPROCESS_KERNEL_SIZE}" \
--croptype-postprocess-method "${POSTPROCESS_METHOD}" \
--croptype-postprocess-kernel-size "${POSTPROCESS_KERNEL_SIZE}" \
--class-probabilities \
--enable-cropland-head \
--enable-croptype-head \
--enforce-cropland-gate \
--seasonal-model-zip "${SEASONAL_MODEL_ZIP}" \
--organization_id ${OPENEO_ORGANIZATION_IDS} \
