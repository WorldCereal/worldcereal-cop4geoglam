#!/bin/bash
set -euo pipefail

# export OPENEO_AUTH_METHOD="client_credentials"
export OPENEO_AUTH_METHOD=""

# DIRECTORIES AND REF_ID
REF_ID="2025_MDA_COPERNICUS4GEOGLAM_POINT_110"
COUNTRY_DIR="/vitodata/worldcereal/data/COP4GEOGLAM/moldova"

# Python and extraction command
PYTHONPATH="/home/kristofvt//miniconda3/envs/worldcereal/bin/python"
EXTRACT_CMD="/home/kristofvt/git/worldcereal-classification/scripts/extractions/extract.py"

# Extraction parameters
PYTHON_MEMORY="3000m"
PARALLEL_JOBS="2"
MAX_LOCATIONS="250"
# ORGANIZATION_ID="10523"  # jrc-inseason-poc

# Build paths
REFDATA_FILE="${COUNTRY_DIR}/refdata/harmonized/${REF_ID}.geoparquet"
OUTDIR="${COUNTRY_DIR}/trainingdata/${REF_ID}/"

echo
printf '%0.s-' {1..50}
echo
echo "Launching extraction for REF_ID: ${REF_ID}"
echo "  Reference file : ${REFDATA_FILE}"
echo "  Output folder  : ${OUTDIR}"
echo
printf '%0.s-' {1..50}
echo

# Run extraction
"${PYTHONPATH}" "${EXTRACT_CMD}" POINT_WORLDCEREAL \
"${OUTDIR}" "${REFDATA_FILE}" \
--ref_id "${REF_ID}" \
--python_memory "${PYTHON_MEMORY}" \
--parallel_jobs "${PARALLEL_JOBS}" \
--max_locations "${MAX_LOCATIONS}" \
--extract_value 0 \
--restart_failed \
# --organization_id "${ORGANIZATION_ID}"
# --image_name "registry.stag.waw3-1.openeo-int.v1.dataspace.copernicus.eu/dev/openeo-geotrellis-kube:20250325-2415"

echo
printf '%0.s-' {1..50}
echo "Finished extraction for ${REF_ID}"
printf '%0.s-' {1..50}
echo

