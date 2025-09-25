# WorldCereal COP4GEOGLAM Scripts

This repository contains scripts for training and running crop type and cropland classification models using Presto embeddings and CatBoost classifiers, as well as for orchestrating large-scale production inference jobs.

## Script Overview
### 0. Prerequisites

Before running the scripts, ensure you have the following:

- **Harmonized training points Parquet file:**
  A Parquet file containing harmonized training points with labeled crop types or land cover classes. Expected to have the path `/vitodata/worldcereal/data/COP4GEOGLAM/{country}/refdata/harmonized/{year}_{country-abbreviation}_COPERNICUS4GEOGLAM_POINT_110.geoparquet`.

- **PSU Parquet/GPKG File:**  
  A file containing the geometry of PSUs and a unique identifier for each unit. Expected to have the path `/vitodata/worldcereal/data/COP4GEOGLAM/{country}/refdata/{country-abbreviation}_PSU.parquet`

- **Production Grid for AOI (Area of Interest):**  
  A 20k x 20k grid covering the area of interest.

To make both compatible for running the openEO scripts, use the `convert_gdf_to_production_grid.py` script as follows:

```sh
python scripts/convert_gdf_to_production_grid.py \
  --input path/to/input.geoparquet \
  --output path/to/output_with_utm.parquet \
  --id-source unique_id_name
```

This conversion script will convert processing geometries to UTM and add the EPSG codes to the GeoDataFrame.

**Parameters:**
- `--input`: Path to the input GeoParquet or GeoPackage file.
- `--output`: Path to save the output production grid file with UTM projection. Should be be `/vitodata/worldcereal/data/COP4GEOGLAM/{country}/refdata/{country-abbreviation}_PSU_UTM.parquet`.
- `--id-source`: Name of the unique identifier column in the input file.

### 1. Optional augmentation of training data points
- Active learning to draw additional points near known points. Make sure the nearest known point has the same label as the newly drawn ones. Afterwards, use QGIS's `Join attributes by nearest` to borrow the original attributes. Finally, use the field calculator tool to update the `sample_id` of the newly drawn points to be based on the original `sample_id`, and extend it by: `_AL_{fid}`.

- Point expansion: ...

### 2. Training data extractions
This step creates the time‑series used to fine‑tune Presto (and later to derive embeddings for CatBoost). It wraps the generic WorldCereal extraction pipeline (`worldcereal-classification` repo) for a specific reference dataset (`ref_id`).

Core script(s):
- `scripts/extractions/extract_points.sh` (convenience wrapper you adapt per ref_id / country)
- Upstream extractor: `worldcereal-classification/scripts/extractions/extract.py`

Typical workflow:
1. Prepare a harmonized reference samples GeoParquet (`{REF_ID}.geoparquet`) containing at least geometry and label columns, plus an `extract` flag (>= given threshold triggers extraction) if you want selective extraction.
2. Create (or copy and modify) `extract_points.sh` and set:
   - `REF_ID` (e.g. `2025_MDA_COPERNICUS4GEOGLAM_POINT_110`)
   - `COUNTRY_DIR` root (e.g. `/vitodata/worldcereal/data/COP4GEOGLAM/moldova`)
3. Ensure output folder path matches: `${COUNTRY_DIR}/trainingdata/${REF_ID}` (script checks the last folder equals `ref_id`).
4. Run the shell script (optionally in a tmux/screen session) and monitor openEO job progress via backend UI or logs.

Example minimal direct invocation (equivalent to what the bash wrapper does):
```bash
python /home/kristofvt/git/worldcereal-classification/scripts/extractions/extract.py \
  POINT_WORLDCEREAL \
  /vitodata/worldcereal/data/COP4GEOGLAM/moldova/trainingdata/2025_MDA_COPERNICUS4GEOGLAM_POINT_110 \
  /vitodata/worldcereal/data/COP4GEOGLAM/moldova/refdata/harmonized/2025_MDA_COPERNICUS4GEOGLAM_POINT_110.geoparquet \
  --ref_id 2025_MDA_COPERNICUS4GEOGLAM_POINT_110 \
  --python_memory 3000m \
  --parallel_jobs 2 \
  --max_locations 250 \
  --restart_failed \
  --extract_value 0
```

Outputs:
- A job tracking CSV (created inside the extraction output folder) recording status per subset.
- One GeoParquet per job consolidated under the `ref_id` directory.
- A merged extractions GeoParquet for the entire ref_id.

### 3. Preprocessed inputs extractions for PSUs
This optional step generates standardized preprocessed input cubes (S1/S2 derived features, etc.) for every Primary Sampling Unit (PSU) tile. Running it allows to perform local model inference for quick iterations on model improvements.

Core script: `scripts/run_collect_inputs.py`

What it does:
1. Reads a production grid parquet (one row per tile) that already contains per‑tile UTM geometry and EPSG.
2. For each tile, builds an inputs process graph via `worldcereal.job.create_inputs_process_graph` (temporal + spatial subset, orbit state handling).
3. Submits openEO batch jobs with resource options (driver/executor/python memory) and up to N parallel jobs.
4. Writes per‑tile NetCDF (or similar) outputs into subfolders named after `tile_name` and stores job & result metadata JSON files.
5. Maintains a persistent `job_tracking.csv` enabling restarts and failure recovery.

Required production grid columns (validated):
- `tile_name`
- `geometry_utm_wkt` (polygon in target UTM)
- `epsg_utm` (integer EPSG code matching geometry)

Adjustable flexible parameters (top of the script):
- `output_folder`: Root destination (e.g. `/vitodata/worldcereal/data/COP4GEOGLAM/mozambique/PSU_preprocessed_inputs`).
- `parallel_jobs`: Max concurrent openEO jobs (e.g. 20).
- `randomize_production_grid`: Shuffle tiles to spread load/time windows.
- `s1_orbit_state`: Fix to `ASCENDING` or `DESCENDING`; `None` lets logic auto‑determine both/optimal.
- `start_date`, `end_date`: Temporal context of inputs (match model season/target year span).
- `production_grid`: Path to PSU parquet produced earlier (see Section 0 conversion / grid generation).
- `restart_failed`: If `True`, rows with status in `{error,start_failed}` are reset to `not_started` before resubmission.

How to run (default in-place settings):
```bash
python scripts/run_collect_inputs.py
```
Edit the flexible parameter block first; or externalize via environment variables + small wrapper if managing multiple countries.

Outputs per tile subfolder under `output_folder`:
- One (or more) downloaded asset file(s) renamed to include `_<tile_name>.nc`.
- `job_<jobid>.json` and `result_<jobid>.json` metadata for traceability.
- A global `job_tracking.csv` in the root tracking statuses (`not_started`, `running`, `finished`, `error`).

Tuning & best practices:
- Keep `start_date`/`end_date` aligned with the model training temporal extent to avoid distribution shift.
- Validate a random sample of produced inputs netCDFs (dimensions, variable presence) before large inference.

### 4. `scripts/finetune_presto.py`
**Purpose:**  
Fine-tunes a Presto model on country-specific crop type or land cover data.

**Main Steps:**
- Loads training/validation/test data from parquet files (see [`COUNTRY_PARQUET_FILES`](src/worldcereal_cop4geoglam/constants.py)).
- Selects a pretrained Presto model (see [`PRESTO_PRETRAINED_MODEL_PATH`](src/worldcereal_cop4geoglam/constants.py)).
- Sets up experiment parameters (augmentation, balancing, time-explicit labeling).
- Runs finetuning and saves the trained model and evaluation results.

**How to Run:**  
You can run with manual arguments or via command line.  
Example (manual args in script):
```sh
python scripts/finetune_presto.py \
    --experiment_tag new_val_ids \
    --timestep_freq month \
    --country moldova \
    --finetune_classes CROPTYPE_Moldova \
    --use_balancing \
    --val_samples_file /vitodata/worldcereal/data/COP4GEOGLAM/moldova/trainingdata/val_ids_moldova_qgis.csv
```
For running experiments, the manual arguments are mostly used example below 
```python
    manual_args = [
        "--experiment_tag",
        "new_val_ids", # change this to the name of your experiment
        "--timestep_freq",
        "month",
        "--country",
        "moldova", # country to processm also used for retrieving the correct class mapping files. currently supported moldova, kenya, mozambique
        "--augment", # if passed, it is automatically True
        "--finetune_classes",
        "CROPTYPE_Moldova", # name of the mapping to use. should match the one in class_mappings_<country>.json 
        "--use_balancing", # if passed, it is automatically True
        "--val_samples_file",
        "/vitodata/worldcereal/data/COP4GEOGLAM/moldova/trainingdata/val_ids_moldova_qgis.csv", # path to sample_ids used for testing. very useful to have a fix one to compare experiment results
    ]
```

---

### 5. `scripts/train_catboost.py`
**Purpose:**  
Trains a CatBoost classifier on embeddings generated by a fine-tuned Presto model.

**Main Steps:**
- Loads Presto model and data (embeddings or raw parquet files).
- Computes or loads embeddings for train/val/test splits.
- Maps classes and applies optional downstream class mapping.
- Trains CatBoost model with cross-validation and saves the model in CBM and ONNX formats.
- Evaluates and saves metrics and plots.

**How to Run:**  
You can use manual configuration (set `USE_MANUAL_CONFIG = True` in the script) or provide arguments via command line:
```sh
python scripts/train_catboost.py \
    --presto_model_path <path_to_finetuned_presto_model> \
    --data_dir <directory_with_parquet_or_embeddings> \
    --output_dir <output_directory> \
    --finetune_classes CROPTYPE_Moldova \
    --detector croptype \
    --country moldova \
    --balance False
```
The manual arguments are currently preferred to passing them via running from the terminal.
example of manual arguments for cropland: 
```python
    balance = ... # apply balancing to the classes 
    country = "moldova" # country (moldova, mozambique) 
    modelversion = "100-MDA" # model version. number-coutry abbreviation
    finetune_classes = "LANDCOVER10" # name of mapping to be used. should match the one reported in class_mappings_<country>.json
    detector = "cropland" # either cropland or croptype
    presto_model_name = "presto-prometheo-cop4geoglam-new_val_ids-month-LANDCOVER10-augment=False-balance=True-timeexplicit=False-run=202508271322" # just name of the model to be used (i.e. no need to provide extension nor whole path)
    downstream_classes = {
        "temporary_crops": "cropland",
        "temporary_grasses": "other",
        "permanent_crops": "cropland",
        "grasslands": "other",
        "wetlands": "other",
        "shrubland": "other",
        "trees": "other",
        "built_up": "other",
        "water": "other",
    } # dictionary mapping the LANDCOVER10 classes used to train presto to the binary classes 
``` 
Make sure to point to the right Presto model. the one used to generate the embeddings for training the cropland model should be Presto trained on `LANDCOVER10` classes.
The same manual arguments can be used for training the croptype model. In the latter case, there is no need of providing the `downstream_classes` as they correspond to the `finetune_classes`. 

---
### 6. Upload models in artifactory  

Once the presto and catboost models have been trained and a version is chosen, they need to be uploaded in artifactory at https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/Copernicus4Geoglam/.
Copy the URLs to the models and report them in the `constants.py` script in the `PRODUCTION_MODELS_URLS` dictionary. 

---

### 7. `scripts/run_inference_locally.py`
**Purpose:**
Run fast, iterative inference (cropland and croptype) on the preprocessed input tiles produced in Section 3, entirely locally (no new openEO jobs). Useful for: (a) validating new CatBoost / Presto model versions, (b) quick error analysis on a handful of PSUs, (c) comparing alternative downstream class mappings, (d) providing inspiration for active learning points.

**Prerequisites:**
- A folder of per‑tile preprocessed inputs (e.g. `/vitodata/worldcereal/data/COP4GEOGLAM/mozambique/PSU_preprocessed_inputs/<tile_name>/*_inputs_<tile_name>.nc`).
- Finetuned Presto model folder (export produced by `finetune_presto.py`).
- Trained CatBoost model artifacts (ONNX) produced by `train_catboost.py` (or URLs listed in `PRODUCTION_MODELS_URLS`).
- Matching class mapping JSON already available under `src/worldcereal_cop4geoglam/` (same key used during training).

**What the script typically does:**
1. Loads the Presto model to produce (or load cached) embeddings per tile/time step from the preprocessed inputs.
2. Aggregates / flattens embeddings to the temporal representation expected by the CatBoost classifier.
3. Applies CatBoost to generate per‑pixel probabilities / hard labels.
4. Writes NetCDF outputs of the products.

---

### 8. `scripts/run_production.py`
**Purpose:**  
Submits large-scale inference jobs for crop type and cropland prediction using trained models.

**Main Steps:**
- Loads a grid of tiles for the target country (check [`generate_country_grid.py`](scripts/generate_country_grid.py) to generate the country grid)
- Sets up feature and classifier parameters using URLs from [`PRODUCTION_MODELS_URLS`](src/worldcereal_cop4geoglam/constants.py).
- Creates and manages inference jobs using OpenEO/CDSE backend.
- Handles job tracking, retries, and output organization.

**How to Run:**  
Edit the flexible parameters at the top of the script (country, output folder, production grid, dates, etc.), then run:
```sh
python scripts/run_production.py
```
The script will handle job creation and submission automatically.

---

## Setting Inputs

- **Country:**  
  Set via `--country` argument or variable in scripts. Supported countries are listed in [`COUNTRY_PARQUET_FILES`](src/worldcereal_cop4geoglam/constants.py).

- **Data Files:**  
  Parquet files for training/validation/testing must be present as specified in [`COUNTRY_PARQUET_FILES`](src/worldcereal_cop4geoglam/constants.py).

- **Pretrained Models:**  
  URLs/paths for pretrained Presto models are set in [`PRESTO_PRETRAINED_MODEL_PATH`](src/worldcereal_cop4geoglam/constants.py).

- **Model URLs for Production:**  
  URLs for production models are set in [`PRODUCTION_MODELS_URLS`](src/worldcereal_cop4geoglam/constants.py).

- **Other Parameters:**  
  Most scripts accept command-line arguments for experiment tags, balancing, augmentation, etc. See each script's argument parser for details.

---
### Data Location and Description

The main data folders used in this project are:

- **refdata/**  
  Contains reference datasets used for validation and benchmarking. (eg PSUs or sampled blocks from country grid to test on bigger areas)

- **auxdata/**  
  Contains auxiliary datasets such as masks, grids (eg country grids used for production), or supporting geospatial layers.

- **trainingdata/**  
  Contains training, validation, and test data (typically parquet files) for model development. (eg validation ids and inputs to train Presto)


Make sure to use the same folder structure for each country

---

## Additional Notes

- All outputs (trained models, metrics, logs) are saved in the specified output directories.
- For debugging or custom runs, you can modify the manual argument lists or variable assignments at the top of each script.
---

