import json
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Optional

import geopandas as gpd
import openeo
import pandas as pd
import requests
from dateutil.parser import parse  # type: ignore[import-untyped]
from loguru import logger
from openeo import BatchJob
from openeo.extra.job_management import MultiBackendJobManager
from openeo_gfmap import BoundingBoxExtent, TemporalContext
from openeo_gfmap.backend import cdse_connection
from worldcereal.job import WorldCerealProductType, create_inference_process_graph
from worldcereal.parameters import (
    ClassifierParameters,
    CropLandParameters,
    CropTypeParameters,
    FeaturesParameters,
    PostprocessParameters,
    PrestoFeatureExtractor,
)

from worldcereal_cop4geoglam.constants import PRODUCTION_MODELS_URLS

ONNX_DEPS_URL = "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies/onnx_deps_python311.zip"
FEATURE_DEPS_URL = "https://s3.waw3-1.cloudferro.com/swift/v1/project_dependencies/torch_deps_python311.zip"
RETRYABLE_CODES = {500, 502, 504}
MAX_RETRIES = 5  # Maximum number of retries for HTTP errors


def is_retryable_http(exc):
    # adjust if CDSE raises a custom exception
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in RETRYABLE_CODES
    return False


class InferenceJobManager(MultiBackendJobManager):
    @classmethod
    def get_latest_used_image(cls, row: pd.Series, result_metadata: dict) -> str:
        """Get the latest used S1 or s2 image from the result metadata."""

        s2_images_dates = [
            parse(Path(x["title"].split(" ")[-1]).stem.split("_")[-1])
            for x in result_metadata["links"]
            if "title" in x and "Sentinel-2" in x["title"]
        ]

        s1_images_dates = [
            parse(Path(x["title"].split(" ")[-1]).stem.split("_")[4])
            for x in result_metadata["links"]
            if "title" in x and "Sentinel-1" in x["title"]
        ]

        latest_image = max(s1_images_dates + s2_images_dates).strftime("%Y-%m-%d")
        logger.info(
            f"Latest S1 or S2 image used for tile {row.tile_name}: {latest_image}"
        )

        return latest_image

    def on_job_done(self, job: BatchJob, row):
        logger.info(f"Job {job.job_id} completed")
        output_dir = generate_output_path_inference(self._root_dir, 0, row)

        # Get job results
        job_result = job.get_results()

        # Get metadata
        job_metadata = job.describe()
        result_metadata = job_result.get_metadata()
        job_metadata_path = output_dir / f"job_{job.job_id}.json"
        result_metadata_path = output_dir / f"result_{job.job_id}.json"

        # Get the latest used image from the result metadata
        latest_image = self.get_latest_used_image(row, result_metadata)

        # Get the products
        assets = job_result.get_assets()
        for asset in assets:
            asset_name = asset.name.split(".")[0].split("_")[0]
            asset_type = asset_name.split("-")[0]
            asset_type = getattr(WorldCerealProductType, asset_type.upper())
            filepath = asset.download(target=output_dir)

            # We want to add the tile name to the filename and replace the end date with the latest image date
            new_filename = (
                "_".join(filepath.stem.split("_")[:-1] + [latest_image, row.tile_name])
                + ".tif"
            )
            new_filepath = filepath.parent / new_filename
            shutil.move(filepath, new_filepath)

        with job_metadata_path.open("w", encoding="utf-8") as f:
            json.dump(job_metadata, f, ensure_ascii=False)
        with result_metadata_path.open("w", encoding="utf-8") as f:
            json.dump(result_metadata, f, ensure_ascii=False)

        # post_job_action(output_file)
        logger.success("Job completed")


def create_worldcereal_inferencejob(
    row: pd.Series,
    connection: openeo.Connection,
    provider,
    connection_provider,
    epsg: int = 4326,
    cropland_parameters=None,
    croptype_parameters=None,
    postprocess_parameters=None,
    s1_orbit_state: Optional[str] = None,
    target_epsg: Optional[int] = None,
):
    temporal_extent = TemporalContext(start_date=row.start_date, end_date=row.end_date)
    spatial_extent = BoundingBoxExtent(*row.geometry.bounds, epsg=epsg)

    inference_result = create_inference_process_graph(
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        product_type=WorldCerealProductType.CROPTYPE,
        cropland_parameters=cropland_parameters,
        croptype_parameters=croptype_parameters,
        postprocess_parameters=postprocess_parameters,
        s1_orbit_state=s1_orbit_state,
        target_epsg=target_epsg,
    )

    # Submit the job
    job_options = {
        "driver-memory": "4g",
        "executor-memory": "2g",
        "executor-memoryOverhead": "1g",
        "python-memory": "3g",
        "soft-errors": 0.1,
        "image-name": "python311",
        "max-executors": 10,
        "udf-dependency-archives": [
            f"{ONNX_DEPS_URL}#onnx_deps",
            f"{FEATURE_DEPS_URL}#feature_deps",
        ],
    }

    return inference_result.create_job(
        title=f"WorldCereal in-season inference for {row.tile_name}",
        job_options=job_options,
    )


def generate_output_path_inference(
    root_folder: Path,
    geometry_index: int,
    row: pd.Series,
    asset_id: Optional[str] = None,
) -> Path:
    """Method to generate the output path for inference jobs.

    Parameters
    ----------
    root_folder : Path
        root folder where the output parquet file will be saved
    geometry_index : int
        For point extractions, only one asset (a geoparquet file) is generated per job.
        Therefore geometry_index is always 0. It has to be included in the function signature
        to be compatible with the GFMapJobManager
    row : pd.Series
        the current job row from the GFMapJobManager
    asset_id : str, optional
        Needed for compatibility with GFMapJobManager but not used.

    Returns
    -------
    Path
        output path for the point extractions parquet file
    """

    tile_name = row.tile_name

    # Create the subfolder to store the output
    subfolder = root_folder / str(tile_name)
    subfolder.mkdir(parents=True, exist_ok=True)

    return subfolder


if __name__ == "__main__":
    # ------------------------
    # Flexible parameters
    country = "moldova"
    output_folder = Path(
        f"/vitodata/worldcereal/data/COP4GEOGLAM/{country}/production/PSU_test_with_cropland_mask/raw"
    )
    epsg = 32635
    parallel_jobs = 15
    randomize_production_grid = (
        True  # If True, it will randomly select tiles from the production grid
    )
    debug = False  # Triggers a selection of tiles
    s1_orbit_state = "DESCENDING"  # If None, it will be automatically determined but we want it fixed here.
    start_date = "2024-10-01"
    end_date = "2025-09-30"
    production_grid = f"/vitodata/worldcereal/data/COP4GEOGLAM/{country}/refdata/MDA_PSU_with_psu_name.parquet"
    restart_failed = True  # If True, it will restart failed jobs
    # ------------------------

    job_tracking_csv = output_folder / "job_tracking.csv"

    # Ensure the output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Create a job dataframe if it does not exist
    if job_tracking_csv.is_file():
        logger.info("Job tracking file already exists, skipping job creation.")
        job_df = pd.read_csv(job_tracking_csv)

        if restart_failed:
            logger.info("Resetting failed jobs.")
            job_df.loc[
                job_df["status"].isin(["error", "start_failed"]),
                "status",
            ] = "not_started"

            # Save new job tracking dataframe
            job_df.to_csv(job_tracking_csv, index=False)

    else:
        logger.info("Job tracking file does not exist, creating new jobs.")

        production_gdf = (
            gpd.read_parquet(production_grid)
            .to_crs(epsg=epsg)
            .rename(columns={"name": "tile_name"})
        )
        if debug:
            logger.info("Running in debug mode, selecting a subset of tiles.")
            # Select a subset of tiles for debugging
            # This is just an example selection, adjust as needed
            selection = ["E382N290", "E412N272", "E370N226", "E364N274", "E338N288"]
            production_gdf = production_gdf[production_gdf["tile_name"].isin(selection)]

        if randomize_production_grid:
            logger.info("Randomizing the production grid tiles.")
            production_gdf = production_gdf.sample(frac=1).reset_index(drop=True)

        job_df = production_gdf[["tile_name", "geometry"]].copy()
        job_df["start_date"] = start_date
        job_df["end_date"] = end_date

    # Set dedicated feature and classifier parameters
    feature_parameters_cropland = FeaturesParameters(
        rescale_s1=False,
        # presto model for cropland embeddings of the country
        presto_model_url=PRODUCTION_MODELS_URLS[country]["presto"]["cropland"],  # NOQA
        compile_presto=False,
    )
    classifier_parameters_cropland = ClassifierParameters(
        # CatBoost model for cropland classification of the country
        classifier_url=PRODUCTION_MODELS_URLS[country]["catboost"]["cropland"]  # NOQA
    )

    feature_parameters_croptype = FeaturesParameters(
        rescale_s1=False,
        # presto model for croptype embeddings of the country
        presto_model_url=PRODUCTION_MODELS_URLS[country]["presto"]["croptype"],  # NOQA
        compile_presto=False,
    )
    classifier_parameters_croptype = ClassifierParameters(
        # CatBoost model for croptype classification of the country
        classifier_url=PRODUCTION_MODELS_URLS[country]["catboost"]["croptype"]  # NOQA
    )

    cropland_parameters = CropLandParameters(
        feature_extractor=PrestoFeatureExtractor,
        feature_parameters=feature_parameters_cropland,
        classifier_parameters=classifier_parameters_cropland,
    )

    croptype_parameters = CropTypeParameters(
        feature_parameters=feature_parameters_croptype,
        classifier_parameters=classifier_parameters_croptype,
        # Save resources, no cropland mask needed for the production run
        mask_cropland=True,
        save_mask=True,
    )

    # No postprocessing for the production run as we do this afterwards
    postprocess_parameters = PostprocessParameters(enable=True,
                                                   method="majority_vote", # to address spure predictions
                                                   save_intermediate=True # saves not postprocessed
                                                   )
    # Retry loop starts here
    attempt = 0
    while True:
        try:
            # Setup connection + manager
            connection = cdse_connection()
            logger.info("Setting up the job manager.")
            manager = InferenceJobManager(root_dir=output_folder)
            manager.add_backend(
                "cdse", connection=connection, parallel_jobs=parallel_jobs
            )

            # Kick off all jobs
            manager.run_jobs(
                df=job_df,
                start_job=partial(
                    create_worldcereal_inferencejob,
                    epsg=epsg,
                    cropland_parameters=cropland_parameters,
                    croptype_parameters=croptype_parameters,
                    postprocess_parameters=postprocess_parameters,
                    s1_orbit_state=s1_orbit_state,
                    target_epsg=epsg,
                ),
                job_db=job_tracking_csv,
            )
            logger.info("All jobs submitted successfully.")
            break  # success: exit loop

        except Exception as exc:
            # Only retry on HTTPError with 500/502/504
            if is_retryable_http(exc) and attempt < MAX_RETRIES:
                attempt += 1
                delay = 2**attempt  # Exponential backoff
                logger.warning(
                    f"Retryable HTTP error (code {exc.response.status_code}) on attempt "  # type: ignore
                    f"{attempt}/{MAX_RETRIES}, retrying in {delay}s..."
                )
                time.sleep(delay)
                continue
            # Non-retryable or maxed-out
            logger.error(f"Failed to submit jobs: {exc}")
            raise  # re-raise the exception that caused failure to avoid infinite loop
