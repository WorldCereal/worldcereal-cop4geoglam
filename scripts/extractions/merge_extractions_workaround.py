import glob
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Union

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from worldcereal.extract.utils import pipeline_log

WORLDCEREAL_BEGIN_DATE = datetime(2017, 1, 1)

DEFAULT_JOB_OPTIONS_POINT_WORLDCEREAL = {
    "driver-memory": "2G",
    "driver-memoryOverhead": "2G",
    "driver-cores": "1",
    "executor-memory": "1800m",
    "python-memory": "3000m",
    "executor-cores": "1",
    "max-executors": 22,
    "soft-errors": 0.1,
}

REQUIRED_ATTRIBUTES = {
    "feature_index": np.int64,
    "sample_id": str,
    "timestamp": "datetime64[ns]",
    "S2-L2A-B02": np.uint16,
    "S2-L2A-B03": np.uint16,
    "S2-L2A-B04": np.uint16,
    "S2-L2A-B05": np.uint16,
    "S2-L2A-B06": np.uint16,
    "S2-L2A-B07": np.uint16,
    "S2-L2A-B08": np.uint16,
    "S2-L2A-B8A": np.uint16,
    "S2-L2A-B11": np.uint16,
    "S2-L2A-B12": np.uint16,
    "S1-SIGMA0-VH": np.uint16,
    "S1-SIGMA0-VV": np.uint16,
    "slope": np.uint16,
    "elevation": np.uint16,
    "AGERA5-PRECIP": np.uint16,
    "AGERA5-TMEAN": np.uint16,
    "lon": np.float64,
    "lat": np.float64,
    "geometry": "geometry",
    "tile": str,
    "h3_l3_cell": str,
    "start_date": str,
    "end_date": str,
    "year": np.int64,
    "valid_time": str,
    "ewoc_code": np.int64,
    "irrigation_status": np.int64,
    "quality_score_lc": np.int64,
    "quality_score_ct": np.int64,
    "extract": np.int64,
}

def merge_output_files_point_worldcereal(
    output_folder: Union[str, Path],
    ref_id: str,
) -> None:
    """Merge the output geoparquet files of the point extractions. Partitioned per ref_id

    Parameters
    ----------
    output_folder : Union[str, Path]
        Location where extractions are saved
    ref_id : str
    collection id of the samples

    Raises
    ------
    FileNotFoundError
        If no geoparquet files are found in the output_folder
    """
    output_folder = Path(output_folder)
    merged_path = output_folder.parent / "worldcereal_merged_extractions.parquet"

    # Locate the files to merge and check whether there are any
    filecheck = list(output_folder.glob("**/*.geoparquet"))
    if len(filecheck) == 0:
        raise FileNotFoundError(f"No geoparquet files found in {output_folder}")
    else:
        pipeline_log.info(f"Merging {len(filecheck)} geoparquet files...")
    files_to_merge = str(output_folder / "**" / "*.geoparquet")

    # DuckDB requires the parent directory to exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # Check if this particular partition is already present in the merged path,
    # and if yes, delete it
    dir_name = merged_path / f"ref_id={ref_id}"
    if dir_name.exists():

        def _on_rm_error(func, path, exc_info):
            # Ensure write permissions and retry once
            try:
                os.chmod(path, 0o755)
                func(path)
            except Exception:
                raise

        # Retry deletion to handle transient "directory not empty" cases
        for _ in range(3):
            try:
                shutil.rmtree(str(dir_name), onerror=_on_rm_error)
                break
            except OSError:
                time.sleep(0.2)

    # Merge the files
    con = duckdb.connect()
    con.execute("INSTALL spatial;")
    con.execute("LOAD spatial;")

    try:

        con.execute(
            f"""
        COPY (
            SELECT * FROM read_parquet('{files_to_merge}', filename=false)
        ) TO '{str(merged_path)}' (FORMAT 'parquet', PARTITION_BY ref_id, OVERWRITE_OR_IGNORE, FILENAME_PATTERN '{ref_id}_{{i}}')
    """
        )

    except Exception:

        allfiles = glob.glob(files_to_merge, recursive=True)

        #load all files and check for files that don't have the required attributes
        for file in allfiles:
            gdf = gpd.read_parquet(file)
            missing_attributes = [attr for attr in REQUIRED_ATTRIBUTES if attr not in gdf.columns]
            if len(missing_attributes) > 0:
                pipeline_log.warning(f"File {file} is missing required attributes: {missing_attributes}")
                if 'timestamp' in missing_attributes:
                    gdf["timestamp"] = pd.to_datetime(gdf["date"], errors="coerce").dt.tz_localize(None)
                    gdf["timestamp"] = gdf["timestamp"].astype("datetime64[ns]")
                    #drop date
                    gdf = gdf.drop(columns=["date"])
                #save again with the required attributes
                gdf.to_parquet(file, index=False)


        #try merging again
        try:
            con.execute(
                f"""
            COPY (
                SELECT * FROM read_parquet('{files_to_merge}', filename=false)
            ) TO '{str(merged_path)}' (FORMAT 'parquet', PARTITION_BY ref_id, OVERWRITE_OR_IGNORE, FILENAME_PATTERN '{ref_id}_{{i}}')
        """
            )
        except Exception as e:
            pipeline_log.error(f"Failed to merge files after fixing missing date/timestamp attributes: {e}")
            raise



    con.close()


if __name__ == "__main__":

    ref_id = "2025_MOZ_COPERNICUS4GEOGLAM_ITC_POINT_110_harmonized_with_EXP_POINTS_merged"
    output_folder = os.path.join("/vitodata/worldcereal/data/COP4GEOGLAM/mozambique/trainingdata/", ref_id)
    merge_output_files_point_worldcereal(output_folder, ref_id)
