"""
python scripts/create_cog.py \
  --raw-dir /vitodata/worldcereal/data/COP4GEOGLAM/moldova/production/V1_11092025/raw \
  --postprocess-dir /vitodata/worldcereal/data/COP4GEOGLAM/moldova/production/V1_11092025/postprocessed \
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from loguru import logger
from scipy.signal import convolve2d
from tqdm import tqdm


def spatial_smoothing(raw_results):
    class_probabilities = raw_results.astype("float32")

    conv_kernel = np.array([[1, 2, 1], [2, 3, 2], [1, 2, 1]], dtype=np.int16)

    for class_idx in range(class_probabilities.shape[0]):
        class_probabilities[class_idx] = (
            convolve2d(
                class_probabilities[class_idx],
                conv_kernel,
                mode="same",
                boundary="symm",
            )
            / conv_kernel.sum()
        )

    # Sum of probabilities should be 1
    class_probabilities = class_probabilities / class_probabilities.sum(axis=0)

    return class_probabilities


def process_tile(
    tile: Path,
    output_dir: Path,
):
    """
    Parameters
    ----------
    tile : Path
        Input raw tile path (expects at least 2 bands: class + prob in first two).
    output_dir : Path
        Directory to write postprocessed tile.
    """
    tile_id = tile.stem.split("_")[-1]
    logger.info(f"Postprocessing tile: {tile_id}")

    output_path = output_dir / tile.name.replace("-raw", "")
    if output_path.exists():
        logger.info(f"Tile {tile_id} already exists at {output_path}, skipping")
        return

    with rasterio.open(tile) as src:
        raw_results = src.read()[:, :2000, :2000]  # [bands, height, width]
        src_profile = src.profile

    # Reduce to probability only
    raw_results = raw_results[
        2:, :, :
    ]  # Assuming band 1 is class, 2 is winning prob, rest are class probs

    # Optional: spatial smoothing
    raw_results = spatial_smoothing(raw_results)

    # Open a new rasterio dataset based on the profile of the source
    src_profile.update(dtype="uint8", count=1, compress="deflate", nodata=255)

    # -----------------------------------------------------
    # Get the classification of croptype
    # clf_array = ...  # [classification]
    clf_array = np.argmax(raw_results, axis=0).astype("uint8")
    # -----------------------------------------------------

    # Get cropland mask: CHANGE PATH TO CHOSEN CROPLAND PRODUCT
    cropland_file = Path(
        "/vitodata/worldcereal/data/COP4GEOGLAM/moldova/production/V1_11092025/postprocessed/"
    ) / tile.name.replace(".tif", "_cropland.tif").replace("-raw", "")
    with rasterio.open(cropland_file) as src_cropland:
        cropland_mask = src_cropland.read(1)[:2000, :2000] == 1
        nodata_mask = src_cropland.read(1)[:2000, :2000] == 255

    clf_array[~cropland_mask] = 254  # No crop class
    clf_array[nodata_mask] = 255  # Set nodata to nodata

    with rasterio.open(output_path, "w", **src_profile) as dst:
        dst.write(clf_array, 1)
        dst.set_band_description(1, "Classification")

        metadata = {}
        metadata["nodata_value"] = "255"
        metadata["description"] = "Crop Classification"

        dst.update_tags(**metadata)

    logger.info(f"Tile {tile_id} done → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Postprocess tiles into masked COG")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=False,
        default=Path(
            "/vitodata/worldcereal/data/COP4GEOGLAM/moldova/production/V1_11092025/raw/"
        ),
        help="Directory with raw per-tile croptype GeoTIFFs",
    )
    parser.add_argument(
        "--postprocess-dir",
        type=Path,
        required=False,
        default=Path(
            "/vitodata/worldcereal/data/COP4GEOGLAM/moldova/production/V1_11092025/postprocessed"
        ),
        help="Directory to write postprocessed (masked) tiles",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker threads"
    )

    manual_args = [
        "--raw-dir",
        "/vitodata/worldcereal/data/COP4GEOGLAM/moldova/production/V3_13012026/raw",
        "--postprocess-dir",
        "/vitodata/worldcereal/data/COP4GEOGLAM/moldova/production/V3_13012026/postprocessed",
    ]
    # manual_args = None

    args = parser.parse_args(manual_args)

    raw_output_path: Path = args.raw_dir
    output_dir: Path = args.postprocess_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get raw tiles
    raw_tiles = list(raw_output_path.rglob("*croptype-raw_*.tif"))
    logger.info(f"Found {len(raw_tiles)} raw tiles")
    if not raw_tiles:
        logger.error("No raw tiles found; aborting")
        return

    completed = 0
    failed = 0
    workers = max(1, args.workers)

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for i, item in enumerate(raw_tiles):
                logger.debug(f"Queueing tile {i + 1}/{len(raw_tiles)}: {item.name}")
                futures.append(
                    executor.submit(
                        process_tile,
                        item,
                        output_dir,
                    )
                )
            for i, fut in enumerate(tqdm(futures, desc="Processing tiles")):
                try:
                    fut.result(timeout=600)  # 10 minute timeout per tile
                    completed += 1
                except Exception as e:
                    failed += 1
                    logger.error(
                        f"Error postprocessing tile {i + 1} ({raw_tiles[i].stem}): {e}"
                    )
    except Exception as e:
        logger.error(f"ThreadPoolExecutor failed: {e}")
        raise

    logger.info(f"Postprocessing complete: {completed} successful, {failed} failed.)")

    logger.success("All done!")


if __name__ == "__main__":
    main()
