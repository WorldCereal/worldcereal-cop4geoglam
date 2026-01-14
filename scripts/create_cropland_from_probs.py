"""
python scripts/create_cog.py \
  --country \"moldova\" \
  --vector-file /vitodata/worldcereal/auxdata/Gaul/GAUL_2024/GAUL_2024_L0.gpkg \
  --raw-dir /vitodata/worldcereal/data/COP4GEOGLAM/moldova/production/V1_11092025/raw \
  --postprocess-dir /vitodata/worldcereal/data/COP4GEOGLAM/moldova/production/V1_11092025/postprocessed \
  --overwrite-final
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fiona
import numpy as np
import rasterio
from loguru import logger
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom
from scipy.signal import convolve2d
from shapely.geometry import mapping, shape
from shapely.ops import unary_union
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
    country_geom_geojson: dict | None,
    geom_crs: str | None,
    country_name: str,
):
    """
    Parameters
    ----------
    tile : Path
        Input raw tile path (expects at least 2 bands: class + prob in first two).
    output_dir : Path
        Directory to write postprocessed tile.
    country_geom_geojson : dict | None
        Geometry (GeoJSON-like) of the country in its source CRS.
    geom_crs : str | None
        CRS of the provided geometry (e.g. 'EPSG:4326').
    country_name : str
        Country name for logging.
    """
    tile_id = tile.stem.split("_")[-1]

    output_path = output_dir / tile.name.replace("croptype", "cropland")
    if output_path.exists():
        logger.info(f"Tile {tile_id} already exists at {output_path}")
        return

    logger.info(f"Postprocessing tile: {tile_id}")

    with rasterio.open(tile) as src:
        raw_results = src.read()
        src_profile = src.profile
        dst_crs = src.crs
        transform = src.transform
        height, width = src.height, src.width

    # # Get alternative map
    # other_file = Path(
    #     str(tile.parent).replace("v4", "v3")
    # ) / tile.name
    # with rasterio.open(other_file) as other_src:
    #     other_map = other_src.read()

    # # Combine
    # raw_results = np.stack([raw_results, other_map], axis=0).mean(axis=0)

    # Optional: spatial smoothing
    raw_results = spatial_smoothing(raw_results)

    # Open a new rasterio dataset based on the profile of the source
    src_profile.update(dtype="uint8", count=2, compress="deflate", nodata=255)

    # Get the classification and probability of cropland
    clf_cropland = np.argmax(raw_results, axis=0) == 5  # Class 5: cropland
    prob_cropland = (raw_results[5, ...] * 100).astype(
        np.uint8
    )  # Band 6: prob cropland
    clf_probs_array = np.stack([clf_cropland, prob_cropland], axis=0)

    # Get road mask
    road_file = list(
        (
            Path(
                "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique/auxdata/osm_roads_rasterized"
            )
        ).rglob(("*" + tile.name.split("_")[-1]))
    )[0]
    road_mask = rasterio.open(road_file).read(1) == 1
    clf_probs_array[0][road_mask] = 0  # Set roads to non-cropland
    clf_probs_array[1][road_mask] = 0  # Set road probabilities to 0

    # Create mask if geometry provided
    if country_geom_geojson is not None and geom_crs is not None:
        try:
            if dst_crs and geom_crs and str(dst_crs) != geom_crs:
                geom_in_tile_crs = transform_geom(
                    geom_crs, dst_crs.to_string(), country_geom_geojson
                )
            else:
                geom_in_tile_crs = country_geom_geojson

            inside_mask = geometry_mask(
                [geom_in_tile_crs],
                out_shape=(height, width),
                transform=transform,
                invert=True,  # True inside the geometry
            )
            if inside_mask.sum() == 0:
                logger.warning(
                    f"Tile {tile_id} has no overlap with country '{country_name}', writing full nodata"
                )
                clf_probs_array[0][:] = 255
                clf_probs_array[1][:] = 255
            else:
                outside_mask = ~inside_mask
                clf_probs_array[0][outside_mask] = 255
                clf_probs_array[1][outside_mask] = 255
        except Exception as e:
            logger.error(f"Failed to apply geometry mask on tile {tile_id}: {e}")
    else:
        logger.debug("No country geometry provided; skipping masking step")

    with rasterio.open(output_path, "w", **src_profile) as dst:
        dst.write(clf_probs_array[0], 1)
        dst.set_band_description(1, "Classification")

        dst.write(clf_probs_array[1], 2)
        dst.set_band_description(2, "Probability")

        metadata = {}
        metadata["nodata_value"] = "255"
        metadata["description"] = "Crop Classification with probabilities"

        dst.update_tags(**metadata)

    logger.info(f"Tile {tile_id} done → {output_path}")


def _find_vector_file(vector_dir: Path) -> Path | None:
    """Try to locate a vector file (gpkg or shp) inside a directory.
    Returns first match or None."""
    for pattern in ("*GAUL_2024_L0.gpkg", "*.shp"):
        files = list(vector_dir.glob(pattern))
        if files:
            return files[0]
    return None


def load_country_geometry(
    vector_path: Path, country_name: str, name_field: str = "gaul1_name"
) -> tuple[dict, str]:
    """Load (and union if needed) geometry for a country.

    Returns GeoJSON-like geometry dict and its CRS string.
    """
    country_name_lower = country_name.lower()
    with fiona.open(vector_path) as src:
        crs = src.crs_wkt or (src.crs.to_string() if src.crs else None)
        matched = []
        for feat in src:
            val = str(feat["properties"].get(name_field, "")).lower()
            if country_name_lower in val:
                matched.append(shape(feat["geometry"]))
        if not matched:
            raise ValueError(
                f"No features found for country '{country_name}' in {vector_path} field '{name_field}'"
            )
        geom = unary_union(matched)
        return mapping(geom), (crs or "EPSG:4326")


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
        "--vector-dir",
        type=Path,
        required=False,
        default=Path("/vitodata/worldcereal/auxdata/Gaul/GAUL_2024"),
        help="Directory containing GAUL vector file (.gpkg or .shp)",
    )
    parser.add_argument(
        "--vector-file",
        type=Path,
        required=False,
        help="Explicit path to GAUL vector file (overrides --vector-dir)",
    )
    parser.add_argument(
        "--country",
        type=str,
        required=False,
        default="moldova",
        help="Country name to mask (case-insensitive)",
    )
    parser.add_argument(
        "--country-field",
        type=str,
        required=False,
        default="ADM0_NAME",
        help="Attribute field in the vector layer containing the country name",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker threads"
    )
    parser.add_argument(
        "--overwrite-final",
        action="store_true",
        help="Overwrite existing final COG if present",
    )

    manual_args = [
        "--country",
        "Zambézia",
        "--vector-file",
        "/vitodata/worldcereal/auxdata/Gaul/GAUL_2024/GAUL_2024_L1.gpkg",
        "--raw-dir",
        "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique/production/v3_landcover/raw",
        "--postprocess-dir",
        "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique/production/v3_landcover/raw/cropland",
    ]
    # manual_args = None

    args = parser.parse_args(manual_args)

    raw_output_path: Path = args.raw_dir
    output_dir: Path = args.postprocess_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Vector file resolution
    vector_file: Path | None = args.vector_file
    if vector_file is None:
        vector_file = _find_vector_file(args.vector_dir)
        if vector_file is None:
            logger.warning(
                f"No vector file found in {args.vector_dir}; proceeding WITHOUT masking"
            )
    if vector_file:
        logger.info(f"Using vector file for masking: {vector_file}")
        try:
            country_geom_geojson, geom_crs = load_country_geometry(
                vector_file, args.country
            )
        except Exception as e:
            logger.error(
                f"Failed to load country geometry: {e}; continuing without mask"
            )
            country_geom_geojson, geom_crs = None, None
            raise
    else:
        country_geom_geojson, geom_crs = None, None

    # Get raw tiles
    raw_tiles = list(raw_output_path.rglob("*croptype_*.tif"))
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
                        country_geom_geojson,
                        geom_crs,
                        args.country,
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

    logger.info(
        f"Postprocessing complete: {completed} successful, {failed} failed (country mask: {args.country if country_geom_geojson else 'NONE'})"
    )

    logger.success("All done!")


if __name__ == "__main__":
    main()
