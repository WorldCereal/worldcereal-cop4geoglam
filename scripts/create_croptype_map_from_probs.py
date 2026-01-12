"""
python scripts/create_cog.py \
  --country \"moldova\" \
  --vector-file /vitodata/worldcereal/auxdata/Gaul/GAUL_2024/GAUL_2024_L0.gpkg \
  --raw-dir /vitodata/worldcereal/data/COP4GEOGLAM/moldova/production/V1_11092025/raw \
  --postprocess-dir /vitodata/worldcereal/data/COP4GEOGLAM/moldova/production/V1_11092025/postprocessed \
  --overwrite-final
"""

import argparse
import json
import subprocess
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


def exclude_and_reassign_classes(probs, ignore_classes=["cowpea", "sugarcane"]):
    classes_list = {
            "maize" : 0,
            "rice" : 1,
            "soybean" : 2,
            "sesame" : 3,
            "cassava" : 4,
            "cowpea" : 5,
            "sweet_potato" : 6,
            "pigeon_pea" : 7,
            "sugarcane" : 8,
            "other_crop" : 9,
        }
    # remove probabilities of classes to ignore
    filtered_probs = probs[[classes_list[c] for c in classes_list if c not in ignore_classes]]

    # reassign probabilities of ignored classes to 'other_crop' class
    filtered_probs[-1] += probs[[classes_list[c] for c in ignore_classes]].sum(axis=0)
    return filtered_probs

def create_cog(
    input_dir: Path,
    output_file: Path,
    class_dict: dict[int, str],
    colormap_json: Path = Path(
        "/home/giorgia/Private/git/worldcereal-cop4geoglam/src/worldcereal_cop4geoglam/data/mozambique/colormap_mozambique.json"
    ),
    overwrite: bool = False,
):
    """Create a final classification COG and apply palette.

    Parameters
    ----------
    input_dir : Path
        Directory containing the per-tile postprocessed classification GeoTIFFs.
    output_file : Path
        Desired path of the output COG.
    colormap_json : Path, optional
        JSON file describing the classification color map.
    """

    # Find all postprocessed classification files
    raw_files = sorted(list(input_dir.rglob("*croptype*.tif")))
    logger.info(f"Found {len(raw_files)} classification files")

    if not raw_files:
        logger.error("No raw files found")
        return False

    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already exists unless overwrite requested
    if output_file.exists() and not overwrite:
        logger.info(
            f"Output file {output_file} already exists (use overwrite=True to regenerate)"
        )
        return True
    elif output_file.exists() and overwrite:
        logger.warning(f"Overwriting existing output file: {output_file}")
        try:
            output_file.unlink()
        except Exception as e:
            logger.error(f"Failed to remove existing file before overwrite: {e}")
            return False

    # Load colormap (for embedding in VRT)
    color_map_full = _load_colormap(colormap_json)
    color_map_rgb = {k: (v[0], v[1], v[2]) for k, v in color_map_full.items()}
    logger.debug(
        f"Loaded {len(color_map_rgb)} colormap entries (alpha stripped if provided)"
    )

    # Use output directory for temporary files so user can monitor progress
    temp_dir = output_file.parent
    temp_files: list[Path] = []  # Track temporary files for cleanup

    try:
        # Step 1: Create VRT mosaic with proper nodata handling
        vrt_file = temp_dir / "mosaic.vrt"
        temp_files.append(vrt_file)

        logger.info("Creating VRT mosaic for classification...")
        # Create VRT without explicit nodata to preserve all probability bands
        vrt_cmd = ["gdalbuildvrt", str(vrt_file)] + [str(f) for f in raw_files]

        try:
            subprocess.run(vrt_cmd, capture_output=True, text=True, check=True)
            logger.info(f"VRT mosaic created: {vrt_file}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create VRT mosaic: {e.stderr}")
            return False

        # NEW Step 2: Sieve small regions (remove blobs <4 pixels, 8-connected)
        sieve_file = temp_dir / "mosaic_sieved.tif"
        sieved_vrt_file = temp_dir / "mosaic_sieved.vrt"
        temp_files.extend([sieve_file, sieved_vrt_file])
        base_vrt_for_next = vrt_file  # fallback if sieving fails
        try:
            logger.info("Applying gdal_sieve (threshold=4, connectivity=8)...")
            sieve_cmd = [
                "gdal_sieve.py",
                "-st",
                "4",
                "-8",
                "-of",
                "GTiff",
                str(vrt_file),
                str(sieve_file),
            ]
            subprocess.run(sieve_cmd, capture_output=True, text=True, check=True)
            logger.info(f"Sieve complete → {sieve_file}")

            # Re-apply band description + colormap (gdal_sieve may drop them)
            try:
                import rasterio

                with rasterio.open(sieve_file, "r+") as ds:
                    # Ensure single band classification dataset
                    ds.set_band_description(1, "Classification")
                    ds.write_colormap(1, color_map_rgb)
                logger.debug("Restored band description and colormap on sieved raster")
            except Exception as meta_err:
                logger.warning(
                    f"Failed to restore metadata on sieved raster: {meta_err}"
                )

            # Build a new VRT from the sieved GeoTIFF
            build_sieved_vrt_cmd = [
                "gdalbuildvrt",
                str(sieved_vrt_file),
                str(sieve_file),
            ]
            subprocess.run(
                build_sieved_vrt_cmd, capture_output=True, text=True, check=True
            )
            logger.info(f"Sieved VRT created: {sieved_vrt_file}")
            base_vrt_for_next = sieved_vrt_file
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Sieve step failed (continuing with unsieved mosaic): {e.stderr}"
            )
        except FileNotFoundError:
            logger.warning(
                "gdal_sieve.py not found in PATH; skipping sieving (install GDAL python utilities to enable)"
            )

        # Step 3: Create intermediate VRT with band descriptions (from sieved or original)
        logger.info("Creating intermediate VRT with band descriptions...")
        vrt_with_bands_file = temp_dir / "mosaic_with_bands.vrt"
        temp_files.append(vrt_with_bands_file)

        with open(base_vrt_for_next, "r") as f:
            vrt_content = f.read()

        vrt_content = vrt_content.replace(
            '<VRTRasterBand dataType="Byte" band="1">',
            '<VRTRasterBand dataType="Byte" band="1">\n'
            "    <Description>Classification</Description>\n"
            "    <ColorInterp>Palette</ColorInterp>",
        )

        with open(vrt_with_bands_file, "w") as f:
            f.write(vrt_content)
        logger.info(f"VRT with band descriptions created: {vrt_with_bands_file}")

        # Step 4: Create final COG (unchanged logic)
        logger.info("Creating raw COG with all optimizations...")

        # Create integer:class name lookup table metadata
        class_lookup = []
        for lab, class_name in class_dict.items():
            class_lookup.extend(["-mo", f"CLASS_{lab:03d}={class_name}"])

        # Single COG creation command with all optimizations
        cog_cmd = (
            [
                "gdal_translate",
                "-of",
                "COG",
                "-co",
                "COMPRESS=deflate",
                "-co",
                "TILED=YES",
                "-co",
                "BLOCKSIZE=512",
                "-co",
                "BIGTIFF=IF_SAFER",
                "-co",
                "NUM_THREADS=ALL_CPUS",
                "-co",
                "INTERLEAVE=PIXEL",
                "-a_nodata",
                "255",
                # Force no mask auto-detection
                "-mask",
                "none",
                # Per-band color interpretation flags (broader GDAL compatibility)
                "-colorinterp_1",
                "palette",
                "-mo",
                f"NUM_CLASSES={len(class_dict)}",
                "-mo",
                "NUM_BANDS=1",
                "-mo",
                "NODATA_VALUE=255",
                "-mo",
                "CREATOR=VITO",
                "-mo",
                "PROJECT=COPERNICUS4GEOGLAM",
            ]
            + class_lookup
            + [str(vrt_with_bands_file), str(output_file)]
        )

        try:
            subprocess.run(cog_cmd, capture_output=True, text=True, check=True)
            logger.info(f"Raw COG created with metadata and nodata: {output_file}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create raw COG: {e.stderr}")
            return False

        # Step 5: Validate COG structure
        logger.info("Validating raw COG...")
        validate_cmd = ["gdalinfo", "-checksum", str(output_file)]

        try:
            validation_proc = subprocess.run(
                validate_cmd, capture_output=True, text=True, check=True
            )
            info_text = validation_proc.stdout
            if "Color Table" in info_text:
                logger.success("Raw COG validation successful (palette present)")
            else:
                logger.warning(
                    "Raw COG validation ok but no 'Color Table' found in gdalinfo output"
                )
        except subprocess.CalledProcessError as e:
            logger.warning(f"Raw COG validation failed: {e.stderr}")

        # Step 6: Clean up temporary files after successful completion
        logger.info("Cleaning up temporary files for raw COG...")
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()
                logger.debug(f"Removed temporary file: {temp_file}")

        # Also clean up any auxiliary XML files that might have been created
        for xml_file in temp_dir.glob("*.xml"):
            if xml_file.exists():
                xml_file.unlink()
                logger.debug(f"Removed auxiliary XML file: {xml_file}")

        logger.success(
            "Raw COG creation and cleanup completed (colormap embedded in VRT)"
        )
        return True

    except Exception as e:
        logger.error(f"Error during raw COG creation: {e}")
        # Clean up temporary files on error too
        for temp_file in temp_files:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                    logger.debug(f"Cleaned up temporary file after error: {temp_file}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up {temp_file}: {cleanup_error}")
        return False

def _load_colormap(colormap_json: Path) -> dict[int, tuple[int, int, int, int]]:
    """Load a worldcereal style colormap JSON into a GDAL/Rasterio compatible dict.

    The JSON is expected to map integer class ids (as strings) to an object
    containing a "color" list (RGB or RGBA) and optionally a name/description.
    We ensure all colors are 4-tuples RGBA with alpha=255 (forcing opaque if needed).
    """
    with open(colormap_json) as f:
        cmap_raw = json.load(f)
    color_map: dict[int, tuple[int, int, int, int]] = {}
    for k, v in cmap_raw.items():
        col = v["color"]
        if len(col) == 3:  # RGB
            r, g, b = col
            a = 255
        elif len(col) == 4:  # RGBA
            r, g, b, a = col
            if a != 255:
                logger.warning(
                    f"Colormap entry {k} has non-opaque alpha ({a}); forcing to 255 for GeoTIFF palette"
                )
                a = 255
        else:
            raise ValueError(
                f"Expected RGB or RGBA list for class {k}, got {col} (len={len(col)})"
            )
        color_map[int(k)] = (r, g, b, a)
    return color_map

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

def get_croptype_prediction(croptype_probs, classes_dict, thresholds, ignore_classes=["cowpea", "sugarcane"]):

    if isinstance(thresholds, (int, float)):
        thresholds = [thresholds] * len(classes_dict["single_crop_classes"])
    elif isinstance(thresholds, list):
        if len(thresholds) != len(classes_dict["single_crop_classes"]):
            raise ValueError(
                f"Length of thresholds ({len(thresholds)}) does not match number of single classes ({len(classes_dict['single_crop_classes'])})."
            )

    # requires other_mix to be assigned with the max label.
    # we retrieve this value to use it to assign the mixed crops not foreseen in our classes
    # and to assign it to pixels where no class passed the thresholding
    other_mixed_value = sorted(classes_dict["mixed_crops_classes"].keys())[-1]

    # reassign excluded classes
    logger.debug(f"Excluding and reassigning classes from croptype prediction: {ignore_classes}")
    croptype_probs = exclude_and_reassign_classes(croptype_probs, ignore_classes=ignore_classes)

    # apply thresholds to get binary labels
    croptype_labels = croptype_probs.copy()
    for i, thr in enumerate(thresholds):
        croptype_labels[i] = np.where(croptype_labels[i] >= thr, 1, 0)

    # generate prediction by concatenating the binary labels
    concat_matrix = np.full(croptype_labels.shape[1:], "", dtype=object)
    for i in range(len(croptype_labels)):
        mask = croptype_labels[i].astype(bool)
        concat_matrix[mask] += str(i + 1)

    concat_matrix = np.where(concat_matrix == "", other_mixed_value, concat_matrix) # keep track of no data

    # cap all mixed classes with unmapped label to other_mixed value
    all_values = list(classes_dict["mixed_crops_classes"].keys()) + list(classes_dict["single_crop_classes"].keys())
    all_values = np.array(all_values).astype(str)

    # assign other_mix class to mixed classes not in our defined classes
    mask_invalid = ~np.isin(concat_matrix, all_values)
    concat_matrix[mask_invalid] = other_mixed_value

    croptype_pred = concat_matrix.astype(np.uint8)

    return croptype_pred

def process_tile(
    tile: Path,
    output_dir: Path,
    country_geom_geojson: dict | None,
    geom_crs: str | None,
    classes_dict: dict,
    thresholds: list[float] | float,
    country_name: str,
    no_crop_value: int = 254,
    nodata: int = 255,
    ignore_classes: list[str] = ["cowpea", "sugarcane"],
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
    logger.info(f"Postprocessing tile: {tile_id}")

    with rasterio.open(tile) as src:
        raw_results = src.read()
        src_profile = src.profile
        dst_crs = src.crs
        transform = src.transform
        height, width = src.height, src.width

    # Optional: spatial smoothing
    raw_results = spatial_smoothing(raw_results)

    # Open a new rasterio dataset based on the profile of the source
    src_profile.update(dtype="uint8", count=1, compress="deflate", nodata=nodata)

    # -----------------------------------------------------
    # Get the classification of croptype
    clf_array = get_croptype_prediction(raw_results, classes_dict, thresholds, ignore_classes=ignore_classes)
    # -----------------------------------------------------

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
                clf_array[:] = nodata
            else:
                outside_mask = ~inside_mask
                clf_array[outside_mask] = nodata
        except Exception as e:
            logger.error(f"Failed to apply geometry mask on tile {tile_id}: {e}")
    else:
        logger.debug("No country geometry provided; skipping masking step")

    # Get cropland mask: CHANGE PATH TO CHOSEN CROPLAND PRODUCT
    cropland_file = Path(
        "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique/production/v4_landcover/raw/cropland-combined"
    ) / tile.name.replace("croptype", "cropland")
    with rasterio.open(cropland_file) as src_cropland:
        cropland_mask = src_cropland.read(1) == 1
        nodata_mask = src_cropland.read(1) == nodata

    clf_array[~cropland_mask] = no_crop_value  # Now set to no_crop
    clf_array[nodata_mask] = nodata  # Set nodata to nodata

    output_path = output_dir / tile.name

    # # Get color table (classification palette) A COLORMAP SHOULD STILL BE PROVIDED
    colormap_json = "/home/giorgia/Private/git/worldcereal-cop4geoglam/src/worldcereal_cop4geoglam/data/mozambique/colormap_mozambique.json"
    colormap = _load_colormap(Path(colormap_json))

    # flatten the dicts into one
    classes_dict = {**classes_dict["single_crop_classes"], **classes_dict["mixed_crops_classes"]}
    classes_dict[no_crop_value] = "no_crop"
    with rasterio.open(output_path, "w", **src_profile) as dst:
        dst.write(clf_array, 1)
        dst.set_band_description(1, "Classification")

        # Set color table for the classification band

        dst.write_colormap(1, colormap)

        # Write metadata
        metadata = {}
        metadata["nodata_value"] = str(nodata)
        metadata["description"] = "Crop Classification"
        metadata["classes"] = json.dumps({
            f"class_{i}": class_name for i, class_name in classes_dict.items()
        })
        metadata["country"] = country_name
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
        "--generate_cog",
        action="store_true",
        help="Generate Cloud Optimized GeoTIFF (COG) from the processed tiles",
    )
    parser.add_argument(
        "--overwrite-final",
        action="store_true",
        help="Overwrite existing final COG if present",
    )
    parser.add_argument(
        "--reprocess-tiles",
        action="store_true",
        help="Reprocess all tiles, even if they have been processed before",
    )

    manual_args = [
        "--country",
        "Zambézia",
        "--vector-file",
        "/vitodata/worldcereal/auxdata/Gaul/GAUL_2024/GAUL_2024_L1.gpkg",
        "--raw-dir",
        "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique/production/v1_croptype/raw",
        "--postprocess-dir",
        "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique/production/v1_croptype_giorgia/raw/postprocessed",
        "--reprocess-tiles",
        "--generate_cog",
        # "--overwrite-final",
    ]
    # manual_args = None

    args = parser.parse_args(manual_args)

    raw_output_path: Path = args.raw_dir
    output_dir: Path = args.postprocess_dir

    no_crop_value = 254

    # thresholds = 0.2
    # thresholds = [0.20, 0.24, 0.20, 0.24, 0.23, 0.20, 0.24, 0.20]
    # thresholds = [0.18, 0.24, 0.20, 0.24, 0.23, 0.20, 0.24, 0.25]
    # thresholds = [0.18, 0.24, 0.19, 0.24, 0.22, 0.19, 0.24, 0.28]
    # thresholds = [0.18, 0.24, 0.17, 0.24, 0.22, 0.17, 0.25, 0.32]
    # thresholds = [0.19, 0.25, 0.16, 0.25, 0.22, 0.15, 0.26, 0.32]
    thresholds = [0.18, 0.24, 0.16, 0.25, 0.21, 0.15, 0.24, 0.30]  # nodata replaced with other_mixed

    classes_dict = {
        "mixed_crops_classes": {
            15: "maize-cassava",
            57: "cassava-pigeon_pea",
            157: "maize-cassava-pigeon_pea",
            200: "other_mixed",
        },
        "single_crop_classes":{
            1: "maize",
            2: "rice",
            3: "soybean",
            4: "sesame",
            5: "cassava",
            # 6: "cowpea",
            6: "sweet_potato",
            7: "pigeon_pea",
            # 9: "sugarcane",
            8: "other_crop",
        }
    }

    classes_dict_mapping = {**classes_dict["single_crop_classes"], **classes_dict["mixed_crops_classes"]}
    classes_dict_mapping[no_crop_value] = "no_crop"

    output_dir = output_dir.with_name(output_dir.name + f"_th{str(thresholds)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ### DEBUG
    # failing_tag = ['MOZ_1204', 'MOZ_1767']
    # raw_tiles = list(raw_output_path.rglob("*croptype_*.tif"))
    # raw_tiles = [r for r in raw_tiles for tag in failing_tag if tag in str(r)]
    # for i, item in enumerate(raw_tiles):
    #     if 'processed' in str(item):
    #         continue
    #     logger.debug(f"Queueing tile {i + 1}/{len(raw_tiles)}: {item.name}")
    #     process_tile(
    #             item,
    #             output_dir,
    #             None,
    #             None,
    #             classes_dict,
    #             thresholds,
    #             args.country,
    #             ignore_classes=["cowpea", "sugarcane"],
    #             nodata=255,
    #         )
    # #### END DEBUG

    if args.reprocess_tiles:
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
                country_geom_geojson = None, None
                raise
        else:
            country_geom_geojson = None, None

        # Get raw tiles
        raw_tiles = list(raw_output_path.rglob("*croptype_*.tif"))
        # raw_tiles = [r for r in raw_tiles for tag in failing_tag if tag in str(r)]
        logger.info(f"Found {len(raw_tiles)} raw tiles")
        if not raw_tiles:
            logger.error("No raw tiles found; aborting")
            return

        completed = 0
        failed = 0
        workers = max(1, args.workers)

        failing_errors = []
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = []
                for i, item in enumerate(raw_tiles):
                    if 'processed' in str(item):
                        continue
                    logger.debug(f"Queueing tile {i + 1}/{len(raw_tiles)}: {item.name}")
                    futures.append(
                        executor.submit(
                            process_tile,
                            item,
                            output_dir,
                            country_geom_geojson,
                            geom_crs,
                            classes_dict,
                            thresholds,
                            args.country,
                            no_crop_value,
                            nodata=255,
                            ignore_classes=["cowpea", "sugarcane"],
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
                        failing_errors.append((i, raw_tiles[i].stem, e))
        except Exception as e:
            logger.error(f"ThreadPoolExecutor failed: {e}")
            raise

        logger.info(
            f"Postprocessing complete: {completed} successful, {failed} failed (country mask: {args.country if country_geom_geojson else 'NONE'})"
        )

        # save error log
        if failing_errors:
            error_log_path = output_dir / "postprocessing_errors.log"
            with open(error_log_path, "w") as f:
                for i, tile_stem, err in failing_errors:
                    f.write(f"Tile {i + 1} ({tile_stem}): {err}\n")
            logger.info(f"Error log saved to {error_log_path}")

        logger.info("saving class mapping")
        # save class mapping
        class_mapping_path = output_dir / "class_mapping.json"
        with open(class_mapping_path, "w") as f:
            json.dump(classes_dict_mapping, f, indent=4)
        logger.success("All done!")

    # generating cog
    final_cog = output_dir.parent / f"Copernicus4GEOGLAM_{args.country}_CropType_2025_th{str(thresholds)}.tif"

    if args.generate_cog:
        logger.info(f"Creating final COG: {final_cog}")
        create_cog(
            input_dir=output_dir,
            output_file=final_cog,
            class_dict=classes_dict_mapping,
            overwrite=args.overwrite_final,
        )

if __name__ == "__main__":
    main()
