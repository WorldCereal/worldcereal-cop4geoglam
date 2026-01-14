"""
python scripts/create_cog_croptype.py \
  --country moldova \
  --vector-file /vitodata/worldcereal/auxdata/Gaul/GAUL_2024/GAUL_2024_L0.gpkg \
  --raw-dir /vitodata/worldcereal/data/COP4GEOGLAM/moldova/production/V1_11092025/raw \
  --postprocess-dir /vitodata/worldcereal/data/COP4GEOGLAM/moldova/production/V1_11092025/postprocessed \
  --overwrite-final
"""

import argparse
import json
import subprocess
from pathlib import Path

from loguru import logger
from worldcereal.utils.models import load_model_lut


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


def create_cog(
    input_dir: Path,
    output_file: Path,
    colormap_json: Path = Path(
        "/home/kristofvt/git/worldcereal-cop4geoglam/src/worldcereal_cop4geoglam/data/moldova/colormap_moldova.json"
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
    raw_files = sorted(list(input_dir.rglob("croptype_*.tif")))
    logger.info(f"Found {len(raw_files)} classification files")

    if not raw_files:
        logger.error(f"No raw files found in {input_dir}")
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

        model_url = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/Copernicus4Geoglam/moldova/Presto_run%3D202509110852_DownstreamCatBoost_croptype_v120-MDA_balance%3DTrue.onnx"
        logger.info(f"Loading class LUT from model: {model_url}")
        class_lut = load_model_lut(model_url)
        class_list = ["_".join(x.split("_")[1:]) for x in list(class_lut.keys())]

        with open(vrt_with_bands_file, "w") as f:
            f.write(vrt_content)
        logger.info(f"VRT with band descriptions created: {vrt_with_bands_file}")

        # Step 4: Create final COG (unchanged logic)
        logger.info("Creating raw COG with all optimizations...")

        # Create integer:class name lookup table metadata
        class_lookup = []
        for i, class_name in enumerate(class_list):
            class_lookup.extend(["-mo", f"CLASS_{i}={class_name}"])

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
                f"NUM_CLASSES={len(class_list)}",
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


def main():
    parser = argparse.ArgumentParser(description="Postprocess tiles into masked COG")
    parser.add_argument(
        "--postprocess-dir",
        type=Path,
        required=False,
        default=Path(
            "/vitodata/worldcereal/data/COP4GEOGLAM/moldova/production/V3_13012026/postprocessed"
        ),
        help="Directory with postprocessed (masked) tiles",
    )
    parser.add_argument(
        "--country",
        type=str,
        required=False,
        default="moldova",
        help="Country name to mask (case-insensitive)",
    )
    parser.add_argument(
        "--overwrite-final",
        action="store_true",
        help="Overwrite existing final COG if present",
    )

    # Manual args override (matches example at top of script)
    manual_args_example = [
        "--country",
        "Moldova",
        "--postprocess-dir",
        "/vitodata/worldcereal/data/COP4GEOGLAM/moldova/production/V3_13012026/postprocessed",
        "--overwrite-final",
    ]
    # manual_args_example = None  # Set to None to use actual command line args

    if manual_args_example is not None:
        args = parser.parse_args(manual_args_example)
    else:
        args = parser.parse_args()

    postprocess_dir: Path = args.postprocess_dir
    country: str = args.country

    final_cog = (
        postprocess_dir.parent / f"Copernicus4GEOGLAM_{country}_CropType_2025.tif"
    )
    logger.info(f"Creating final COG: {final_cog}")
    create_cog(
        input_dir=postprocess_dir,
        output_file=final_cog,
        overwrite=args.overwrite_final,
    )

    logger.success("All done!")


if __name__ == "__main__":
    main()
