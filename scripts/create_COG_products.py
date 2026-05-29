"""Create final Cloud Optimized GeoTIFF (COG) products from postprocessed tiles.

Takes the output of postprocess_tiles.py and mosaics it into three COG files:

  1. cropland.tif        — 1-band cropland classification (0=no_cropland, 1=cropland, 255=nodata)
  2. croptype.tif        — 1-band croptype classification (class labels, 254=no_crop, 255=nodata)
  3. croptype-probs.tif  — N-band croptype probabilities (0–100 per class, 255=nodata)

Usage:
  python scripts/create_COG_products.py
"""

import json
import subprocess
from pathlib import Path

import rasterio
from loguru import logger

# ---------------------------------------------------------------------------
# CONFIGURATION — edit here to match the target production run
# ---------------------------------------------------------------------------

POSTPROCESSED_DIR = Path(
    "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique_pm/production/v5_PM/postprocessed"
)
CROPLAND_INPUT_DIR = POSTPROCESSED_DIR / "cropland"
CROPTYPE_INPUT_DIR = POSTPROCESSED_DIR / "croptype"
FINAL_DIR = POSTPROCESSED_DIR / "final"

CROPLAND_COLORMAP_JSON = Path(
    "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique_pm/production/colormap_cropland.json"
)
CROPTYPE_COLORMAP_JSON = Path(
    "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique_pm/production/colormap_croptype.json"
)

OVERWRITE = False
NODATA = 255
NO_CROP_VALUE = 254

# Class label → name mapping for the cropland COG metadata
CROPLAND_CLASSES: dict[int, str] = {
    0: "no_cropland",
    1: "cropland",
}

# Class label → name mapping for the croptype COG metadata (must match postprocess_tiles.py)
CROPTYPE_CLASSES_DICT: dict = {
    "single_crop_classes": {
        1: "maize",
        2: "rice",
        3: "soybean",
        4: "sesame",
        5: "cassava",
        6: "sweet_potato",
        7: "pigeon_pea",
    },
    "mixed_crops_classes": {
        15: "maize-cassava",
        17: "maize-pigeon_pea",
        57: "cassava-pigeon_pea",
        157: "maize-cassava-pigeon_pea",
        200: "other_crop/mixtures",
    },
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def _load_colormap(colormap_json: Path) -> dict[int, tuple[int, int, int, int]]:
    """Load a worldcereal-style colormap JSON into a GDAL/Rasterio RGBA dict."""
    with open(colormap_json) as f:
        cmap_raw = json.load(f)
    color_map: dict[int, tuple[int, int, int, int]] = {}
    for k, v in cmap_raw.items():
        col = v["color"]
        if len(col) == 3:
            r, g, b = col
            a = 255
        elif len(col) == 4:
            r, g, b, a = col
            if a != 255:
                logger.warning(
                    f"Colormap entry {k} has non-opaque alpha ({a}); forcing to 255"
                )
                a = 255
        else:
            raise ValueError(f"Expected RGB or RGBA for class {k}, got {col}")
        color_map[int(k)] = (r, g, b, a)
    return color_map


def _run_cmd(cmd: list[str], label: str) -> None:
    """Run a subprocess command, raising CalledProcessError on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"{label} failed:\n{result.stderr.strip()}")
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
    if result.stderr.strip():
        logger.debug(f"{label} stderr: {result.stderr.strip()}")


def _check_overwrite(output_file: Path, overwrite: bool) -> bool:
    """Return True if processing should proceed.

    Removes the existing file when overwrite is True; returns False (skip) otherwise.
    """
    if output_file.exists():
        if not overwrite:
            logger.info(
                f"Already exists, skipping (set OVERWRITE=True to regenerate): {output_file.name}"
            )
            return False
        logger.warning(f"Overwriting: {output_file}")
        output_file.unlink()
    return True


def _cleanup(temp_files: list[Path], temp_dir: Path, stem: str) -> None:
    """Remove temporary files and associated .aux.xml sidecar files."""
    for tf in temp_files:
        if tf.exists():
            try:
                tf.unlink()
                logger.debug(f"Removed temp file: {tf.name}")
            except Exception as e:
                logger.warning(f"Could not remove {tf.name}: {e}")
    for xml in temp_dir.glob(f"{stem}*.xml"):
        try:
            xml.unlink()
            logger.debug(f"Removed sidecar: {xml.name}")
        except Exception:
            pass


def _validate_cog(output_file: Path) -> None:
    """Run gdalinfo to verify the COG was written correctly."""
    try:
        proc = subprocess.run(
            ["gdalinfo", "-checksum", str(output_file)],
            capture_output=True,
            text=True,
            check=True,
        )
        if "Color Table" in proc.stdout:
            logger.success(f"[{output_file.name}] COG validation OK (palette present)")
        else:
            logger.info(f"[{output_file.name}] COG validation OK")
    except subprocess.CalledProcessError as e:
        logger.warning(f"[{output_file.name}] COG validation failed: {e.stderr.strip()}")


# ---------------------------------------------------------------------------
# COG CREATORS
# ---------------------------------------------------------------------------


def create_classification_cog(
    tiles: list[Path],
    output_file: Path,
    colormap_json: Path,
    class_dict: dict[int, str],
    source_band: int = 1,
    apply_sieve: bool = True,
    sieve_threshold: int = 4,
    overwrite: bool = False,
) -> bool:
    """Create a single-band classification COG from postprocessed tiles.

    Parameters
    ----------
    tiles : list[Path]
        Input tile GeoTIFFs.
    output_file : Path
        Output COG file path.
    colormap_json : Path
        Path to the colormap JSON file.
    class_dict : dict[int, str]
        Integer label → class name mapping, embedded as COG metadata.
    source_band : int
        Band to extract from input tiles (1-indexed). Pass 1 for single-band tiles
        or to select the classification band from multi-band tiles.
    apply_sieve : bool
        Whether to apply gdal_sieve to remove small isolated patches.
    sieve_threshold : int
        Minimum patch size in pixels to keep during sieving.
    overwrite : bool
        Overwrite existing output file.

    Returns
    -------
    bool
        True on success, False on failure.
    """
    if not _check_overwrite(output_file, overwrite):
        return True

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not tiles:
        logger.error(f"[{output_file.name}] No input tiles provided")
        return False

    logger.info(
        f"[{output_file.name}] Creating classification COG from {len(tiles)} tiles"
    )

    temp_dir = output_file.parent
    stem = output_file.stem
    temp_files: list[Path] = []

    try:
        color_map = _load_colormap(colormap_json)
        color_map_rgb = {k: (v[0], v[1], v[2]) for k, v in color_map.items()}

        # Step 1: VRT mosaic — explicitly select source_band so multi-band tiles
        # (e.g. cropland: band1=classification, band2=probability) yield a 1-band VRT.
        vrt_file = temp_dir / f"{stem}_mosaic.vrt"
        temp_files.append(vrt_file)

        _run_cmd(
            ["gdalbuildvrt", "-b", str(source_band), str(vrt_file)]
            + [str(t) for t in tiles],
            "gdalbuildvrt",
        )
        logger.info(f"[{output_file.name}] VRT mosaic → {vrt_file.name}")

        base_vrt = vrt_file

        # Step 2: Optional sieve (remove small isolated patches)
        if apply_sieve:
            sieve_tif = temp_dir / f"{stem}_sieved.tif"
            sieve_vrt = temp_dir / f"{stem}_sieved.vrt"
            temp_files += [sieve_tif, sieve_vrt]
            try:
                _run_cmd(
                    [
                        "gdal_sieve.py",
                        "-st",
                        str(sieve_threshold),
                        "-8",
                        "-of",
                        "GTiff",
                        str(vrt_file),
                        str(sieve_tif),
                    ],
                    "gdal_sieve",
                )
                # Restore band description + colormap (gdal_sieve may strip them)
                with rasterio.open(sieve_tif, "r+") as ds:
                    ds.set_band_description(1, "Classification")
                    ds.write_colormap(1, color_map_rgb)
                _run_cmd(
                    ["gdalbuildvrt", str(sieve_vrt), str(sieve_tif)],
                    "gdalbuildvrt (sieved)",
                )
                base_vrt = sieve_vrt
                logger.info(
                    f"[{output_file.name}] Sieve applied (threshold={sieve_threshold}) → {sieve_tif.name}"
                )
            except subprocess.CalledProcessError as e:
                logger.warning(
                    f"[{output_file.name}] Sieve failed (using unsieved mosaic): {e}"
                )
            except FileNotFoundError:
                logger.warning(
                    f"[{output_file.name}] gdal_sieve.py not found; skipping sieve"
                )

        # Step 3: Inject Description + ColorInterp into VRT XML
        desc_vrt = temp_dir / f"{stem}_desc.vrt"
        temp_files.append(desc_vrt)

        with open(base_vrt) as f:
            vrt_content = f.read()

        vrt_content = vrt_content.replace(
            '<VRTRasterBand dataType="Byte" band="1">',
            '<VRTRasterBand dataType="Byte" band="1">\n'
            "    <Description>Classification</Description>\n"
            "    <ColorInterp>Palette</ColorInterp>",
        )

        with open(desc_vrt, "w") as f:
            f.write(vrt_content)

        # Step 4: gdal_translate → COG
        class_meta: list[str] = []
        for lab, name in class_dict.items():
            class_meta += ["-mo", f"CLASS_{lab:03d}={name}"]

        _run_cmd(
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
                str(NODATA),
                "-mask",
                "none",
                "-colorinterp_1",
                "palette",
                "-mo",
                f"NUM_CLASSES={len(class_dict)}",
                "-mo",
                "NUM_BANDS=1",
                "-mo",
                f"NODATA_VALUE={NODATA}",
                "-mo",
                "CREATOR=VITO",
                "-mo",
                "PROJECT=COPERNICUS4GEOGLAM",
            ]
            + class_meta
            + [str(desc_vrt), str(output_file)],
            "gdal_translate (COG)",
        )

        _validate_cog(output_file)
        logger.success(f"[{output_file.name}] Created: {output_file}")
        return True

    except Exception as e:
        logger.error(f"[{output_file.name}] Failed: {e}")
        if output_file.exists():
            output_file.unlink()
        return False

    finally:
        _cleanup(temp_files, temp_dir, stem)


def create_probs_cog(
    tiles: list[Path],
    output_file: Path,
    overwrite: bool = False,
) -> bool:
    """Create a multi-band probability COG from postprocessed tiles.

    Band descriptions (e.g. ``'probability_maize'``) are read from the first
    tile and embedded in the output COG as per-band descriptions and metadata.

    Parameters
    ----------
    tiles : list[Path]
        Input tile GeoTIFFs (all must share the same band layout).
    output_file : Path
        Output COG file path.
    overwrite : bool
        Overwrite existing output file.

    Returns
    -------
    bool
        True on success, False on failure.
    """
    if not _check_overwrite(output_file, overwrite):
        return True

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not tiles:
        logger.error(f"[{output_file.name}] No input tiles provided")
        return False

    logger.info(
        f"[{output_file.name}] Creating probability COG from {len(tiles)} tiles"
    )

    # Read band count and descriptions from the first tile
    with rasterio.open(tiles[0]) as src:
        n_bands = src.count
        band_descriptions = [
            src.descriptions[i] or f"probability_band_{i + 1}" for i in range(n_bands)
        ]
    logger.info(f"[{output_file.name}] {n_bands} bands: {band_descriptions}")

    temp_dir = output_file.parent
    stem = output_file.stem
    temp_files: list[Path] = []

    try:
        # Step 1: VRT mosaic (all bands)
        vrt_file = temp_dir / f"{stem}_mosaic.vrt"
        temp_files.append(vrt_file)

        _run_cmd(
            ["gdalbuildvrt", str(vrt_file)] + [str(t) for t in tiles],
            "gdalbuildvrt",
        )
        logger.info(f"[{output_file.name}] VRT mosaic → {vrt_file.name}")

        # Step 2: Inject per-band descriptions into VRT XML
        desc_vrt = temp_dir / f"{stem}_desc.vrt"
        temp_files.append(desc_vrt)

        with open(vrt_file) as f:
            vrt_content = f.read()

        for i, desc in enumerate(band_descriptions, start=1):
            vrt_content = vrt_content.replace(
                f'<VRTRasterBand dataType="Byte" band="{i}">',
                f'<VRTRasterBand dataType="Byte" band="{i}">\n'
                f"    <Description>{desc}</Description>",
            )

        with open(desc_vrt, "w") as f:
            f.write(vrt_content)

        # Step 3: gdal_translate → COG
        band_meta: list[str] = []
        for i, desc in enumerate(band_descriptions, start=1):
            band_meta += ["-mo", f"BAND_{i:02d}={desc}"]

        _run_cmd(
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
                str(NODATA),
                "-mask",
                "none",
                "-mo",
                f"NUM_BANDS={n_bands}",
                "-mo",
                f"NODATA_VALUE={NODATA}",
                "-mo",
                "CREATOR=VITO",
                "-mo",
                "PROJECT=COPERNICUS4GEOGLAM",
            ]
            + band_meta
            + [str(desc_vrt), str(output_file)],
            "gdal_translate (COG)",
        )

        _validate_cog(output_file)
        logger.success(f"[{output_file.name}] Created: {output_file}")
        return True

    except Exception as e:
        logger.error(f"[{output_file.name}] Failed: {e}")
        if output_file.exists():
            output_file.unlink()
        return False

    finally:
        _cleanup(temp_files, temp_dir, stem)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main() -> None:
    logger.info("Starting COG product creation")
    logger.info(f"  CROPLAND_INPUT : {CROPLAND_INPUT_DIR}")
    logger.info(f"  CROPTYPE_INPUT : {CROPTYPE_INPUT_DIR}")
    logger.info(f"  FINAL_DIR      : {FINAL_DIR}")
    logger.info(f"  OVERWRITE      : {OVERWRITE}")

    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, bool] = {}

    # ── 1. Cropland classification COG ──────────────────────────────────────
    # Cropland tiles have 2 bands: band 1 = classification (0/1/255),
    # band 2 = probability. We select band 1 via source_band=1.
    cropland_tiles = sorted(CROPLAND_INPUT_DIR.glob("cropland_*.tif"))
    logger.info(f"Cropland tiles found: {len(cropland_tiles)}")

    results["cropland"] = create_classification_cog(
        tiles=cropland_tiles,
        output_file=FINAL_DIR / "cropland.tif",
        colormap_json=CROPLAND_COLORMAP_JSON,
        class_dict=CROPLAND_CLASSES,
        source_band=1,
        apply_sieve=True,
        overwrite=OVERWRITE,
    )

    # ── 2. Croptype classification COG ───────────────────────────────────────
    # Croptype tiles are named croptype_*.tif (single classification band).
    # The glob croptype_*.tif does not match croptype-probs_*.tif files.
    croptype_tiles = sorted(CROPTYPE_INPUT_DIR.glob("croptype_*.tif"))
    logger.info(f"Croptype tiles found: {len(croptype_tiles)}")

    flat_croptype_classes: dict[int, str] = {
        **CROPTYPE_CLASSES_DICT["single_crop_classes"],
        **CROPTYPE_CLASSES_DICT["mixed_crops_classes"],
        NO_CROP_VALUE: "no_crop",
    }

    results["croptype"] = create_classification_cog(
        tiles=croptype_tiles,
        output_file=FINAL_DIR / "croptype.tif",
        colormap_json=CROPTYPE_COLORMAP_JSON,
        class_dict=flat_croptype_classes,
        source_band=1,
        apply_sieve=True,
        overwrite=OVERWRITE,
    )

    # ── 3. Croptype probabilities COG ────────────────────────────────────────
    # Probability tiles are named croptype-probs_*.tif (N bands, 0–100 per class).
    probs_tiles = sorted(CROPTYPE_INPUT_DIR.glob("croptype-probs_*.tif"))
    logger.info(f"Croptype-probs tiles found: {len(probs_tiles)}")

    results["croptype-probs"] = create_probs_cog(
        tiles=probs_tiles,
        output_file=FINAL_DIR / "croptype-probs.tif",
        overwrite=OVERWRITE,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("─" * 60)
    for product, ok in results.items():
        logger.info(f"  {product:<20} {'OK' if ok else 'FAILED'}")
    if all(results.values()):
        logger.success("All COG products created successfully")
    else:
        failed = [k for k, v in results.items() if not v]
        logger.error(f"Failed products: {failed}")


if __name__ == "__main__":
    main()
