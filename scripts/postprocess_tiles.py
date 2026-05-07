"""Per-tile postprocessing script for COP4GEOGLAM Mozambique PM production.

Processes raw cropland and croptype tiles (probability arrays):
  - Spatial smoothing of probability bands
  - ROI masking (Zambezia province)
  - Road masking (cropland only)
  - Croptype reclassification from smoothed probabilities
  - Optional reprojection to a target CRS
  - Writing postprocessed tiles to output directories

COG generation is handled by a separate script.

Outputs per tile:
  1. cropland/cropland_*.tif          — 2-band: classification (0/1/255) + probability (0-100, 255=nodata)
  2. croptype/croptype_*.tif          — 1-band: class label (uint8, 254=no_crop, 255=nodata)
  3. croptype/croptype-probs_*.tif    — N-band: per-class probabilities (0-100, 255=nodata)
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fiona
import numpy as np
import rasterio
from loguru import logger
from rasterio.crs import CRS
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.warp import transform_geom
from scipy.signal import convolve2d
from shapely.geometry import mapping, shape
from shapely.ops import unary_union
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIGURATION — edit here to adjust for a different run
# ---------------------------------------------------------------------------

RAW_DIR = Path(
    "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique_pm/production/v5_PM/raw"
)
OUTPUT_DIR = Path(
    "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique_pm/production/v5_PM/postprocessed"
)
CROPLAND_OUTPUT_DIR = OUTPUT_DIR / "cropland"
CROPTYPE_OUTPUT_DIR = OUTPUT_DIR / "croptype"

ROI_GPKG = Path("/vitodata/worldcereal/auxdata/Gaul/GAUL_2024/GAUL_2024_L1.gpkg")
ROI_NAME = "Zambézia"
# Field in the GAUL L1 GeoPackage that holds province names.
# Common candidates: "gaul1_name", "adm1_name".
# A startup check prints available fields if the name is not found.
ROI_NAME_FIELD = "gaul1_name"

ROADS_DIR = Path(
    "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique_pm/auxdata/osm_roads_rasterized"
)

DO_SMOOTH_CROPLAND = False
DO_SMOOTH_CROPTYPE = True
DO_REPROJECT = True
TARGET_EPSG = 32737

NUM_WORKERS = 4
NODATA = 255
NO_CROP_VALUE = 254

# ---------------------------------------------------------------------------
# Croptype classification parameters (from mozambique / v5_PM calibration)
# ---------------------------------------------------------------------------

IGNORE_CLASSES = []

# Per-class detection thresholds (fraction, 0-1), keyed by class name.
# Any class in CLASSES_DICT["single_crop_classes"] not listed here gets THRESHOLD_DEFAULT.
THRESHOLD_DEFAULT = 0.5
THRESHOLDS = {
    "maize": 0.35,
    "rice": 0.24,
    "soybean": 0.16,
    "sesame": 0.45,
    "cassava": 0.30,
    "sweet_potato": 0.25,
    "pigeon_pea": 0.24,
}

CLASSES_DICT = {
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
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------


def load_roi_geometry(gpkg: Path, name_field: str, name: str) -> tuple[dict, str]:
    """Load and union the ROI geometry from a GeoPackage.

    Parameters
    ----------
    gpkg : Path
        Path to the GeoPackage file.
    name_field : str
        Attribute field containing the region name.
    name : str
        Region name to match (case-insensitive substring match).

    Returns
    -------
    geojson_dict : dict
        GeoJSON-like geometry dict (union of matched features).
    crs_str : str
        CRS of the geometry as WKT or EPSG string.
    """
    name_lower = name.lower()
    with fiona.open(gpkg) as src:
        crs = src.crs_wkt or (src.crs.to_string() if src.crs else "EPSG:4326")
        available_fields = list(src.schema["properties"].keys())
        logger.debug(f"Available fields in {gpkg.name}: {available_fields}")
        matched = []
        for feat in src:
            val = str(feat["properties"].get(name_field, "")).lower()
            if name_lower in val:
                matched.append(shape(feat["geometry"]))

    if not matched:
        raise ValueError(
            f"No features found for '{name}' in field '{name_field}' of {gpkg}.\n"
            f"Available fields: {available_fields}"
        )

    geom = unary_union(matched)
    logger.info(f"Loaded ROI geometry for '{name}' ({len(matched)} feature(s))")
    return mapping(geom), crs


def find_road_mask(tile_folder_name: str, roads_dir: Path) -> Path | None:
    """Find the road mask raster matching a tile folder name.

    Road mask files follow the pattern: ``*{tile_folder_name}.tif``
    (e.g. ``croptype_..._zambezia_36KYE_18.tif`` for folder ``zambezia_36KYE_18``).

    Parameters
    ----------
    tile_folder_name : str
        Tile folder name, e.g. ``'zambezia_36KYE_18'``.
    roads_dir : Path
        Directory containing rasterized road masks.

    Returns
    -------
    Path | None
        Path to the road mask file, or None if not found.
    """
    matches = list(roads_dir.glob(f"*{tile_folder_name}.tif"))
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning(
            f"Multiple road mask files for '{tile_folder_name}', using: {matches[0].name}"
        )
    return matches[0]


def spatial_smoothing(prob_array: np.ndarray, nodata: int = 255) -> np.ndarray:
    """Apply spatial smoothing to a multi-band probability array.

    Input values are in [0, 100] with ``nodata`` (default 255) marking missing pixels.
    Nodata pixels are zeroed before convolution to avoid contamination.
    The output is float32, normalized so probabilities sum to 1 per valid pixel.

    Parameters
    ----------
    prob_array : np.ndarray
        Shape ``(bands, H, W)``, uint8-compatible, values 0-100 + nodata value.
    nodata : int
        Nodata sentinel value (default 255).

    Returns
    -------
    np.ndarray
        float32, shape ``(bands, H, W)``, values in [0, 1].
        Nodata pixels have value 0 across all bands (caller must restore nodata mask).
    """
    arr = prob_array.astype("float32")

    # Zero out nodata pixels so they don't bleed into the convolution
    nodata_mask = np.any(arr == nodata, axis=0)  # (H, W)
    arr[:, nodata_mask] = 0.0

    conv_kernel = np.array([[1, 2, 1], [2, 3, 2], [1, 2, 1]], dtype=np.float32)

    for band_idx in range(arr.shape[0]):
        arr[band_idx] = (
            convolve2d(arr[band_idx], conv_kernel, mode="same", boundary="symm")
            / conv_kernel.sum()
        )

    # Normalize so probabilities sum to 1 per pixel; guard against zero-sum
    band_sum = arr.sum(axis=0)
    band_sum[band_sum == 0] = 1.0
    arr = arr / band_sum

    return arr  # [0, 1] float32; nodata pixels have ~0 values


def apply_roi_mask(
    array: np.ndarray,
    geom_json: dict,
    geom_crs: str,
    src_crs,
    transform,
    height: int,
    width: int,
    nodata: int = 255,
) -> np.ndarray:
    """Set pixels outside the ROI geometry to nodata.

    Parameters
    ----------
    array : np.ndarray
        2-D ``(H, W)`` or 3-D ``(bands, H, W)`` uint8 array; modified in-place.
    geom_json : dict
        GeoJSON-like geometry in ``geom_crs``.
    geom_crs : str
        CRS of the geometry (WKT or EPSG string).
    src_crs : rasterio.crs.CRS
        CRS of the raster tile.
    transform : affine.Affine
        Tile affine transform.
    height, width : int
        Tile pixel dimensions.
    nodata : int
        Value to assign to outside-ROI pixels.

    Returns
    -------
    np.ndarray
        Array with outside-ROI pixels set to nodata.
    """
    if str(src_crs) != geom_crs:
        geom_in_tile_crs = transform_geom(geom_crs, src_crs.to_string(), geom_json)
    else:
        geom_in_tile_crs = geom_json

    inside_mask = geometry_mask(
        [geom_in_tile_crs],
        out_shape=(height, width),
        transform=transform,
        invert=True,  # True = inside the geometry
    )

    if inside_mask.sum() == 0:
        logger.warning("Tile has no overlap with ROI; writing full nodata")
        array[:] = nodata
        return array

    outside_mask = ~inside_mask
    if array.ndim == 2:
        array[outside_mask] = nodata
    else:
        array[:, outside_mask] = nodata

    return array


def apply_road_mask(array: np.ndarray, road_path: Path) -> np.ndarray:
    """Zero out pixels that fall on rasterized OSM roads.

    Parameters
    ----------
    array : np.ndarray
        2-D ``(H, W)`` or 3-D ``(bands, H, W)`` array; modified in-place.
    road_path : Path
        Rasterized road mask GeoTIFF (non-zero where roads exist, band 1).

    Returns
    -------
    np.ndarray
        Array with road pixels set to 0.
    """
    with rasterio.open(road_path) as src:
        road_mask = src.read(1) > 0

    if array.ndim == 2:
        array[road_mask] = 0
    else:
        array[:, road_mask] = 0

    return array


def write_tile(
    array: np.ndarray,
    profile: dict,
    out_path: Path,
    do_reproject: bool = False,
    target_epsg: int = 32737,
    band_descriptions: list[str] | None = None,
    tags: dict | None = None,
) -> None:
    """Write a raster tile, optionally reprojecting to a target CRS.

    Parameters
    ----------
    array : np.ndarray
        Shape ``(bands, H, W)``, uint8.
    profile : dict
        Rasterio write profile (already updated for band count / dtype / nodata).
        Must contain 'crs', 'transform', 'height', 'width'.
    out_path : Path
        Output file path (parent directory is created if needed).
    do_reproject : bool
        If True, reproject to ``target_epsg`` using nearest-neighbour resampling.
    target_epsg : int
        Target EPSG code (used only when ``do_reproject=True``).
    band_descriptions : list[str] | None
        Per-band descriptions to embed (length must match ``array.shape[0]``).
    tags : dict | None
        Dataset-level metadata tags to embed.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_bands = array.shape[0]
    nodata = profile.get("nodata", NODATA)

    if do_reproject:
        dst_crs = CRS.from_epsg(target_epsg)
        bounds = rasterio.transform.array_bounds(
            profile["height"], profile["width"], profile["transform"]
        )
        dst_transform, dst_width, dst_height = calculate_default_transform(
            profile["crs"], dst_crs, profile["width"], profile["height"], *bounds
        )
        out_profile = profile.copy()
        out_profile.update(
            crs=dst_crs,
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
        )
        dst_data = np.full((n_bands, dst_height, dst_width), nodata, dtype=array.dtype)
        for i in range(n_bands):
            reproject(
                source=array[i],
                destination=dst_data[i],
                src_transform=profile["transform"],
                src_crs=profile["crs"],
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
                src_nodata=nodata,
                dst_nodata=nodata,
            )
        write_array = dst_data
        write_profile = out_profile
    else:
        write_array = array
        write_profile = profile

    with rasterio.open(out_path, "w", **write_profile) as dst:
        dst.write(write_array)
        if band_descriptions:
            for i, desc in enumerate(band_descriptions, start=1):
                dst.set_band_description(i, desc)
        if tags:
            dst.update_tags(**tags)


def get_croptype_prediction(
    probs: np.ndarray,
    prob_class_names: list[str],
    classes_dict: dict,
    thresholds: dict[str, float],
    ignore_classes: list[str],
) -> np.ndarray:
    """Classify croptype probability bands into a single-band uint8 label map.

    Band-to-class mapping is derived from ``prob_class_names`` (read from the
    file's band descriptions), so no hardcoded band order is needed.

    Parameters
    ----------
    probs : np.ndarray
        Shape ``(N_prob_bands, H, W)`` float32, values in [0, 1].
        Each band corresponds to the class in ``prob_class_names``.
    prob_class_names : list[str]
        Class name for each band in ``probs`` (e.g. ``['maize', 'rice', ...]``).
        Derived by stripping the ``'probability_'`` prefix from band descriptions.
    classes_dict : dict
        With keys ``'single_crop_classes'`` (int label → name) and
        ``'mixed_crops_classes'`` (int label → name).
    thresholds : dict[str, float]
        Detection threshold per class name. Missing classes fall back to
        ``THRESHOLD_DEFAULT``.
    ignore_classes : list[str]
        Class names to skip entirely (treated as absent regardless of probability).

    Returns
    -------
    np.ndarray
        Shape ``(H, W)`` uint8. Values are integer labels from ``classes_dict``,
        or the highest mixed-class key for unrecognised combinations.
    """
    other_mixed_value = max(classes_dict["mixed_crops_classes"].keys())

    # Build per-class index lookup (class_name → band index in probs)
    single_classes = [
        (label, class_name)
        for label, class_name in classes_dict["single_crop_classes"].items()
        if class_name not in ignore_classes and class_name in prob_class_names
    ]
    for class_name in classes_dict["single_crop_classes"].values():
        if class_name not in ignore_classes and class_name not in prob_class_names:
            logger.warning(
                f"Class '{class_name}' not found in probability bands {prob_class_names}; skipping"
            )

    # Build concatenated label string per pixel by iterating single classes
    # in their label-key order (so string encoding is deterministic).
    concat = np.full(probs.shape[1:], "", dtype=object)
    for label, class_name in single_classes:
        band_idx = prob_class_names.index(class_name)
        thr = thresholds.get(class_name, THRESHOLD_DEFAULT)
        concat[probs[band_idx] >= thr] += str(label)

    all_valid = set(
        str(k)
        for k in list(classes_dict["single_crop_classes"].keys())
        + list(classes_dict["mixed_crops_classes"].keys())
    )

    # Build a (H*W,) argmax array over the active single-class bands for fallback use.
    # Index into single_classes list (not band index directly).
    active_band_indices = [prob_class_names.index(cn) for _, cn in single_classes]
    active_labels = np.array([lbl for lbl, _ in single_classes], dtype=np.uint8)
    if active_band_indices:
        active_probs = probs[active_band_indices]  # (N_active, H, W)
        argmax_label = active_labels[np.argmax(active_probs, axis=0)]  # (H, W)
    else:
        argmax_label = np.full(probs.shape[1:], other_mixed_value, dtype=np.uint8)

    # Pixels where no class passed threshold → fall back to argmax single class.
    # These are typically sub-threshold field edges, not genuinely unknown crops.
    no_class_mask = concat == ""
    concat[no_class_mask] = argmax_label[no_class_mask].astype(str)

    # Unrecognised mixed combinations (multiple classes fired but combo not in CLASSES_DICT)
    # → keep as other_crop/mixtures (200): could be an untrained crop species.
    unrecognised_mask = ~np.isin(concat, list(all_valid))
    concat[unrecognised_mask] = str(other_mixed_value)

    return concat.astype(np.uint8)


# ---------------------------------------------------------------------------
# TILE PROCESSING
# ---------------------------------------------------------------------------


def process_tile(
    tile_folder: Path,
    roi_geom: dict | None,
    roi_crs: str | None,
) -> None:
    """Postprocess all products in a single raw tile folder.

    Writes up to 3 output files:
      - cropland tile (2 bands: classification + probability)
      - croptype classification tile (1 band)
      - croptype probabilities tile (N bands)

    Parameters
    ----------
    tile_folder : Path
        Directory containing raw ``cropland*.tif`` and ``croptype*.tif``.
    roi_geom : dict | None
        GeoJSON-like ROI geometry. If None, ROI masking is skipped.
    roi_crs : str | None
        CRS string of ``roi_geom``.
    """
    tile_id = tile_folder.name

    cropland_files = sorted(tile_folder.glob("cropland*.tif"))
    croptype_files = sorted(tile_folder.glob("croptype*.tif"))

    if not cropland_files:
        logger.warning(f"[{tile_id}] No cropland*.tif found, skipping")
        return
    if not croptype_files:
        logger.warning(f"[{tile_id}] No croptype*.tif found, skipping")
        return

    raw_cropland = cropland_files[0]
    raw_croptype = croptype_files[0]

    cl_out = CROPLAND_OUTPUT_DIR / raw_cropland.name
    ct_out = CROPTYPE_OUTPUT_DIR / raw_croptype.name
    ct_probs_out = CROPTYPE_OUTPUT_DIR / raw_croptype.name.replace(
        "croptype_", "croptype-probs_", 1
    )

    if cl_out.exists() and ct_out.exists() and ct_probs_out.exists():
        logger.info(f"[{tile_id}] All outputs already exist, skipping")
        return

    # -------------------------------------------------------------------
    # CROPLAND PIPELINE
    # -------------------------------------------------------------------
    logger.info(f"[{tile_id}] Processing cropland")

    with rasterio.open(raw_cropland) as src:
        # Band layout: 1=classification (0/1/255), 2=prob_cropland (0-100), 3=prob_other (0-100)
        raw_cl = src.read()  # (3, H, W) uint8
        cl_profile = src.profile.copy()
        src_crs = src.crs
        transform = src.transform
        height, width = src.height, src.width

    # Track nodata from the probability band before smoothing
    nodata_mask = raw_cl[1] == NODATA  # (H, W)

    # Smooth probability bands only (bands 2 and 3: prob_cropland, prob_other)
    if DO_SMOOTH_CROPLAND:
        probs_cl = spatial_smoothing(raw_cl[1:3].copy(), nodata=NODATA)
        # probs_cl: float32 (2, H, W), values in [0, 1]
        clf = np.where(probs_cl[0] > probs_cl[1], 1, 0).astype(np.uint8)
        prob_cl = np.clip(probs_cl[0] * 100, 0, 100).astype(np.uint8)
    else:
        # Use raw probabilities directly (already 0-100 uint8)
        clf = np.where(raw_cl[1] > raw_cl[2], 1, 0).astype(np.uint8)
        prob_cl = raw_cl[1].copy()
    clf[nodata_mask] = NODATA
    prob_cl[nodata_mask] = NODATA

    cl_out_array = np.stack([clf, prob_cl], axis=0)  # (2, H, W)

    # Road mask — applied before ROI mask so roads inside ROI are still zeroed.
    # Also zero clf so road pixels are treated as non-cropland in the croptype pipeline.
    road_file = find_road_mask(tile_id, ROADS_DIR)
    if road_file is not None:
        cl_out_array = apply_road_mask(cl_out_array, road_file)
        clf = apply_road_mask(clf[np.newaxis], road_file)[0]
    else:
        logger.warning(f"[{tile_id}] No road mask found in {ROADS_DIR}")

    # ROI mask
    if roi_geom is not None and roi_crs is not None:
        try:
            cl_out_array = apply_roi_mask(
                cl_out_array, roi_geom, roi_crs, src_crs, transform, height, width
            )
        except Exception as e:
            logger.error(f"[{tile_id}] ROI mask failed for cropland: {e}")

    cl_profile.update(dtype="uint8", count=2, compress="deflate", nodata=NODATA)
    write_tile(
        cl_out_array,
        cl_profile,
        cl_out,
        do_reproject=DO_REPROJECT,
        target_epsg=TARGET_EPSG,
        band_descriptions=["Classification", "Probability"],
        tags={
            "nodata_value": str(NODATA),
            "description": "Cropland classification with probability",
        },
    )
    logger.info(f"[{tile_id}] Cropland → {cl_out}")

    # -------------------------------------------------------------------
    # CROPTYPE PIPELINE
    # -------------------------------------------------------------------
    logger.info(f"[{tile_id}] Processing croptype")

    with rasterio.open(raw_croptype) as src:
        all_band_names = [
            src.descriptions[i] or f"band_{i + 1}" for i in range(src.count)
        ]
        # Select only per-class probability bands (skip 'classification', 'probability')
        prob_band_indices = [
            i
            for i, name in enumerate(all_band_names)
            if name.startswith("probability_")
        ]
        if not prob_band_indices:
            logger.error(
                f"[{tile_id}] No 'probability_*' bands found in {raw_croptype.name}; "
                f"available bands: {all_band_names}"
            )
            return
        prob_class_names = [
            all_band_names[i].replace("probability_", "", 1) for i in prob_band_indices
        ]
        # rasterio.read() bands are 1-indexed
        raw_ct = src.read([i + 1 for i in prob_band_indices])  # (N_prob, H, W) uint8
        ct_profile = src.profile.copy()
        ct_src_crs = src.crs
        ct_transform = src.transform
        ct_height, ct_width = src.height, src.width

    logger.debug(f"[{tile_id}] Croptype prob bands: {prob_class_names}")
    n_bands = raw_ct.shape[0]

    # Track original nodata pixels
    ct_nodata_mask = np.any(raw_ct == NODATA, axis=0)  # (H, W)

    # Spatial smoothing (nodata pixels are zeroed inside, caller restores)
    if DO_SMOOTH_CROPTYPE:
        ct_probs_smoothed = spatial_smoothing(raw_ct.copy(), nodata=NODATA)
        # ct_probs_smoothed: float32 (N, H, W), values in [0, 1]
    else:
        # Normalise raw 0-100 values to [0, 1] float32 without spatial filtering
        ct_raw_f = raw_ct.astype("float32")
        ct_raw_f[:, ct_nodata_mask] = 0.0
        band_sum = ct_raw_f.sum(axis=0)
        band_sum[band_sum == 0] = 1.0
        ct_probs_smoothed = ct_raw_f / band_sum

    # Croptype classification from (optionally smoothed) probabilities
    clf_ct = get_croptype_prediction(
        ct_probs_smoothed, prob_class_names, CLASSES_DICT, THRESHOLDS, IGNORE_CLASSES
    )  # (H, W) uint8

    # Apply cropland mask using the in-memory (pre-reprojection) clf band.
    # Both tiles originate from the same folder so their grids must match.
    if clf.shape == (ct_height, ct_width):
        clf_ct[clf == 0] = NO_CROP_VALUE
        clf_ct[clf == NODATA] = NODATA
        # Zero out probs for non-cropland pixels before renormalizing
        ct_probs_smoothed[:, clf == 0] = 0.0
        ct_probs_smoothed[:, clf == NODATA] = 0.0
    else:
        logger.warning(
            f"[{tile_id}] Cropland and croptype grids differ "
            f"({clf.shape} vs ({ct_height}, {ct_width})); skipping cropland mask on croptype"
        )

    # Restore nodata from the original raw tile
    clf_ct[ct_nodata_mask] = NODATA

    # Convert smoothed probs to 0-100 uint8, normalized per pixel to sum to 100
    prob_sum = ct_probs_smoothed.sum(axis=0)  # (H, W)
    prob_sum[prob_sum == 0] = 1.0
    ct_probs_100 = np.clip(ct_probs_smoothed / prob_sum * 100, 0, 100).astype(np.uint8)

    # Restore nodata on probability output
    ct_probs_100[:, ct_nodata_mask] = NODATA
    ct_probs_100[:, clf == NODATA] = NODATA

    # ROI mask on both outputs
    if roi_geom is not None and roi_crs is not None:
        try:
            clf_ct = apply_roi_mask(
                clf_ct,
                roi_geom,
                roi_crs,
                ct_src_crs,
                ct_transform,
                ct_height,
                ct_width,
            )
            ct_probs_100 = apply_roi_mask(
                ct_probs_100,
                roi_geom,
                roi_crs,
                ct_src_crs,
                ct_transform,
                ct_height,
                ct_width,
            )
        except Exception as e:
            logger.error(f"[{tile_id}] ROI mask failed for croptype: {e}")

    # Build class metadata tags for classification output
    flat_classes = {
        **CLASSES_DICT["single_crop_classes"],
        **CLASSES_DICT["mixed_crops_classes"],
        NO_CROP_VALUE: "no_crop",
    }
    clf_tags = {f"class_{k}": v for k, v in flat_classes.items()}
    clf_tags.update(
        {
            "nodata_value": str(NODATA),
            "no_crop_value": str(NO_CROP_VALUE),
            "description": "Croptype classification",
            "thresholds": str(THRESHOLDS),
        }
    )

    ct_profile.update(dtype="uint8", count=1, compress="deflate", nodata=NODATA)
    write_tile(
        clf_ct[np.newaxis],  # (1, H, W)
        ct_profile,
        ct_out,
        do_reproject=DO_REPROJECT,
        target_epsg=TARGET_EPSG,
        band_descriptions=["Classification"],
        tags=clf_tags,
    )
    logger.info(f"[{tile_id}] Croptype classification → {ct_out}")

    ct_probs_profile = ct_profile.copy()
    ct_probs_profile.update(count=n_bands)
    write_tile(
        ct_probs_100,
        ct_probs_profile,
        ct_probs_out,
        do_reproject=DO_REPROJECT,
        target_epsg=TARGET_EPSG,
        band_descriptions=[f"probability_{name}" for name in prob_class_names],
        tags={
            "nodata_value": str(NODATA),
            "description": "Per-class croptype probabilities (0-100)",
        },
    )
    logger.info(f"[{tile_id}] Croptype probabilities → {ct_probs_out}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main() -> None:
    logger.info("Starting tile postprocessing")
    logger.info(f"  RAW_DIR      : {RAW_DIR}")
    logger.info(f"  CROPLAND_OUT : {CROPLAND_OUTPUT_DIR}")
    logger.info(f"  CROPTYPE_OUT : {CROPTYPE_OUTPUT_DIR}")
    logger.info(
        f"  ROI          : '{ROI_NAME}' ({ROI_GPKG.name}, field='{ROI_NAME_FIELD}')"
    )
    logger.info(f"  SMOOTH_CROPLAND : {DO_SMOOTH_CROPLAND}")
    logger.info(f"  SMOOTH_CROPTYPE : {DO_SMOOTH_CROPTYPE}")
    logger.info(
        f"  REPROJECT    : {DO_REPROJECT}"
        + (f" → EPSG:{TARGET_EPSG}" if DO_REPROJECT else "")
    )
    logger.info(f"  NUM_WORKERS  : {NUM_WORKERS}")

    CROPLAND_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CROPTYPE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load ROI geometry
    roi_geom, roi_crs = None, None
    try:
        roi_geom, roi_crs = load_roi_geometry(ROI_GPKG, ROI_NAME_FIELD, ROI_NAME)
    except Exception as e:
        logger.error(
            f"Failed to load ROI geometry: {e}\nProceeding WITHOUT ROI masking."
        )

    # Discover tile folders — each must contain at least one cropland*.tif
    tile_folders = sorted({f.parent for f in RAW_DIR.glob("*/cropland*.tif")})
    logger.info(f"Found {len(tile_folders)} tile folder(s) to process")

    if not tile_folders:
        logger.error(f"No tile folders found under {RAW_DIR}")
        return

    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_tile, folder, roi_geom, roi_crs): folder
            for folder in tile_folders
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Postprocessing tiles"
        ):
            folder = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"[{folder.name}] FAILED: {e}")
                errors.append(folder.name)

    if errors:
        logger.warning(f"{len(errors)} tile(s) failed: {errors}")
    else:
        logger.success(f"All {len(tile_folders)} tiles processed successfully")


if __name__ == "__main__":
    main()
