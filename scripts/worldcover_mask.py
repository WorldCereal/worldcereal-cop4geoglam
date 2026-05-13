import glob
import os
import subprocess

import numpy as np
import rasterio
from rasterio.coords import BoundingBox
from rasterio.errors import RasterioIOError
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import bounds as window_bounds
from tqdm import tqdm

worldcover_folder = "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique_pm/worldcover"
croptype_path = "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique_pm/production/v5_PM/postprocessed/final/croptype.tif"
cropland_path = "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique_pm/production/v5_PM/postprocessed/final/cropland.tif"
probability_path = "/vitodata/worldcereal/data/COP4GEOGLAM/mozambique_pm/production/v5_PM/postprocessed/final/croptype-probs.tif"

values_to_mask = [50, 80]  # 50 = built-up, 80 = water

landcover_paths = glob.glob(os.path.join(worldcover_folder, "*", "*Map.tif"))
out_mask_path = os.path.join(worldcover_folder, "worldcover_mask_cog.tif")

landcover_classes_to_remove = {50, 60, 70, 80, 100}

def intersects(a: BoundingBox, b: BoundingBox) -> bool:
    return not (
        a.right <= b.left or
        a.left >= b.right or
        a.top <= b.bottom or
        a.bottom >= b.top
    )

def copy_colormap_if_present(src, dst):
    """
    Copy band-1 colormap from src to dst if it exists.
    Works for paletted categorical rasters such as croptype maps.
    """
    try:
        cmap = src.colormap(1)
        if cmap:
            dst.write_colormap(1, cmap)
    except ValueError:
        # No colormap present
        pass

def translate_to_cog(
    src_path,
    dst_path,
    overwrite=True,
    compress="DEFLATE",
    blocksize=1024,
    resampling="NEAREST",
    bigtiff="IF_SAFER",
):
    """
    Convert a GeoTIFF to a proper GDAL COG using gdal_translate.
    Color tables, georeferencing, nodata, and metadata are normally preserved
    by gdal_translate from the source raster.
    """
    if os.path.exists(dst_path):
        if overwrite:
            os.remove(dst_path)
        else:
            raise FileExistsError(f"Output already exists: {dst_path}")

    cmd = [
        "gdal_translate",
        src_path,
        dst_path,
        "-of", "COG",
        "-co", f"COMPRESS={compress}",
        "-co", f"BLOCKSIZE={blocksize}",
        "-co", f"RESAMPLING={resampling}",
        "-co", f"BIGTIFF={bigtiff}",
    ]

    subprocess.run(cmd, check=True)

def createMask(
    template_path,
    landcover_paths,
    out_mask_path,
    landcover_classes_to_remove,
    overwrite=False,
):
    if os.path.exists(out_mask_path) and not overwrite:
        print(f"Mask already exists, skipping: {out_mask_path}")
        return

    out_dir = os.path.dirname(out_mask_path)
    os.makedirs(out_dir, exist_ok=True)

    tmp_mask_path = out_mask_path.replace(".tif", "_tmp.tif")

    if os.path.exists(tmp_mask_path):
        os.remove(tmp_mask_path)

    with rasterio.open(template_path) as template:
        if template.crs is None:
            raise ValueError(f"Template raster has no CRS: {template_path}")

        profile = template.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="uint8",
            count=1,
            nodata=0,
            compress="deflate",
            predictor=1,
            tiled=True,
            blockxsize=1024,
            blockysize=1024,
            BIGTIFF="IF_SAFER",
            interleave="band",
        )

        landcover_sources = []

        for path in landcover_paths:
            try:
                src = rasterio.open(path)

                if src.crs is None:
                    print(f"Skipping {path}: missing CRS")
                    src.close()
                    continue

                if src.crs != template.crs:
                    left, bottom, right, top = transform_bounds(
                        src.crs,
                        template.crs,
                        src.bounds.left,
                        src.bounds.bottom,
                        src.bounds.right,
                        src.bounds.top,
                        densify_pts=21,
                    )
                    bounds_in_template_crs = BoundingBox(left, bottom, right, top)
                else:
                    bounds_in_template_crs = src.bounds

                landcover_sources.append(
                    {
                        "path": path,
                        "src": src,
                        "bounds_in_template_crs": bounds_in_template_crs,
                    }
                )

            except RasterioIOError:
                print(f"Could not open {path}")

        with rasterio.open(tmp_mask_path, "w", **profile) as dst:
            # Optional mask colormap:
            # 0 = removed, 1 = kept
            dst.write_colormap(
                1,
                {
                    0: (0, 0, 0, 255),
                    1: (255, 255, 255, 255),
                },
            )

            bar = tqdm(
                total=template.width * template.height,
                desc="Creating worldcover mask",
                unit="px",
            )

            for _, window in template.block_windows(1):
                bar.update(window.width * window.height)

                dst_transform = template.window_transform(window)
                dst_bounds = BoundingBox(*window_bounds(window, template.transform))

                remove_window = np.zeros((window.height, window.width), dtype=bool)

                for lc_info in landcover_sources:
                    lc = lc_info["src"]
                    lc_bounds = lc_info["bounds_in_template_crs"]

                    # CRS-safe intersection:
                    # dst_bounds and lc_bounds are both in template.crs
                    if not intersects(dst_bounds, lc_bounds):
                        continue

                    lc_on_template = np.full(
                        (window.height, window.width),
                        fill_value=0,
                        dtype=lc.dtypes[0],
                    )

                    reproject(
                        source=rasterio.band(lc, 1),
                        destination=lc_on_template,
                        src_transform=lc.transform,
                        src_crs=lc.crs,
                        src_nodata=lc.nodata,
                        dst_transform=dst_transform,
                        dst_crs=template.crs,
                        dst_nodata=0,
                        resampling=Resampling.nearest,
                    )

                    remove_window |= np.isin(
                        lc_on_template,
                        list(landcover_classes_to_remove),
                    )

                # 1 = keep croptype pixel
                # 0 = remove croptype pixel
                mask_window = (~remove_window).astype(np.uint8)

                dst.write(mask_window, 1, window=window)

            bar.close()

        for lc_info in landcover_sources:
            lc_info["src"].close()

    translate_to_cog(
        src_path=tmp_mask_path,
        dst_path=out_mask_path,
        overwrite=True,
        compress="DEFLATE",
        blocksize=1024,
        resampling="NEAREST",
        bigtiff="IF_SAFER",
    )

    os.remove(tmp_mask_path)

    print(f"Wrote GDAL COG mask to: {out_mask_path}")

def applyMask(mask_file, croptype_file, out_file, fill_value = 254,overwrite=False):
    """
    Apply worldcover mask to croptype.

    Where mask == 0, croptype is set to 254.
    The output is converted to a proper GDAL COG.
    The original croptype colormap is copied to the temporary raster before
    COG conversion, so gdal_translate preserves it.
    """
    if os.path.exists(out_file) and not overwrite:
        print(f"Masked croptype already exists, skipping: {out_file}")
        return

    out_dir = os.path.dirname(out_file)
    os.makedirs(out_dir, exist_ok=True)

    tmp_out_file = out_file.replace(".tif", "_tmp.tif")

    if os.path.exists(tmp_out_file):
        os.remove(tmp_out_file)

    with rasterio.open(croptype_file) as croptype_src, rasterio.open(mask_file) as mask_src:
        if croptype_src.crs != mask_src.crs:
            raise ValueError(
                "Croptype and mask CRS differ. "
                f"croptype CRS={croptype_src.crs}, mask CRS={mask_src.crs}"
            )

        if croptype_src.transform != mask_src.transform:
            raise ValueError("Croptype and mask transforms differ.")

        if croptype_src.width != mask_src.width or croptype_src.height != mask_src.height:
            raise ValueError("Croptype and mask dimensions differ.")

        profile = croptype_src.profile.copy()
        profile.update(
            driver="GTiff",
            tiled=True,
            blockxsize=2048,
            blockysize=2048,
            compress="deflate",
            predictor=1,
            BIGTIFF="IF_SAFER",
            interleave="band",
        )

        with rasterio.open(tmp_out_file, "w", **profile) as dst:
            # Preserve colormap from original croptype
            copy_colormap_if_present(croptype_src, dst)

            # Preserve useful dataset/band tags
            dst.update_tags(**croptype_src.tags())
            dst.update_tags(1, **croptype_src.tags(1))

            bar = tqdm(
                total=croptype_src.width * croptype_src.height,
                desc="Applying mask to croptype",
                unit="px",
            )

            for _, window in croptype_src.block_windows(1):
                bar.update(window.width * window.height)

                croptype_data = croptype_src.read(1, window=window)
                mask_data = mask_src.read(1, window=window)

                nodata_mask = croptype_data == croptype_src.nodata

                masked_croptype = np.where(
                    mask_data == 0,
                    fill_value,
                    croptype_data,
                ).astype(croptype_src.dtypes[0])

                # Ensure nodata pixels remain nodata after masking
                masked_croptype[nodata_mask] = croptype_src.nodata

                dst.write(masked_croptype, 1, window=window)

            bar.close()

    translate_to_cog(
        src_path=tmp_out_file,
        dst_path=out_file,
        overwrite=True,
        compress="DEFLATE",
        blocksize=1024,
        resampling="NEAREST",
        bigtiff="IF_SAFER",
    )

    os.remove(tmp_out_file)

    print(f"Wrote GDAL COG masked croptype to: {out_file}")

def applyMaskMultiBand(
    mask_file,
    raster_file,
    out_file,
    fill_value=0,
    overwrite=True,
    preserve_nodata=True,
):
    """
    Apply a single-band mask to a raster with one or more bands.

    Where mask == 0, each raster band is set to fill_value.
    Where mask == 1, original raster values are retained.

    The output is converted to a proper GDAL COG.

    Preserves:
      - dataset tags
      - band tags
      - band descriptions / names
      - nodata value
      - color interpretation
      - colormap for paletted single-band rasters when present
    """

    if os.path.exists(out_file) and not overwrite:
        print(f"Masked raster already exists, skipping: {out_file}")
        return

    out_dir = os.path.dirname(out_file)
    os.makedirs(out_dir, exist_ok=True)

    tmp_out_file = out_file.replace(".tif", "_tmp.tif")

    if os.path.exists(tmp_out_file):
        os.remove(tmp_out_file)

    with rasterio.open(raster_file) as raster_src, rasterio.open(mask_file) as mask_src:
        if raster_src.crs != mask_src.crs:
            raise ValueError(
                "Raster and mask CRS differ. "
                f"raster CRS={raster_src.crs}, mask CRS={mask_src.crs}"
            )

        if raster_src.transform != mask_src.transform:
            raise ValueError("Raster and mask transforms differ.")

        if raster_src.width != mask_src.width or raster_src.height != mask_src.height:
            raise ValueError("Raster and mask dimensions differ.")

        if mask_src.count != 1:
            raise ValueError(f"Mask should have exactly 1 band, got {mask_src.count}")

        profile = raster_src.profile.copy()
        profile.update(
            driver="GTiff",
            tiled=True,
            blockxsize=2048,
            blockysize=2048,
            compress="deflate",
            predictor=1,
            BIGTIFF="IF_SAFER",
            interleave="band",
        )

        with rasterio.open(tmp_out_file, "w", **profile) as dst:
            # Preserve dataset-level tags
            dst.update_tags(**raster_src.tags())

            # Preserve color interpretation, e.g. red/green/blue/alpha or gray/palette
            try:
                dst.colorinterp = raster_src.colorinterp
            except Exception:
                pass

            # Preserve band descriptions/names and per-band tags
            for band_idx in range(1, raster_src.count + 1):
                description = raster_src.descriptions[band_idx - 1]
                if description:
                    dst.set_band_description(band_idx, description)

                band_tags = raster_src.tags(band_idx)
                if band_tags:
                    dst.update_tags(band_idx, **band_tags)

            # Preserve colormap where applicable.
            # Usually only valid for single-band paletted rasters.
            if raster_src.count == 1:
                copy_colormap_if_present(raster_src, dst)

            bar = tqdm(
                total=raster_src.width * raster_src.height,
                desc=f"Applying mask to {raster_src.count}-band raster",
                unit="px",
            )

            for _, window in raster_src.block_windows(1):
                bar.update(window.width * window.height)

                mask_data = mask_src.read(1, window=window)

                # Read all bands for this window: shape = (bands, rows, cols)
                raster_data = raster_src.read(window=window)

                masked_data = raster_data.copy()

                for band_idx in range(raster_src.count):
                    band_data = raster_data[band_idx]

                    masked_band = np.where(
                        mask_data == 0,
                        fill_value,
                        band_data,
                    ).astype(raster_src.dtypes[band_idx])

                    if preserve_nodata:
                        band_nodata = raster_src.nodatavals[band_idx]

                        if band_nodata is not None:
                            nodata_mask = band_data == band_nodata
                            masked_band[nodata_mask] = band_nodata

                    masked_data[band_idx] = masked_band

                dst.write(masked_data, window=window)

            bar.close()

    translate_to_cog(
        src_path=tmp_out_file,
        dst_path=out_file,
        overwrite=True,
        compress="DEFLATE",
        blocksize=1024,
        resampling="NEAREST",
        bigtiff="IF_SAFER",
    )

    os.remove(tmp_out_file)

    print(f"Wrote GDAL COG masked raster to: {out_file}")

if __name__ == "__main__":
    createMask(
        template_path=croptype_path,
        landcover_paths=landcover_paths,
        out_mask_path=out_mask_path,
        landcover_classes_to_remove=landcover_classes_to_remove,
    )

    masked_croptype_path = croptype_path.replace(".tif", "_masked.tif")
    masked_cropland_path = cropland_path.replace(".tif", "_masked.tif")
    masked_probability_path = probability_path.replace(".tif", "_masked.tif")

    applyMask(
        mask_file=out_mask_path,
        croptype_file=croptype_path,
        out_file=masked_croptype_path,
    )

    applyMask(
        mask_file=out_mask_path,
        croptype_file=cropland_path,
        out_file=masked_cropland_path,
        fill_value = 0,
    )

    applyMaskMultiBand(
        mask_file=out_mask_path,
        raster_file=probability_path,
        out_file=masked_probability_path,
        fill_value = 0,
    )
