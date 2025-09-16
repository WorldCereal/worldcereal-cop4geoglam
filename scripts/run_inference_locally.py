"""Run feature extraction + cropland & croptype mapping locally for preprocessed NetCDF patches.

This utility lets you quickly test new Presto (feature) and CatBoost / ONNX (classification)
models by running inference on local, already preprocessed NetCDF input patches. It
produces three products per input file:

1. Presto feature embeddings
2. Cropland (binary) classification
3. Croptype (multiclass) classification

Outputs are written as NetCDF files preserving original geospatial metadata.
"""

import logging
from pathlib import Path

import xarray as xr
from pyproj import CRS
from worldcereal.openeo.feature_extractor import extract_presto_embeddings
from worldcereal.openeo.inference import apply_inference
from worldcereal.parameters import CropLandParameters, CropTypeParameters

NODATAVALUE = 65535

logging.basicConfig(level=logging.INFO)


def reconstruct_dataset(arr: xr.DataArray, ds: xr.Dataset) -> xr.Dataset:
    """Reconstruct CRS attributes."""
    crs_attrs = ds["crs"].attrs
    x = ds.coords.get("x", None)
    y = ds.coords.get("y", None)

    new_ds = arr.assign_coords(bands=arr.bands.astype(str)).to_dataset(dim="bands")
    new_ds = new_ds.assign_coords(x=x)
    new_ds["x"].attrs.setdefault("standard_name", "projection_x_coordinate")
    new_ds["x"].attrs.setdefault("units", "m")

    new_ds = new_ds.assign_coords(y=y)
    new_ds["y"].attrs.setdefault("standard_name", "projection_y_coordinate")
    new_ds["y"].attrs.setdefault("units", "m")

    crs_name = "spatial_ref"
    new_ds[crs_name] = xr.DataArray(0, attrs=crs_attrs)

    for v in new_ds.data_vars:
        new_ds[v].attrs["grid_mapping"] = crs_name

    return new_ds


def run_full_mapping(
    arr: xr.DataArray,
    epsg: int = 32631,
    target_date: str | None = None,
    feature_model_url: str | None = None,
    cropland_classifier_model_url: str | None = None,
    croptype_classifier_model_url: str | None = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Run end-to-end mapping pipeline.

    Steps:
      1. Feature extraction (Presto embeddings)
      2. Cropland classification (using `CropLandParameters`)
      3. Croptype classification (using `CropTypeParameters`)

    Parameters
    ----------
    arr : xr.DataArray
        Input stacked array (bands dimension) derived from preprocessed NetCDF.
    epsg : int
        EPSG code for CRS expected by feature extractor.
    target_date : str | None
        Optional target date (currently unused but kept for API symmetry / future use).
    feature_model_url : str | None
        Optional override URL for the Presto encoder weights.
    cropland_classifier_model_url : str | None
        Optional override URL for cropland classifier model.
    croptype_classifier_model_url : str | None
        Optional override URL for croptype classifier model.

    Returns
    -------
    (features, cropland, croptype) : tuple[xr.DataArray, xr.DataArray, xr.DataArray]
        The embeddings and two classification outputs.
    """

    # --- Feature extraction (shared for both downstream tasks) ---
    print("Running feature extraction (Presto) ...")
    cropland_params = CropLandParameters()  # use cropland parameter spec for features
    feature_params = cropland_params.feature_parameters.model_dump()
    feature_params.update({"ignore_dependencies": True})
    if feature_model_url:
        feature_params["presto_model_url"] = feature_model_url

    features = extract_presto_embeddings(
        inarr=arr, parameters=feature_params, epsg=epsg
    )
    print(
        f"Features extracted: shape={features.shape}; bands={list(features.bands.values)}"
    )

    # --- Cropland classification ---
    print("Running cropland classification ...")
    cropland_classifier_params = cropland_params.classifier_parameters.model_dump()
    cropland_classifier_params.update({"ignore_dependencies": True})
    if cropland_classifier_model_url:
        cropland_classifier_params["classifier_url"] = cropland_classifier_model_url
    cropland = apply_inference(inarr=features, parameters=cropland_classifier_params)
    print(
        f"Cropland classification done: shape={cropland.shape}; bands={list(cropland.bands.values)}"
    )

    # --- Croptype classification ---
    print("Running croptype classification ...")
    croptype_params = CropTypeParameters()
    croptype_classifier_params = croptype_params.classifier_parameters.model_dump()
    croptype_classifier_params.update({"ignore_dependencies": True})
    if croptype_classifier_model_url:
        croptype_classifier_params["classifier_url"] = croptype_classifier_model_url
    croptype = apply_inference(inarr=features, parameters=croptype_classifier_params)
    print(
        f"Croptype classification done: shape={croptype.shape}; bands={list(croptype.bands.values)}"
    )

    return features, cropland, croptype


def main():
    """Main function to process all NetCDF files in the input directory."""
    # Manually define arguments here
    logging.info("Starting.")
    input_dir = Path(
        "/vitodata/worldcereal/data/COP4GEOGLAM/moldova/preprocessed_inputs_newimage"
    )
    output_dir = Path(
        "/vitodata/worldcereal/data/COP4GEOGLAM/moldova/production/local_debug"
    )
    target_date = None

    # Specify model URLs (override as needed). You can leave any as None to use defaults
    feature_model_url = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/Copernicus4Geoglam/moldova/presto-prometheo-cop4geoglam-run-with-AL-and-freezing-month-LANDCOVER10-augment%3DFalse-balance%3DTrue-timeexplicit%3DFalse-run%3D202509111120_encoder.pt"
    cropland_classifier_model_url = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/Copernicus4Geoglam/moldova/Presto_run%3D202509111120_DownstreamCatBoost_cropland_v120-MDA_balance%3DFalse.onnx"
    croptype_classifier_model_url = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/Copernicus4Geoglam/moldova/Presto_run%3D202509110852_DownstreamCatBoost_croptype_v120-MDA_balance%3DTrue.onnx"

    input_files = list(input_dir.rglob("*.nc"))

    if not input_files:
        print(f"No NetCDF files found in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for input_file in input_files:
        print(f"Processing file: {input_file}")

        try:
            ds = xr.open_dataset(input_file)
            epsg = CRS.from_wkt(
                ds.crs.attrs["spatial_ref"]
            ).to_epsg()  # Get the EPSG code
            arr = ds.drop_vars("crs").to_array(dim="bands")

            features, cropland, croptype = run_full_mapping(
                arr,
                target_date=target_date,
                feature_model_url=feature_model_url,
                cropland_classifier_model_url=cropland_classifier_model_url,
                croptype_classifier_model_url=croptype_classifier_model_url,
                epsg=epsg,
            )

            features_ds = reconstruct_dataset(arr=features, ds=ds)
            cropland_ds = reconstruct_dataset(arr=cropland, ds=ds)
            croptype_ds = reconstruct_dataset(arr=croptype, ds=ds)

            features_output_path = output_dir / f"{input_file.stem}_features.nc"
            cropland_output_path = output_dir / f"{input_file.stem}_cropland.nc"
            croptype_output_path = output_dir / f"{input_file.stem}_croptype.nc"

            features_ds.to_netcdf(features_output_path)
            cropland_ds.to_netcdf(cropland_output_path)
            croptype_ds.to_netcdf(croptype_output_path)

            print(f"Features saved to: {features_output_path}")
            print(f"Cropland classification saved to: {cropland_output_path}")
            print(f"Croptype classification saved to: {croptype_output_path}")

        except Exception as e:
            print(f"Error processing file {input_file}: {e}")
            raise


if __name__ == "__main__":
    main()
