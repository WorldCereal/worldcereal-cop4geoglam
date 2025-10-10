"""Run feature extraction + cropland & croptype mapping locally for preprocessed NetCDF patches.

This utility lets you quickly test new Presto (feature) and CatBoost / ONNX (classification)
models by running inference on local, already preprocessed NetCDF input patches. It
produces three products per input file:

1. Presto feature embeddings
2. Cropland (binary) classification
3. Croptype (multiclass) classification

Outputs are written as NetCDF files preserving original geospatial metadata.
"""

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

# from typing import Optional
import requests
import torch
import xarray as xr

# import onnxruntime as ort
# import numpy as np
from catboost import CatBoostRegressor
from einops import rearrange
from prometheo.datasets.worldcereal import (
    extract_features_from_model,
    generate_predictor,
)
from prometheo.models.pooling import PoolingMethods
from pyproj import CRS
from torch import nn
from worldcereal.openeo.feature_extractor import extract_presto_embeddings
from worldcereal.openeo.inference import apply_inference
from worldcereal.parameters import CropLandParameters, CropTypeParameters

NODATAVALUE = 65535

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def load_and_prepare_regressor_model(model_path: str, path_to_config: str='') -> tuple[CatBoostRegressor, list[str]]:
    """Load a CatBoost regressor model from a local file or URL, and its configuration.

    Parameters
    ----------
    model_path : str
        Path to the CatBoost model file (local or URL).
    path_to_config : Optional[str]
        Path to the model configuration JSON file. Required if loading from URL.

    Returns
    -------
    model : CatBoostRegressor
        The loaded CatBoost regressor model.
    label_names : list[str]
        List of label names (classes) if available in the configuration, otherwise an empty list.

    Raises
    ------
    ValueError
        If loading the model from a URL and path_to_config is not provided.
    """

    # Download model if URL
    if model_path.startswith("http://") or model_path.startswith("https://"):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            response = requests.get(model_path)
            tmp.write(response.content)
            tmp_path = tmp.name
            if path_to_config == '':
                raise ValueError("When loading model from URL, path_to_config must be provided.")
    else:
        path_to_config = str(list(Path(model_path).parent.glob("*config.json"))[0])
        tmp_path = model_path

    with open(path_to_config) as f:
        model_config = json.load(f)
        label_names = model_config.get("classes", [])
        if len(label_names) == 0:
            raise ValueError("Model configuration does not include 'classes' key or it is empty.")

    model = CatBoostRegressor()
    model.load_model(tmp_path)

    return model, label_names

def apply_inference_regressor(inarr: xr.DataArray, parameters: dict, ) -> xr.DataArray:
    """
    Apply a CatBoost regressor model to an input xarray DataArray.
    This function loads a regressor model from the specified URL or local path, applies it to the input data,
    and returns the regression results as a new DataArray with appropriate dimensions and coordinates.
    Parameters
    ----------
    inarr : xr.DataArray
        Input data array with dimensions ("bands", "x", "y").
    parameters : dict
        Dictionary containing model parameters. Must include "classifier_url" key.
    Returns
    -------
    xr.DataArray
        Output data array containing regression results, with dimensions ("bands", "y", "x").
    Raises
    ------
    ValueError
        If "classifier_url" is not present in parameters.
    """

    if "classifier_url" not in parameters:
        raise ValueError('Missing required parameter "classifier_url"')
    classifier_url = parameters.get("classifier_url")
    logger.info(f'Loading regressor model from "{classifier_url}"')
    if not isinstance(classifier_url, str):
        classifier_url = str(classifier_url)
    # shape and indices for output ("xy", "bands")
    x_coords, y_coords = inarr.x.values, inarr.y.values
    inarr = inarr.transpose("bands", "x", "y").stack(xy=["x", "y"]).transpose()  # Transpose to xy since CatBoost expects this

    model, output_labels = load_and_prepare_regressor_model(classifier_url)

    # Run catboost regression
    logger.info("Catboost regression with input shape: %s", inarr.shape)
    regression = model.predict(inarr.values)
    logger.info("Regression done with shape: %s", inarr.shape)
    regression = regression.reshape(len(x_coords), len(y_coords), (len(output_labels)))
    regression_da = xr.DataArray(
        np.moveaxis(regression, -1, 0),  # move bands to front
        dims=["bands", "x", "y"],
        coords={
            "bands": output_labels,
            "x": x_coords,
            "y": y_coords,
        },
    ).transpose("bands", "y", "x")  # openEO expects yx order after the UDF

    return regression_da

def run_presto_model_inference(
    inarr: Union[pd.DataFrame, xr.DataArray],
    model: nn.Module,  # Wrapper
    epsg: int = 4326,
    batch_size: int = 8192,
    pooling_method: PoolingMethods = PoolingMethods.GLOBAL,
) -> Union[np.ndarray, xr.DataArray]:
    """
    Runs a forward pass of the model on the input data.

    Parameters
    ----------
    inarr : xr.DataArray or pd.DataFrame
        Input data as xarray DataArray or pandas DataFrame.
    model : nn.Module
        A Prometheo compatible (wrapper) model.
    epsg : int
        EPSG code describing the coordinates.
    batch_size : int
        Batch size to be used for Presto inference.
    pooling_method : PoolingMethods
        Pooling method to be used for the model output.
        If PoolingMethods.GLOBAL, the output will be a single feature vector per pixel.
        If PoolingMethods.TIME, the output will retain the temporal dimension.

    Returns
    -------
    xr.DataArray or np.ndarray
        Model output as xarray DataArray or numpy ndarray.
    """

    predictor = generate_predictor(inarr, epsg)
    # fixing the pooling method to keep the function signature the same
    # as in presto-worldcereal but this could be an input argument too
    features = extract_features_from_model(model, predictor, batch_size, pooling_method)

    predictions = (
        torch.softmax(features, dim=-1)  # Softmax on the logits
        .cpu()
        .numpy()
    )

    # todo - return the output tensors to the right shape, either xarray or df
    if isinstance(inarr, pd.DataFrame):
        return predictions
    else:
        if pooling_method == PoolingMethods.TIME:
            # If pooling method is TIME, we need to keep the time dimension
            predictions = rearrange(
                predictions,
                "(y x) 1 1 t c -> x y t c",
                x=len(inarr.x),
                y=len(inarr.y),
                t=len(inarr.t),
            )
            predictions_da = xr.DataArray(
                predictions,
                dims=["x", "y", "t", "bands"],
                coords={"x": inarr.x, "y": inarr.y, "t": inarr.t},
            )
        else:
            # If pooling method is GLOBAL, we collapse the time dimension
            predictions = rearrange(
                predictions,
                "(y x) 1 1 1 c -> x y c",
                x=len(inarr.x),
                y=len(inarr.y),
            )
            predictions_da = xr.DataArray(
                predictions, dims=["x", "y", "bands"], coords={"x": inarr.x, "y": inarr.y}
            )
        return predictions_da

def predict_with_presto(
    inarr: xr.DataArray, parameters: dict, epsg: int
) -> xr.DataArray:
    """Executes the feature extraction process on the input array."""
    from worldcereal.openeo.feature_extractor import (
        GFMAP_BAND_MAPPING,
        PROMETHEO_WHL_URL,
        compute_slope,
        evaluate_resolution,
        unpack_prometheo_wheel,
    )
    if epsg is None:
        raise ValueError(
            "EPSG code is required for Presto feature extraction, but was "
            "not correctly initialized."
        )
    if "presto_model_url" not in parameters:
        raise ValueError('Missing required parameter "presto_model_url"')

    presto_model_url = parameters.get("presto_model_url")
    logger.info(f'Loading Presto model from "{presto_model_url}"')
    prometheo_wheel_url = parameters.get("prometheo_wheel_url", PROMETHEO_WHL_URL)
    logger.info(f'Loading Prometheo wheel from "{prometheo_wheel_url}"')

    ignore_dependencies = parameters.get("ignore_dependencies", False)
    if ignore_dependencies:
        logger.info(
            "`ignore_dependencies` flag is set to True. Make sure that "
            "Presto and its dependencies are available on the runtime "
            "environment"
        )

    # The below is required to avoid flipping of the result
    # when running on OpenEO backend!
    inarr = inarr.transpose(
        "bands", "t", "x", "y"
    )  # Presto/Prometheo expects xy dimension order

    # Change the band names
    new_band_names = [GFMAP_BAND_MAPPING.get(b.item(), b.item()) for b in inarr.bands]
    inarr = inarr.assign_coords(bands=new_band_names)

    # Log pixel statistics
    total_pixels = inarr.size
    num_nan_pixels = np.isnan(inarr.values).sum()
    num_zero_pixels = (inarr.values == 0).sum()
    num_nodatavalue_pixels = (inarr.values == 65535).sum()
    logger.info("Band names: " + ", ".join(inarr.bands.values))
    logger.debug(
        f"Array dtype: {inarr.dtype}, "
        f"Array size: {inarr.shape}, total pixels: {total_pixels}, "
        f"Pixel statistics: NaN pixels = {num_nan_pixels} "
        f"({num_nan_pixels / total_pixels * 100:.2f}%), "
        f"0 pixels = {num_zero_pixels} "
        f"({num_zero_pixels / total_pixels * 100:.2f}%), "
        f"NODATAVALUE pixels = {num_nodatavalue_pixels} "
        f"({num_nodatavalue_pixels / total_pixels * 100:.2f}%)"
    )

    # Log mean value (ignoring NaNs) per band
    for band in inarr.bands.values:
        band_data = inarr.sel(bands=band).values
        mean_value = np.nanmean(band_data)
        logger.debug(f"Band '{band}': Mean value (ignoring NaNs) = {mean_value:.2f}")

    # Handle NaN values in Presto compatible way
    inarr = inarr.fillna(65535)

    if not ignore_dependencies:
        # Unzip the Presto dependencies on the backend
        logger.info("Unpacking prometheo wheel")
        deps_dir = unpack_prometheo_wheel(prometheo_wheel_url)

        logger.info("Appending dependencies")
        sys.path.append(str(deps_dir))

    if "slope" not in inarr.bands:
        # If 'slope' is not present we need to compute it here
        logger.warning("`slope` not found in input array. Computing ...")
        resolution = evaluate_resolution(inarr.isel(t=0), epsg)
        slope = compute_slope(inarr.isel(t=0), resolution)
        slope = slope.expand_dims({"t": inarr.t}, axis=0).astype("float32")

        inarr = xr.concat([inarr.astype("float32"), slope], dim="bands")

    batch_size = parameters.get("batch_size", 256)
    logger.info(
        (
            f"Extracting Presto features with batch size {batch_size}, "
        )
    )

    # TODO: compile_presto not used for now?
    # compile_presto = parameters.get("compile_presto", False)
    # self.logger.info(f"Compile presto: {compile_presto}")

    logger.info("Loading Presto model for inference")

    # TODO: try to take run_model_inference from worldcereal
    from prometheo.models.pooling import PoolingMethods
    from prometheo.models.presto.wrapper import (
        PretrainedPrestoWrapper,
        load_presto_weights,
    )

    presto_model = PretrainedPrestoWrapper(num_outputs=parameters['num_outputs'], regression=False)
    presto_model = load_presto_weights(presto_model, presto_model_url)

    logger.info("Extracting presto features")
    # Check if we have the expected 12 timesteps
    if len(inarr.t) != 12:
        raise ValueError(f"Can only run Presto on 12 timesteps, got: {len(inarr.t)}")

    pooling_method = PoolingMethods.GLOBAL
    logger.info(f"Using pooling method: {pooling_method}")

    predictions = run_presto_model_inference(
        inarr,
        presto_model,
        epsg=epsg,
        batch_size=batch_size,
        pooling_method=pooling_method,
    )

    predictions['bands'] = parameters["classes"]
    predictions = predictions.transpose(
        "bands", "y", "x"
    )  # openEO expects yx order after the UDF

    return predictions

def run_full_mapping(
    arr: xr.DataArray,
    epsg: int = 32631,
    target_date: str | None = None,
    cropland_feature_model_url: str | None = None,
    croptype_feature_model_url: str | None = None,
    cropland_classifier_model_url: str | None = None,
    croptype_classifier_model_url: str | None = None,
    classes_list: list[str] | None = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
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
    print("Running cropland feature extraction (Presto) ...")
    cropland_params = CropLandParameters()  # use cropland parameter spec for features
    cropland_feature_params = cropland_params.feature_parameters.model_dump()
    cropland_feature_params.update({"ignore_dependencies": True})
    if cropland_feature_model_url:
        cropland_feature_params["presto_model_url"] = cropland_feature_model_url

    cropland_features = extract_presto_embeddings(
        inarr=arr, parameters=cropland_feature_params, epsg=epsg
    )
    print(
        f"Features extracted: shape={cropland_features.shape}; bands={list(cropland_features.bands.values)}"
    )

    # --- Cropland classification ---
    print("Running cropland classification ...")
    cropland_classifier_params = cropland_params.classifier_parameters.model_dump()
    cropland_classifier_params.update({"ignore_dependencies": True})
    if cropland_classifier_model_url:
        cropland_classifier_params["classifier_url"] = cropland_classifier_model_url
    cropland = apply_inference(inarr=cropland_features, parameters=cropland_classifier_params)
    print(
        f"Cropland classification done: shape={cropland.shape}; bands={list(cropland.bands.values)}"
    )

    # --- Feature extraction (shared for both downstream tasks) ---
    print("Running croptype feature extraction (Presto) ...")
    croptype_params = CropTypeParameters()  # use croptype parameter spec for features
    croptype_feature_params = croptype_params.feature_parameters.model_dump()
    croptype_feature_params.update({"ignore_dependencies": True})
    if croptype_feature_model_url:
        croptype_feature_params["presto_model_url"] = croptype_feature_model_url

    if classes_list is None:
        croptype_features = extract_presto_embeddings(
            inarr=arr, parameters=croptype_feature_params, epsg=epsg
        )
        print(
            f"Features extracted: shape={croptype_features.shape}; bands={list(croptype_features.bands.values)}"
        )
        # --- Croptype classification ---
        croptype_classifier_params = croptype_params.classifier_parameters.model_dump()
        croptype_classifier_params.update({"ignore_dependencies": True})
        if croptype_classifier_model_url:
            croptype_classifier_params["classifier_url"] = croptype_classifier_model_url
        croptype = apply_inference_regressor(inarr=croptype_features, parameters=croptype_classifier_params)
        print(
            f"Croptype classification done: shape={croptype.shape}; bands={list(croptype.bands.values)}"
        )
        return cropland_features, croptype_features, cropland, croptype

    # classification using presto
    else:
        croptype_feature_params["num_outputs"] = len(classes_list)
        croptype_feature_params["classes"] = classes_list
        croptype_predictions = predict_with_presto(
            inarr=arr, parameters=croptype_feature_params, epsg=epsg
        )
        print(
            f"Croptype classification done: shape={croptype_predictions.shape}; bands={list(croptype_predictions.bands.values)}"
        )
        return cropland_features, croptype_predictions, cropland, croptype_predictions


def main():
    """Main function to process all NetCDF files in the input directory."""
    # Manually define arguments here
    logging.info("Starting.")
    country = "mozambique"
    exp_tag = "local_with_fuzzy_test_10samples"
    input_dir = Path(
        # f"/vitodata/worldcereal/data/COP4GEOGLAM/{country}/PSU_preprocessed_inputs"
        f"/projects/worldcereal/COP4GEOGLAM/{country}/PSU_preprocessed_inputs"
    )
    output_dir = Path(
        # f"/vitodata/worldcereal/data/COP4GEOGLAM/{country}/production/{exp_tag}"
        f"/projects/worldcereal/COP4GEOGLAM/{country}/production/{exp_tag}"
    )
    target_date = None
    predict_with_presto = False

    model_suffix = "_encoder" if not predict_with_presto else ""
    # classes list is only used when predicting with presto
    classes_list = [
        "maize",
        "soybean",
        "sesame",
        "sweet_potato",
        "cassava",
        "pigeon pea",
        "rice",
        "other"
    ]
    # Specify model URLs (override as needed). You can leave any as None to use defaults
    cropland_feature_model_url = "/projects/worldcereal/COP4GEOGLAM/mozambique/models/presto/cropland/presto-prometheo-cop4geoglam-exp_points_no_agroforestry-month-LANDCOVER10-augment=False-balance=True-timeexplicit=False-freezing=True-run=202509261104_encoder.pt"
    croptype_feature_model_url = f"/projects/worldcereal/COP4GEOGLAM/mozambique/models/presto/v3/presto-prometheo-cop4geoglam-test-fuzzy-month-CROPTYPE_Mozambique_fuzzy-augment=False-balance=True-timeexplicit=False-freezing=True-run=202510081032/presto-prometheo-cop4geoglam-test-fuzzy-month-CROPTYPE_Mozambique_fuzzy-augment=False-balance=True-timeexplicit=False-freezing=True-run=202510081032{model_suffix}.pt" #_encoder
    cropland_classifier_model_url = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/Copernicus4Geoglam/mozambique/catboost/Presto_run%3D202509261104_DownstreamCatBoost_cropland_v0_balance%3DTrue.onnx"
    croptype_classifier_model_url = "/projects/worldcereal/COP4GEOGLAM/mozambique/models/catboost/v3/croptype/Presto_run=202510081032_DownstreamCatBoost_croptype_v3_balance=True/Presto_run=202510081032_DownstreamCatBoost_croptype_v3_balance=True.cbm"

    input_files = list(input_dir.rglob("*.nc"))[:5]

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

            cropland_features, croptype_features, cropland, croptype = run_full_mapping(
                arr,
                target_date=target_date,
                cropland_feature_model_url=cropland_feature_model_url,
                croptype_feature_model_url=croptype_feature_model_url,
                cropland_classifier_model_url=cropland_classifier_model_url,
                croptype_classifier_model_url=croptype_classifier_model_url,
                epsg=epsg,
                classes_list=classes_list if predict_with_presto else None,
            )

            # Apply cropland mask to croptype classification: set croptype to NODATAVALUE where cropland==0
            classification_band_idx = int((cropland.bands == "classification").argmax().item())
            no_crop_mask = cropland[classification_band_idx, :, :]
            croptype_masked = croptype.copy()
            croptype_masked = croptype.where(no_crop_mask != 0, NODATAVALUE)

            # cropland_features
            cropland_features_ds = reconstruct_dataset(arr=cropland_features, ds=ds)
            cropland_features_output_path = output_dir / f"{input_file.stem}_cropland_features.nc"
            cropland_features_ds.to_netcdf(cropland_features_output_path)
            print(f"Cropland Features saved to: {cropland_features_output_path}")

            # cropland
            cropland_ds = reconstruct_dataset(arr=cropland, ds=ds)
            cropland_output_path = output_dir / f"{input_file.stem}_cropland.nc"
            cropland_ds.to_netcdf(cropland_output_path)
            print(f"Cropland classification saved to: {cropland_output_path}")

            # croptype features if predicting with regressor
            if not predict_with_presto:
                croptype_features_ds = reconstruct_dataset(arr=croptype_features, ds=ds)
                croptype_features_output_path = output_dir / f"{input_file.stem}_croptype_features.nc"
                croptype_features_ds.to_netcdf(croptype_features_output_path)
                print(f"Croptype Features saved to: {croptype_features_output_path}")

            croptype_ds = reconstruct_dataset(arr=croptype, ds=ds)
            croptype_suffix = "_presto" if predict_with_presto else ""
            croptype_output_path = output_dir / f"{input_file.stem}_croptype{croptype_suffix}.nc"
            croptype_ds.to_netcdf(croptype_output_path)
            print(f"Croptype classification saved to: {croptype_output_path}")

            croptype_masked_ds = reconstruct_dataset(arr=croptype_masked, ds=ds)
            croptype_masked_output_path = output_dir / f"{input_file.stem}_croptype_masked.nc"
            croptype_masked_ds.to_netcdf(croptype_masked_output_path)
            print(f"Croptype masked classification saved to: {croptype_masked_output_path}")

        except Exception as e:
            print(f"Error processing file {input_file}: {e}")
            raise

    inference_settings = {
        "cropland_feature_model": Path(cropland_feature_model_url).name if cropland_feature_model_url else None,
        "croptype_feature_model": Path(croptype_feature_model_url).name if croptype_feature_model_url else None,
        "cropland_classifier_model": Path(cropland_classifier_model_url).name if cropland_classifier_model_url else None,
        "croptype_classifier_model": Path(croptype_classifier_model_url).name if croptype_classifier_model_url else None,
    }

    report_path = output_dir / "inference_settings.json"
    with open(report_path, "w") as f:
        json.dump(inference_settings, f, indent=2)
    print(f"inference settings saved to: {report_path}")


if __name__ == "__main__":
    main()
