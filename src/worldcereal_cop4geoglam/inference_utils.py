import copy
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
from openeo.udf import XarrayDataCube
from openeo.udf.udf_data import UdfData
from prometheo.datasets.worldcereal import (
    extract_features_from_model,
    generate_predictor,
)
from prometheo.models.pooling import PoolingMethods
from torch import nn
from worldcereal.openeo.feature_extractor import (
    EPSG_HARMONIZED_NAME,
    rescale_s1_backscatter,
)

logger = logging.getLogger(__name__)

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

# Apply the Prediction UDF
def apply_udf_data(udf_data: UdfData) -> UdfData:
    """This is the actual openeo UDF that will be executed by the backend."""

    cube = udf_data.datacube_list[0]
    parameters = copy.deepcopy(udf_data.user_context)

    proj = udf_data.proj
    if proj is not None:
        proj = proj["EPSG"]

    parameters[EPSG_HARMONIZED_NAME] = proj

    arr = cube.get_array().transpose("bands", "t", "y", "x")

    epsg = parameters.pop(EPSG_HARMONIZED_NAME)
    logger.info(f"EPSG code determined for feature extraction: {epsg}")

    if parameters.get("rescale_s1", True):
        arr = rescale_s1_backscatter(arr)

    arr = predict_with_presto(inarr=arr, parameters=parameters, epsg=epsg)

    cube = XarrayDataCube(arr)

    udf_data.datacube_list = [cube]

    return udf_data
