"""openEO UDF to compute Presto/Prometheo features."""

import logging
import os
import random
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np

# from prometheo.models.pooling import PoolingMethods
# from torch import nn
# from prometheo.datasets.worldcereal import (
#     extract_features_from_model,
#     generate_predictor,
# )
import pandas as pd
import xarray as xr
from openeo.metadata import CollectionMetadata
from openeo.udf import XarrayDataCube
from openeo.udf.udf_data import UdfData
from pyproj import Transformer
from scipy.ndimage import (
    convolve,
    zoom,
)
from shapely.geometry import Point
from shapely.ops import transform

sys.path.append("feature_deps")

import torch  # noqa: E402

logger = logging.getLogger(__name__)
_MODULE_CACHE_KEY = f"__model_cache_{__name__}"

# Constants
PROMETHEO_WHL_URL = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/dependencies/prometheo-0.0.3-py3-none-any.whl"

GFMAP_BAND_MAPPING = {
    "S2-L2A-B02": "B2",
    "S2-L2A-B03": "B3",
    "S2-L2A-B04": "B4",
    "S2-L2A-B05": "B5",
    "S2-L2A-B06": "B6",
    "S2-L2A-B07": "B7",
    "S2-L2A-B08": "B8",
    "S2-L2A-B8A": "B8A",
    "S2-L2A-B11": "B11",
    "S2-L2A-B12": "B12",
    "S1-SIGMA0-VH": "VH",
    "S1-SIGMA0-VV": "VV",
    "AGERA5-TMEAN": "temperature_2m",
    "AGERA5-PRECIP": "total_precipitation",
}

LAT_HARMONIZED_NAME = "GEO-LAT"
LON_HARMONIZED_NAME = "GEO-LON"
EPSG_HARMONIZED_NAME = "GEO-EPSG"

S1_BANDS = ["S1-SIGMA0-VV", "S1-SIGMA0-VH", "S1-SIGMA0-HV", "S1-SIGMA0-HH"]
NODATA_VALUE = 65535

NUM_THREADS = 2

sys.path.append("feature_deps")
_PROMETHEO_INSTALLED = False

# =============================================================================
# STANDALONE FUNCTIONS (Work in both apply_udf_data and apply_metadata contexts)
# =============================================================================
def get_model_cache():
    """Get or create module-specific cache."""
    if not hasattr(sys, _MODULE_CACHE_KEY):
        setattr(sys, _MODULE_CACHE_KEY, {})
    return getattr(sys, _MODULE_CACHE_KEY)

def _ensure_prometheo_dependencies():
    """Non-cached dependency check."""
    global _PROMETHEO_INSTALLED

    global prometheo, Presto, PretrainedPrestoWrapper, load_presto_weights, PoolingMethods

    if _PROMETHEO_INSTALLED:
        return

    try:
        # Try to import first
        import prometheo
        optimize_pytorch_cpu_performance(NUM_THREADS)
        _PROMETHEO_INSTALLED = True
        return
    except ImportError:
        pass

    # Installation required
    logger.info("Prometheo not available, installing...")
    _install_prometheo()
    optimize_pytorch_cpu_performance(NUM_THREADS)

    import prometheo
    from prometheo.models import Presto
    from prometheo.models.pooling import PoolingMethods
    from prometheo.models.presto.wrapper import (
        PretrainedPrestoWrapper,
        load_presto_weights,
    )
    _PROMETHEO_INSTALLED = True

def _install_prometheo():
    """Non-cached installation function."""
    import shutil
    import tempfile

    temp_dir = Path(tempfile.mkdtemp())
    try:
        # Download wheel
        wheel_path, _ = urllib.request.urlretrieve(PROMETHEO_WHL_URL)

        # Extract to temp directory
        with zipfile.ZipFile(wheel_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Add to Python path
        sys.path.append(str(temp_dir))
        logger.info(f"Prometheo installed to {temp_dir}")

    except Exception as e:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        logger.error(f"Failed to install prometheo: {e}")
        raise

def load_presto_weights_cached(presto_model_url: str, parameters: dict):
    """Manual caching for Presto weights with dependency check."""
    from prometheo.models.presto.wrapper import (
            PretrainedPrestoWrapper,
            load_presto_weights,
        )

    cache = get_model_cache()
    if presto_model_url in cache:
        logger.info(f"Presto model cache hit for {presto_model_url}")
        return cache[presto_model_url]

    # Ensure dependencies are available (not cached)
    _ensure_prometheo_dependencies()

    logger.info(f"Loading Presto weights from: {presto_model_url}")

    presto_model = PretrainedPrestoWrapper(num_outputs=parameters['num_outputs'], regression=False)
    result = load_presto_weights(presto_model, presto_model_url)

    cache[presto_model_url] = result
    return result

def optimize_pytorch_cpu_performance(num_threads):
    """CPU-specific optimizations for Prometheo."""

    # Thread configuration

    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads) #TODO test setting to 4 due to parallel slope cal ect
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)

    logger.info(f"PyTorch CPU:  using {num_threads} threads")

    # CPU-specific optimizations
    if hasattr(torch.backends, 'mkldnn'):
        torch.backends.mkldnn.enabled = True

    torch.set_grad_enabled(False)  # Disable gradients for inference

    return num_threads

def evaluate_resolution(inarr: xr.DataArray, epsg: int) -> int:
    """Helper function to get the resolution in meters for
    the input array.

    Parameters
    ----------
    inarr : xr.DataArray
        input array to determine resolution for.

    Returns
    -------
    int
        resolution in meters.
    """

    if epsg == 4326:
        logger.info(
            "Converting WGS84 coordinates to EPSG:3857 to determine resolution."
        )

        transformer = Transformer.from_crs(epsg, 3857, always_xy=True)
        points = [Point(x, y) for x, y in zip(inarr.x.values, inarr.y.values)]
        points = [transform(transformer.transform, point) for point in points]

        resolution = abs(points[1].x - points[0].x)

    else:
        resolution = abs(inarr.x[1].values - inarr.x[0].values)

    logger.info(f"Resolution for computing slope: {resolution}")

    return resolution


# =============================================================================
# CLASSES (Main logic for apply_udf_data)
# =============================================================================

class SlopeCalculator:
    """Handles slope computation from elevation data."""

    @staticmethod
    def compute(resolution: float, elevation_data: np.ndarray) -> np.ndarray:
        """Compute slope from elevation data."""
        dem_arr = SlopeCalculator._prepare_dem_array(elevation_data)
        dem_downsampled = SlopeCalculator._downsample_to_20m(dem_arr, resolution)
        slope = SlopeCalculator._compute_slope_gradient(dem_downsampled)
        result =  SlopeCalculator._upsample_to_original(slope, dem_arr.shape, resolution)
        return result


    @staticmethod
    def _prepare_dem_array(dem: np.ndarray) -> np.ndarray:
        """Prepare DEM array by handling NaNs and invalid values."""
        dem_arr = dem.astype(np.float32)
        dem_arr[dem_arr == NODATA_VALUE] = np.nan
        return SlopeCalculator._fill_nans(dem_arr)

    @staticmethod
    def _fill_nans(dem_arr: np.ndarray, max_iter: int = 2) -> np.ndarray:
        """Fill NaN values using rolling fill approach."""
        if max_iter == 0 or not np.any(np.isnan(dem_arr)):
            return dem_arr

        mask = np.isnan(dem_arr)
        roll_params = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(roll_params)

        for roll_param in roll_params:
            rolled = np.roll(dem_arr, roll_param, axis=(0, 1))
            dem_arr[mask] = rolled[mask]

        return SlopeCalculator._fill_nans(dem_arr, max_iter - 1)

    @staticmethod
    def _downsample_to_20m(dem_arr: np.ndarray, resolution: float) -> np.ndarray:
        """Downsample DEM to 20m resolution for slope computation."""
        factor = int(20 / resolution)
        if factor < 1 or factor % 2 != 0:
            raise ValueError(f"Unsupported resolution for slope: {resolution}")

        X, Y = dem_arr.shape
        pad_X, pad_Y = (factor - (X % factor)) % factor, (factor - (Y % factor)) % factor
        padded = np.pad(dem_arr, ((0, pad_X), (0, pad_Y)), mode="reflect")

        reshaped = padded.reshape((X + pad_X) // factor, factor, (Y + pad_Y) // factor, factor)
        return np.nanmean(reshaped, axis=(1, 3))

    @staticmethod
    def _compute_slope_gradient(dem: np.ndarray) -> np.ndarray:
        """Compute slope gradient using Sobel operators."""
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / (8.0 * 20)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / (8.0 * 20)

        dx = convolve(dem, kernel_x)
        dy = convolve(dem, kernel_y)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)

        return np.arctan(gradient_magnitude) * (180 / np.pi)

    @staticmethod
    def _upsample_to_original(slope: np.ndarray, original_shape: Tuple[int, int],
                            resolution: float) -> np.ndarray:
        """Upsample slope back to original resolution."""
        factor = int(20 / resolution)
        slope_upsampled = zoom(slope, zoom=factor, order=1)

        # Handle odd dimensions
        if original_shape[0] % 2 != 0:
            slope_upsampled = slope_upsampled[:-1, :]
        if original_shape[1] % 2 != 0:
            slope_upsampled = slope_upsampled[:, :-1]

        return slope_upsampled.astype(np.uint16)


class CoordinateTransformer:
    """Handles coordinate transformations and spatial operations."""

    @staticmethod
    def get_resolution(inarr: xr.DataArray, epsg: int) -> float:
        """Calculate resolution in meters."""
        if epsg == 4326:
            return CoordinateTransformer._get_wgs84_resolution(inarr)
        return abs(inarr.x[1].values - inarr.x[0].values)

    @staticmethod
    def _get_wgs84_resolution(inarr: xr.DataArray) -> float:
        """Convert WGS84 coordinates to meters for resolution calculation."""
        transformer = Transformer.from_crs(4326, 3857, always_xy=True)
        points = [Point(x, y) for x, y in zip(inarr.x.values, inarr.y.values)]
        points = [transform(transformer.transform, point) for point in points]
        return abs(points[1].x - points[0].x)

    @staticmethod
    def get_lat_lon_array(inarr: xr.DataArray, epsg: int) -> xr.DataArray:
        """Create latitude/longitude array from coordinates."""
        lon, lat = np.meshgrid(inarr.x.values, inarr.y.values)

        if epsg != 4326:
            transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
            lon, lat = transformer.transform(lon, lat)

        latlon = np.stack([lat, lon])
        return xr.DataArray(
            latlon,
            dims=["bands", "y", "x"],
            coords={"bands": [LAT_HARMONIZED_NAME, LON_HARMONIZED_NAME],
                   "y": inarr.y, "x": inarr.x}
        )


class DataPreprocessor:
    """Handles data preprocessing operations."""

    @staticmethod
    def rescale_s1_backscatter(arr: xr.DataArray) -> xr.DataArray:
        """Rescale Sentinel-1 backscatter from uint16 to dB values."""
        s1_bands_present = [b for b in S1_BANDS if b in arr.bands.values]
        if not s1_bands_present:
            return arr

        s1_data = arr.sel(bands=s1_bands_present).astype(np.float32)
        DataPreprocessor._validate_s1_data(s1_data.values)

        # Convert to power values then to dB
        power_values = 20.0 * np.log10(s1_data.values) - 83.0
        power_values = np.power(10, power_values / 10.0)
        power_values[~np.isfinite(power_values)] = np.nan

        db_values = 10.0 * np.log10(power_values)
        arr.loc[dict(bands=s1_bands_present)] = db_values

        return arr

    @staticmethod
    def _validate_s1_data(data: np.ndarray) -> None:
        """Validate S1 data meets preprocessing requirements."""
        if data.min() < 1 or data.max() > NODATA_VALUE:
            raise ValueError(
                "S1 data should be uint16 format with values 1-65535. "
                "Set 'rescale_s1' to False to disable scaling."
            )

    @staticmethod
    def log_array_statistics(arr: xr.DataArray) -> None:
        """Log comprehensive array statistics."""
        total_pixels = arr.size
        values = arr.values

        stats = {
            "NaN": np.isnan(values).sum(),
            "Zero": (values == 0).sum(),
            "NODATA": (values == NODATA_VALUE).sum()
        }

        logger.info(f"Bands: {', '.join(arr.bands.values)}")
        logger.info(f"Shape: {arr.shape}, Dtype: {arr.dtype}")

        for name, count in stats.items():
            percentage = (count / total_pixels) * 100
            logger.info(f"{name} pixels: {count} ({percentage:.2f}%)")

        # Log band means
        for band in arr.bands.values:
            mean_val = np.nanmean(arr.sel(bands=band).values)
            logger.info(f"Band '{band}' mean: {mean_val:.2f}")


class PrestoPredictor:
    """Handles Presto prediction pipeline."""

    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters

    def predict(self, inarr: xr.DataArray, epsg: int) -> xr.DataArray:
        """Extract Presto features from input array."""
        self._validate_inputs(inarr, epsg)
        inarr = self._preprocess_input(inarr)

        if "slope" not in inarr.bands:
            inarr = self._add_slope_band(inarr, epsg)

        return self._run_presto_inference(inarr, epsg)

    def _validate_inputs(self, inarr: xr.DataArray, epsg: int) -> None:
        """Validate input parameters and array"""
        if epsg is None:
            raise ValueError("EPSG code required for Presto prediction")
        # ONLY check top level - no nested lookup
        presto_model_url = self.parameters.get('presto_model_url')
        if not presto_model_url:
            logger.error(f"Missing presto_model_url. Available keys: {list(self.parameters.keys())}")
            raise ValueError('Missing required parameter "presto_model_url"')

        if len(inarr.t) != 12:
            raise ValueError(f"Presto requires 12 timesteps, got {len(inarr.t)}")

    def _preprocess_input(self, inarr: xr.DataArray) -> xr.DataArray:
        """Preprocess input array for Presto."""
        inarr = inarr.transpose("bands", "t", "x", "y")

        # Harmonize band names
        new_bands = [GFMAP_BAND_MAPPING.get(b.item(), b.item()) for b in inarr.bands]
        inarr = inarr.assign_coords(bands=new_bands)

        #TODO commented out to minimize loggingfs
        #DataPreprocessor.log_array_statistics(inarr)
        return inarr.fillna(NODATA_VALUE)

    def _add_slope_band(self, inarr: xr.DataArray, epsg: int) -> xr.DataArray:
        """Compute and add slope band to array."""
        logger.warning("Slope band not found, computing...")
        resolution = CoordinateTransformer.get_resolution(inarr.isel(t=0), epsg)
        elevation_data = inarr.sel(bands="elevation").isel(t=0).values

        slope_array = SlopeCalculator.compute(resolution, elevation_data)
        slope_da = xr.DataArray(
            slope_array[None, :, :],
            dims=("bands", "y", "x"),
            coords={"bands": ["slope"], "y": inarr.y, "x": inarr.x}
        ).expand_dims({"t": inarr.t}).astype("float32")

        return xr.concat([inarr.astype("float32"), slope_da], dim="bands")

    def _run_presto_inference(self, inarr: xr.DataArray, epsg: int) -> xr.DataArray:
        """Run Presto model inference with safe dependency handling."""
        # Dependencies are now handled by load_presto_weights_cached
        import gc

        from prometheo.datasets.worldcereal import (
            extract_features_from_model,
            generate_predictor,
        )
        from prometheo.models.pooling import PoolingMethods
        from torch import nn

        _ensure_prometheo_dependencies()

        presto_model_url = self.parameters['presto_model_url']

        model = load_presto_weights_cached(presto_model_url, self.parameters)

        # Import here to ensure dependencies are available
        pooling_method = PoolingMethods.TIME if self.parameters.get("temporal_prediction") else PoolingMethods.GLOBAL

        logger.info("Running presto inference")

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
            from einops import rearrange

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

        try:
            with torch.inference_mode():
                predictions = run_presto_model_inference(
                    inarr, model, epsg=epsg,
                    batch_size=self.parameters.get("batch_size", 256), #TODO optimize?
                    pooling_method=pooling_method
                )
            logger.info("Inference completed")
                # predictions['bands'] = parameters["classes"]
            predictions = predictions.transpose(
                "bands", "y", "x"
            )  # openEO expects yx order after the UDF
            return predictions

        finally:
            gc.collect()

# Apply the Prediction UDF
def apply_udf_data(udf_data: UdfData) -> UdfData:
    """This is the actual openeo UDF that will be executed by the backend."""

    cube = udf_data.datacube_list[0]
    parameters = udf_data.user_context.copy()

    epsg = udf_data.proj["EPSG"] if udf_data.proj else None

    arr = cube.get_array().transpose("bands", "t", "y", "x")

    if parameters.get("rescale_s1", True):
        arr =  DataPreprocessor.rescale_s1_backscatter(arr)
    presto_predictor = PrestoPredictor(parameters)

    arr = presto_predictor.predict(inarr=arr, epsg=epsg)

    cube = XarrayDataCube(arr)

    udf_data.datacube_list = [cube]

    return udf_data

# Change band names, since the target labels are parameterized in the UDF
def apply_metadata(metadata: CollectionMetadata, context: dict) -> CollectionMetadata:
    return metadata.rename_labels(dimension="bands", target=context["classes_list"])
