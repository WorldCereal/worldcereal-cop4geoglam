from pathlib import Path
from typing import Literal, Optional

import openeo
from openeo import DataCube
from openeo_gfmap import Backend, BackendContext, BoundingBoxExtent, TemporalContext
from openeo_gfmap.backend import BACKEND_CONNECTIONS
from worldcereal.openeo.mapping import _cropland_map, _croptype_map
from worldcereal.openeo.preprocessing import worldcereal_preprocessed_inputs
from worldcereal.parameters import (
    CropLandParameters,
    CropTypeParameters,
    WorldCerealProductType,
)


def _croptype_map_from_presto(
    inputs: DataCube,
    temporal_extent: TemporalContext,
    croptype_parameters: "CropTypeParameters",
    cropland_parameters: CropLandParameters,
    classes_list: Optional[list] = None,
) -> DataCube:
    """Method to produce croptype map from preprocessed inputs, using
    a Presto feature extractor and a CatBoost classifier.
    """

    # Run inference
    feature_parameters = croptype_parameters.feature_parameters.model_dump()
    if classes_list is None:
        raise ValueError("Please provide a `classes_list` parameter. Got None.")
    feature_parameters["classes_list"] = classes_list
    feature_parameters["num_outputs"] = len(classes_list)
    inference_udf = openeo.UDF.from_file(
        path=Path(__file__).resolve().parent / "predict_with_presto_udf.py",
        context=feature_parameters,
    )

    predictions = inputs.apply_neighborhood(
        process=inference_udf,
        size=[
            {"dimension": "x", "unit": "px", "value": 128},
            {"dimension": "y", "unit": "px", "value": 128},
            # {"dimension": "t", "value": "P1D"},
        ],
        overlap=[
            {"dimension": "x", "unit": "px", "value": 0},
            {"dimension": "y", "unit": "px", "value": 0},
        ],
    )

    # Get rid of temporal dimension
    # predictions = predictions.reduce_dimension(dimension="t", reducer="mean")

    # Mask cropland
    if cropland_mask is not None:
        predictions = predictions.mask(cropland_mask == 0, replacement=65335)

    # # Postprocess
    # if postprocess_parameters.enable:
    #     if postprocess_parameters.save_intermediate:
    #         predictions = predictions.save_result(
    #             format="GTiff",
    #             options=dict(
    #                 filename_prefix=f"{WorldCerealProductType.CROPTYPE.value}-raw_{temporal_extent.start_date}_{temporal_extent.end_date}"
    #             ),
    #         )
    #     predictions = _postprocess(
    #         predictions,
    #         postprocess_parameters,
    #         classifier_url=croptype_parameters.classifier_parameters.classifier_url,
    #     )

    # Cast to uint16
    # classes = compress_uint16(classes)

    return predictions


def create_inference_process_graph(
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
    product_type: WorldCerealProductType = WorldCerealProductType.CROPLAND,
    cropland_parameters: CropLandParameters = CropLandParameters(),
    croptype_parameters: CropTypeParameters = CropTypeParameters(),
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]] = None,
    out_format: str = "GTiff",
    backend_context: BackendContext = BackendContext(Backend.CDSE),
    tile_size: Optional[int] = 128,
    target_epsg: Optional[int] = None,
    predict_with_presto: bool = False,
    classes_list: Optional[list] = None,
    connection: Optional[openeo.Connection] = None,
) -> openeo.DataCube:
    """Wrapper function that creates the inference openEO process graph.

    Parameters
    ----------
    spatial_extent : BoundingBoxExtent
        spatial extent of the map
    temporal_extent : TemporalContext
        temporal range to consider
    product_type : WorldCerealProductType, optional
        product describer, by default WorldCerealProductType.CROPLAND
    cropland_parameters: CropLandParameters
        Parameters for the cropland product inference pipeline.
    croptype_parameters: Optional[CropTypeParameters]
        Parameters for the croptype product inference pipeline. Only required
        whenever `product_type` is set to `WorldCerealProductType.CROPTYPE`,
        will be ignored otherwise.
    s1_orbit_state: Optional[Literal["ASCENDING", "DESCENDING"]]
        Sentinel-1 orbit state to use for the inference. If not provided,
        the orbit state will be dynamically determined based on the spatial extent.
    out_format : str, optional
        Output format, by default "GTiff"
    backend_context : BackendContext
        backend to run the job on, by default CDSE.
    tile_size: int, optional
        Tile size to use for the data loading in OpenEO, by default 128.
    target_epsg: Optional[int] = None
        EPSG code to use for the output products. If not provided, the
        default EPSG will be used.
    connection: Optional[openeo.Connection] = None,
        Optional OpenEO connection to use. If not provided, a new connection
        will be created based on the backend_context.

    Returns
    -------
    List[openeo.DataCube]
        A list with one or more result objects or a list of DataCube objects, representing the inference
        process graph. This object can be used to execute the job on the OpenEO backend.
        The result will be a DataCube with the classification results.

    Raises
    ------
    ValueError
        if the product is not supported
    ValueError
        if the out_format is not supported
    """
    if product_type not in WorldCerealProductType:
        raise ValueError(f"Product {product_type.value} not supported.")

    if out_format not in ["GTiff", "NetCDF"]:
        raise ValueError(f"Format {format} not supported.")

    # Make a connection to the OpenEO backend
    if connection is None:
        connection = BACKEND_CONNECTIONS[backend_context.backend]()

    # Preparing the input cube for inference
    inputs = worldcereal_preprocessed_inputs(
        connection=connection,
        backend_context=backend_context,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        tile_size=tile_size,
        s1_orbit_state=s1_orbit_state,
        target_epsg=target_epsg,
        # disable_meteo=True,
    )

    # Spatial filtering
    inputs = inputs.filter_bbox(dict(spatial_extent))

    # Construct the feature extraction and model inference pipeline
    if product_type == WorldCerealProductType.CROPLAND:
        results = _cropland_map(
            inputs,
            temporal_extent,
            cropland_parameters=cropland_parameters,
        )

    elif product_type == WorldCerealProductType.CROPTYPE:
        if not isinstance(croptype_parameters, CropTypeParameters):
            raise ValueError(
                f"Please provide a valid `croptype_parameters` parameter."
                f" Received: {croptype_parameters}"
            )

        # Generate crop type map
        if predict_with_presto:
            results = _croptype_map_from_presto(
                inputs,
                temporal_extent,
                croptype_parameters=croptype_parameters,
                cropland_parameters=cropland_parameters,
                classes_list=classes_list,
            )
        else:
            # Generate crop type map with optional cropland masking
            results = _croptype_map(
                inputs,
                temporal_extent,
                cropland_parameters=cropland_parameters,
                croptype_parameters=croptype_parameters,
            )

    return results
