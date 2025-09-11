from typing import Dict, List

PRESTO_PRETRAINED_MODEL_PATH: Dict[str, str] = {
    "LANDCOVER": "/projects/worldcereal/COP4GEOGLAM/presto-prometheo-landcover-month-LANDCOVER10-augment=True-balance=True-timeexplicit=False-run=202507170930_encoder.pt",
    "CROPTYPE": "/projects/worldcereal/COP4GEOGLAM/presto-prometheo-landcover-month-CROPTYPE27-augment=True-balance=True-timeexplicit=True-run=202507181013_encoder.pt",
    "DEFAULT": "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ss-wc_longparquet_random-window-cut_no-time-token_epoch96.pt",
}

COUNTRY_PARQUET_FILES: Dict[str, Dict[str, List[str]]] = {
    "kenya": {
        "month": [
            "/vitodata/worldcereal/data/COP4GEOGLAM/kenya/trainingdata/2021_KEN_COPERNICUS-GEOGLAM-SR_POINT_111.geoparquet",
            "/vitodata/worldcereal/data/COP4GEOGLAM/kenya/trainingdata/2021_KEN_COPERNICUS-GEOGLAM-LR_POINT_111.geoparquet",
        ],
        "dekad": [],
    },
    "moldova": {
        "month": [
            "/projects/worldcereal/COP4GEOGLAM/moldova/worldcereal_merged_extractions.parquet"
        ],
        "dekad": [],
    },
    "mozambique": {"month": [], "dekad": []},
}

COUNTRY_CLASS_MAPPINGS: Dict[str, str] = {
    "kenya": "kenya",
    "moldova": "moldova",
    "mozambique": "mozambique",
}

PRODUCTION_MODELS_URLS: Dict[str, Dict[str, Dict[str, str]]] = {
    "moldova": {
        "presto": {
            "cropland": "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/Copernicus4Geoglam/moldova/presto-prometheo-cop4geoglam-run-with-AL-and-freezing-month-LANDCOVER10-augment%3DFalse-balance%3DTrue-timeexplicit%3DFalse-run%3D202509111120_encoder.pt",
            "croptype": "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/Copernicus4Geoglam/moldova/presto-prometheo-cop4geoglam-run-with-AL-and-freezing-month-CROPTYPE_Moldova-augment%3DFalse-balance%3DTrue-timeexplicit%3DFalse-run%3D202509110852_encoder.pt",
        },
        "catboost": {
            "cropland": "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/Copernicus4Geoglam/moldova/Presto_run%3D202509111120_DownstreamCatBoost_cropland_v120-MDA_balance%3DFalse.onnx",
            "croptype": "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/Copernicus4Geoglam/moldova/Presto_run%3D202509110852_DownstreamCatBoost_croptype_v120-MDA_balance%3DTrue.onnx",
        },
    }
}
