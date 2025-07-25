from typing import Dict, List

PRESTO_PRETRAINED_MODEL_PATH: Dict[str, str] = {
    "LANDCOVER" : "/vitodata/worldcereal/models/WorldCerealPresto/presto-prometheo-landcover-month-LANDCOVER10-augment=True-balance=True-timeexplicit=False-run=202507170930/presto-prometheo-landcover-month-LANDCOVER10-augment=True-balance=True-timeexplicit=False-run=202507170930_encoder.pt",
    "CROPTYPE" : "/vitodata/worldcereal/models/WorldCerealPresto/presto-prometheo-landcover-month-CROPTYPE27-augment=True-balance=True-timeexplicit=True-run=202507181013/presto-prometheo-landcover-month-CROPTYPE27-augment=True-balance=True-timeexplicit=True-run=202507181013_encoder.pt",
    "DEFAULT" : "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ss-wc_longparquet_random-window-cut_no-time-token_epoch96.pt"
}

COUNTRY_PARQUET_FILES: Dict[str, Dict[str, List[str]]] = {
    "kenya": {
        "month": [
            "/vitodata/worldcereal/data/COP4GEOGLAM/kenya/trainingdata/2021_KEN_COPERNICUS-GEOGLAM-SR_POINT_111.geoparquet",
            "/vitodata/worldcereal/data/COP4GEOGLAM/kenya/trainingdata/2021_KEN_COPERNICUS-GEOGLAM-LR_POINT_111.geoparquet"
        ],
        "dekad": []
    },

    "moldova": {
        "month": [
            "/vitodata/worldcereal/data/COP4GEOGLAM/moldova/trainingdata/worldcereal_merged_extractions.parquet"
        ],
        "dekad": []
    },
    "mozambique": {
        "month": [],
        "dekad": []
    }
}

COUNTRY_CLASS_MAPPINGS: Dict[str, str] = {
    "kenya": "kenya",
    "moldova": "Moldova_prelim",
    "mozambique": "Mozambique_prelim"
}
