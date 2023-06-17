import yaml
from typing import Dict

from wood_analyzer_processor.models import CamelBaseModel
from wood_analyzer_processor.resources_paths import config_path


class DataPaths(CamelBaseModel):
    base: str
    prediction: str
    postprocessed: str
    segmentation: str
    measurement: str


class General(CamelBaseModel):
    patch_size: int
    test_fold_id: int
    total_folds_no: int


class Prediction(CamelBaseModel):
    stride: str


class ImageAnalyzerConfig(CamelBaseModel):
    data_paths: DataPaths
    general: General
    prediction: Prediction
    labels: Dict[str, int]


def get_config(config_path: str) -> ImageAnalyzerConfig:
    with open(config_path, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.Loader)
    return ImageAnalyzerConfig(**config_dict)


config = get_config(config_path)
