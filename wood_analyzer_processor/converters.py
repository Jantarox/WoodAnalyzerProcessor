import json

from wood_analyzer_processor.logger import logger
from wood_analyzer_processor.models import Segmentation


def save_segmentation_to_json(segmentation: Segmentation, path: str):
    json_string = json.dumps(segmentation.encode())
    logger.info("Saving segmentation...")
    with open(path, "w") as file:
        file.write(json_string)
    logger.info("Segmentation saved.")


def get_segmentation_from_json(path: str) -> Segmentation:
    logger.info("Loading segmentation...")
    with open(path, "r") as file:
        data = json.load(file)
    segmentation = Segmentation.decode(data)
    logger.info("Segmentation loaded.")
    return segmentation
