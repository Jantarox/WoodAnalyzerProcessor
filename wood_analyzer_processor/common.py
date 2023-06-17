import os

import numpy as np
from skimage import io

from wood_analyzer_processor.config import config
from wood_analyzer_processor.segmentation.predict import (
    predict_image,
    resample_to_uint8,
)
from wood_analyzer_processor.segmentation.postprocessing import postprocess_array
from wood_analyzer_processor.measuring.measuring import (
    label_measure_tree_rings,
    label_measure_resin_ducts,
    measure_ducts_per_ring,
)
from wood_analyzer_processor.models import Segmentation
from wood_analyzer_processor.converters import (
    save_segmentation_to_json,
    get_segmentation_from_json,
)
from wood_analyzer_processor.logger import logger


def predict_images(main_path: str, save_image: bool = False) -> np.ndarray:
    base_image_dir = os.path.join(main_path, config.data_paths.base)
    pred_image_dir = os.path.join(main_path, config.data_paths.prediction)

    files = []

    for dirpath, dirnames, filenames in os.walk(base_image_dir):
        files.extend(filenames)
        break

    for file in files:
        base_image_path = os.path.join(base_image_dir, file)
        pred_image_path = os.path.join(pred_image_dir, file)

        prediction = predict_image(base_image_path)
        prediction = resample_to_uint8(prediction)
        if save_image:
            io.imsave(pred_image_path, prediction)

        return prediction


def postprocess_images(main_path: str):
    pred_path = os.path.join(main_path, config.data_paths.prediction)
    postprocess_path = os.path.join(main_path, config.data_paths.postprocessed)

    files = []

    for dirpath, dirnames, filenames in os.walk(pred_path):
        files.extend(filenames)
        break

    for file in files:
        pred_image_path = os.path.join(pred_path, file)
        post_image_path = os.path.join(postprocess_path, file)
        print(pred_image_path)

        pred_array = io.imread(pred_image_path)
        post_array = postprocess_array(pred_array)

        post_image = np.zeros(
            (post_array.shape[0], post_array.shape[1], 4), dtype=np.uint8
        )
        growth_idx = post_array == 1
        post_image[:, :, 2][growth_idx] = 255
        post_image[:, :, 3][growth_idx] = 255

        io.imsave(post_image_path, post_image)


def generate_segmentations(main_dir: str, generate_png: bool = False):
    logger.info("Generating segmentations...")
    base_image_dir = os.path.join(main_dir, config.data_paths.base)
    postprocess_image_dir = os.path.join(main_dir, config.data_paths.postprocessed)
    seg_image_dir = os.path.join(main_dir, config.data_paths.segmentation)

    files = []

    for dirpath, dirnames, filenames in os.walk(base_image_dir):
        files.extend(filenames)
        break

    for file in files:
        base_image_path = os.path.join(base_image_dir, file)

        file = file.replace(".png", ".json")
        seg_file_path = os.path.join(seg_image_dir, file)

        pred_array = predict_image(base_image_path)
        pred_array = resample_to_uint8(pred_array)
        post_array = postprocess_array(pred_array)

        segmentation = Segmentation.from_array(segmentation_array=post_array)

        save_segmentation_to_json(segmentation, seg_file_path)

        if generate_png:
            post_image_path = os.path.join(postprocess_image_dir, file)
            post_image = np.zeros(
                (post_array.shape[0], post_array.shape[1], 4), dtype=np.uint8
            )
            growth_idx = post_array == 1
            post_image[:, :, 2][growth_idx] = 255
            post_image[:, :, 3][growth_idx] = 255
            io.imsave(post_image_path, post_image)

    logger.info("All done.")


def generate_segmentation(main_dir: str, file: str, generate_png: bool = False):
    logger.info(f"Generating segmentation for {file}...")
    base_image_dir = os.path.join(main_dir, config.data_paths.base)
    postprocess_image_dir = os.path.join(main_dir, config.data_paths.postprocessed)
    seg_image_dir = os.path.join(main_dir, config.data_paths.segmentation)

    base_image_path = os.path.join(base_image_dir, file)

    file = file.replace(".png", ".json")
    seg_file_path = os.path.join(seg_image_dir, file)

    pred_array = predict_image(base_image_path)
    pred_array = resample_to_uint8(pred_array)
    post_array = postprocess_array(pred_array)
    segmentation = Segmentation.from_array(segmentation_array=post_array)

    logger.info(f"Calculating measurements...")

    tree_rings_labeled_mask, tree_rings_areas = label_measure_tree_rings(
        segmentation_array=segmentation.segmentation_array
    )
    resin_ducts_mask, resin_ducts_area, resin_ducts_count = label_measure_resin_ducts(
        segmentation_array=segmentation.segmentation_array
    )
    ducts_areas_per_ring = measure_ducts_per_ring(
        tree_rings_labeled_mask, resin_ducts_mask
    )
    segmentation.rings_array = tree_rings_labeled_mask
    segmentation.add_areas(
        tree_rings_areas=tree_rings_areas,
        ducts_areas_per_ring=ducts_areas_per_ring,
        resin_ducts_area=resin_ducts_area,
        resin_ducts_count=resin_ducts_count,
    )

    save_segmentation_to_json(segmentation, seg_file_path)

    if generate_png:
        post_image_path = os.path.join(postprocess_image_dir, file)
        post_image = np.zeros(
            (post_array.shape[0], post_array.shape[1], 4), dtype=np.uint8
        )
        growth_idx = post_array == 1
        post_image[:, :, 2][growth_idx] = 255
        post_image[:, :, 3][growth_idx] = 255
        io.imsave(post_image_path, post_image)

    logger.info("All done.")


def calculate_measurements(main_dir: str, file: str):
    logger.info(f"Calculating measurements for segmentation {file}...")
    segmentation_dir = os.path.join(main_dir, config.data_paths.segmentation)

    seg_file_path = os.path.join(segmentation_dir, file)

    segmentation = get_segmentation_from_json(seg_file_path)

    tree_rings_labeled_mask, tree_rings_areas = label_measure_tree_rings(
        segmentation_array=segmentation.segmentation_array
    )
    resin_ducts_mask, resin_ducts_area, resin_ducts_count = label_measure_resin_ducts(
        segmentation_array=segmentation.segmentation_array
    )
    ducts_areas_per_ring = measure_ducts_per_ring(
        tree_rings_labeled_mask, resin_ducts_mask
    )
    segmentation.rings_array = tree_rings_labeled_mask
    segmentation.add_areas(
        tree_rings_areas=tree_rings_areas,
        ducts_areas_per_ring=ducts_areas_per_ring,
        resin_ducts_area=resin_ducts_area,
        resin_ducts_count=resin_ducts_count,
    )

    save_segmentation_to_json(segmentation, seg_file_path)
    logger.info("All done.")
