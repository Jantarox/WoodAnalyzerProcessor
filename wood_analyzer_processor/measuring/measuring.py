import numpy as np
from skimage.measure import label

from wood_analyzer_processor.config import config
from wood_analyzer_processor.logger import logger


def label_measure_tree_rings(segmentation_array: np.ndarray):
    logger.info(f"Labeling and measuring growth rings...")
    labels_dict = config.labels
    tree_rings_mask = np.where(segmentation_array == labels_dict["G"], 0, 1).astype(
        np.uint8
    )
    tree_rings_labeled_mask = label(tree_rings_mask, connectivity=1).astype(np.uint8)
    labels, counts = np.unique(tree_rings_labeled_mask, return_counts=True)
    tree_rings_areas = dict(zip(labels, counts))
    tree_rings_areas.pop(0)

    return tree_rings_labeled_mask, tree_rings_areas


def label_measure_resin_ducts(segmentation_array: np.ndarray):
    logger.info(f"Labeling and measuring resin ducts...")
    labels_dict = config.labels
    resin_ducts_mask = np.where(segmentation_array == labels_dict["R"], 1, 0).astype(
        np.uint8
    )
    resin_ducts_area = np.sum(resin_ducts_mask)
    resin_ducts_labeled_mask = label(resin_ducts_mask, connectivity=1)
    resin_ducts_count = np.max(resin_ducts_labeled_mask)

    return resin_ducts_mask, resin_ducts_area, resin_ducts_count


def measure_ducts_per_ring(
    tree_rings_labeled_mask: np.ndarray, resin_ducts_mask: np.ndarray
):
    logger.info(f"Calculating resin ducts per ring area...")
    ducts_per_ring_mask = np.multiply(resin_ducts_mask, tree_rings_labeled_mask)
    labels, counts = np.unique(ducts_per_ring_mask, return_counts=True)
    ducts_areas_per_ring = dict(zip(labels, counts))
    ducts_areas_per_ring.pop(0)

    return ducts_areas_per_ring
