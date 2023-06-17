"""
Created on Sat Oct 30 22:45:49 2021

@author: an_fab
"""
import numpy as np
from scipy.signal import find_peaks
from skimage.morphology import square, closing, disk, thin
from scipy.ndimage import binary_closing, binary_dilation, median_filter
from skimage.segmentation import watershed
from skimage import measure
from skimage.filters import median

from wood_analyzer_processor.logger import logger

# ---------------------------------------------------------------------------------------------


def find_local_peaks(img):
    logger.info("Finding local peaks...")
    img = closing(img, square(3))
    img = median(img, disk(9))

    peaksMap = np.zeros(img.shape)

    for y in range(0, img.shape[0]):
        line = img[y, :]
        # peaks = find_peaks_cwt(line, np.arange(10,15))
        peaks, _ = find_peaks(line, prominence=2)

        if peaks.size > 0:
            peaksMap[y, peaks] = 255

    for x in range(0, img.shape[1]):
        line = img[:, x]
        # peaks = find_peaks_cwt(line, np.arange(10,15))
        peaks, _ = find_peaks(line, prominence=2)

        if peaks.size > 0:
            peaksMap[peaks, x] = 255

    labels = measure.label(peaksMap, background=0)

    # labels, _ = ndimage.label(peaksMap)

    (count, lab) = np.histogram(labels, bins=np.max(labels))

    for i in range(0, np.max(labels)):
        if count[i] < 10:
            indx = np.where(labels == lab[i])
            peaksMap[indx] = 0
            # peaksMap[labels == count[1][i]] = 0

    # labeled, nr_objects = ndimage.label(peaksMap)

    # i = range(1, nr_objects + 1)
    # count = np.histogram(labeled, bins=nr_objects)

    # for i in range(1, nr_objects+1):

    #     count = np.count_nonzero(labeled == i)

    #     if count < 100:
    #         peaksMap[labeled == i] = 0

    return peaksMap


def local_thresholding(img, offset):
    bw = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            y1 = y - offset if y - offset > 0 else 0
            y2 = y + offset if y + offset < img.shape[0] else img.shape[0]
            x1 = x - offset if x - offset > 0 else 0
            x2 = x + offset if x + offset < img.shape[1] else img.shape[1]

            subImg = img[y1:y2, x1:x2]
            th = 1.05 * np.mean(subImg)

            bw[y, x] = 255 if img[y, x] > th else 0

    return bw


# ---------------------------------------------------------------------------------------------


def postprocess_array_old(array: np.ndarray):
    logger.info("Postprocessing array...")
    R = array[:, :, 0]
    G = array[:, :, 1]
    B = array[:, :, 2]

    B = median(B)

    bw_B = find_local_peaks(B)
    indB = bw_B > 0

    label_array = np.zeros_like(B)
    label_array[indB] = 1

    logger.info("Postprocessing done...")
    return label_array


def postprocess_array(array: np.ndarray):
    logger.info("Postprocessing array...")

    (Y, X, c) = array.shape

    # tree-ring boundaries detection

    B = array[:, :, 2].astype(float)
    B = 255 * B / np.max(B)
    B = median_filter(B, size=(5, 5))
    ind = B < 85
    B[ind] = 0

    ###
    bw_B1 = local_thresholding(B, 25)
    bw_B1 = binary_closing(bw_B1, np.ones((5, 5)))
    bw_B1 = thin(bw_B1)

    ####
    L = watershed(B, watershed_line=True)
    bw_B2 = np.zeros((Y, X)).astype(bool)
    ind = L == 0
    bw_B2[ind] = True

    ### combine
    bw_B = np.logical_or(bw_B1, bw_B2)
    bw_B = thin(bw_B).astype(int)

    # label_img = label(bw_B)
    # regions = regionprops(label_img)

    # for num, reg in enumerate(regions):
    #     if reg.num_pixels < 20:
    #         indx = (label_img == num)
    #         bw_B[indx] = 0

    bw_B = binary_dilation(bw_B, np.ones((5, 5)))

    indB = bw_B > 0

    # resin ducts detection

    R = array[:, :, 0].astype(float)
    R = 255 * R / np.max(R)
    R = median_filter(R, size=(5, 5))
    ind = R < 100
    R[ind] = 0
    indR = R > 0

    label_array = np.zeros_like(array[:, :, 0])
    label_array[indB] = 1
    label_array[indR] = 2

    logger.info("Postprocessing done...")
    return label_array
