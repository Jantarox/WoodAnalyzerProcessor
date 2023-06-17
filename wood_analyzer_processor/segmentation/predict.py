# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 20:01:15 2021

@author: an_fab
"""
import numpy as np
from skimage import io
from keras.models import model_from_json

from wood_analyzer_processor.config import config
from wood_analyzer_processor.resources_paths import model_path, weights_path
from wood_analyzer_processor.logger import logger

# ----------------------------------------------------------------------------
# read config file

patch_size = int(config.general.patch_size)
stride = int(config.prediction.stride)
# ----------------------------------------------------------------------------


def get_patches(img, patchSize, stride):
    logger.info("Getting patches from the image...")
    h = img.shape[0]
    w = img.shape[1]

    assert (h - patchSize) % stride == 0 and (w - patchSize) % stride == 0

    H = (h - patchSize) // stride + 1
    W = (w - patchSize) // stride + 1

    patches = np.empty((W * H, patchSize, patchSize, 3), dtype=np.float16)
    iter_tot = 0

    for h in range(H):
        for w in range(W):
            patch = img[
                h * stride : (h * stride) + patchSize,
                w * stride : (w * stride) + patchSize,
                :,
            ]
            patches[iter_tot] = patch
            iter_tot += 1

    return patches


# ---------------------------------------------------------------------------------------------


def build_img_from_patches(preds, img_h, img_w, stride) -> np.ndarray:
    logger.info("Building prediction array from predicted patches...")

    patch_h = preds.shape[1]
    patch_w = preds.shape[2]
    nch = preds.shape[3]

    H = (img_h - patch_h) // stride + 1
    W = (img_w - patch_w) // stride + 1

    prob = np.zeros((img_h, img_w, nch), dtype=np.float16)
    _sum = np.zeros((img_h, img_w, nch), dtype=np.float16)

    k = 0

    for h in range(H):
        for w in range(W):
            sth = preds[k, :, :, :]
            sth = np.reshape(sth, (patch_h, patch_w, nch))
            prob[
                h * stride : (h * stride) + patch_h,
                w * stride : (w * stride) + patch_w,
                :,
            ] += sth
            _sum[
                h * stride : (h * stride) + patch_h,
                w * stride : (w * stride) + patch_w,
                :,
            ] += 1
            k += 1

    final_avg = prob / _sum

    logger.debug(f"Shape of prediction array built from patches: {final_avg.shape}")

    return final_avg


# ---------------------------------------------------------------------------------------------


def add_outline(img, patchSize, stride):
    logger.info("Adding outline...")
    logger.debug(f"Array shape: {img.shape}")
    img_h = img.shape[0]
    img_w = img.shape[1]
    n_ch = img.shape[2]
    leftover_h = (img_h - patchSize) % stride
    leftover_w = (img_w - patchSize) % stride

    buf = np.zeros((2 * img_h, 2 * img_w, n_ch))
    img_vert = np.flip(img, axis=0)
    buf[0:img_h, 0:img_w, :] = img
    buf[img_h : 2 * img_h, 0:img_w, :] = img_vert

    if leftover_h != 0:
        tmp = np.zeros((img_h + (stride - leftover_h), img_w, 3), dtype=np.float16)
        # tmp[0:img_h,0:img_w,:] = img
        tmp = buf[0 : img_h + (stride - leftover_h), 0:img_w, :]
        img = tmp
    if leftover_w != 0:
        tmp = np.zeros(
            (img.shape[0], img_w + (stride - leftover_w), 3), dtype=np.float16
        )
        tmp[0 : img.shape[0], 0:img_w, :] = img
        img = tmp

    logger.debug(f"New array shape: {img.shape}")

    return img


# ---------------------------------------------------------------------------------------------


def predict_image(org_path) -> np.ndarray:
    logger.info(f"Generating prediction array...")
    patch_size = int(config.general.patch_size)
    stride = int(config.prediction.stride)

    org = io.imread(org_path)
    org = np.asarray(org, dtype="float16")
    org = org / 255
    org = org[:, :, 0:3]

    logger.debug(f"Original image path: {org_path}")

    height = org.shape[0]
    width = org.shape[1]

    logger.info(f"Array dimentions: ({height} x {width})")

    org = add_outline(org, patch_size, stride)

    new_height = org.shape[0]
    new_width = org.shape[1]

    org = np.reshape(org, (new_height, new_width, 3))

    patches = get_patches(org, patch_size, stride)

    logger.info("Predicting growth rings and resin ducts...")
    model = model_from_json(open(model_path).read())
    model.load_weights(weights_path)
    predictions = model.predict(patches, batch_size=32, verbose=2)
    logger.info("Prediction done.")

    logger.debug(f"Prediction array size : {predictions.shape}")

    pred_patches = predictions
    pred_img = build_img_from_patches(pred_patches, new_height, new_width, stride)
    pred_img = pred_img[0:height, 0:width, :]

    logger.info("Prediction array done.")
    return pred_img


# ---------------------------------------------------------------------------------------------


def resample_to_uint8(array: np.ndarray) -> np.ndarray:
    logger.info("Resampling array...")
    R = array[:, :, 1]
    R = 255 * R / np.max(R)

    G = np.zeros(R.shape)

    B = array[:, :, 2]
    B = 255 * B / np.max(B)

    array = np.zeros((array.shape[0], array.shape[1], 3), dtype=np.uint8)
    array[:, :, 0] = R.astype(np.uint8)
    array[:, :, 1] = G.astype(np.uint8)
    array[:, :, 2] = B.astype(np.uint8)

    logger.info("Resampling done.")
    return array


# ---------------------------------------------------------------------------------------------
