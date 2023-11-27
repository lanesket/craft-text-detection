"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""
from typing import Tuple, List
import cv2
import numpy as np


def read_image(image):
    if type(image) == str:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif type(image) == bytes:
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif type(image) == np.ndarray:
        if len(image.shape) == 2:  # grayscale
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            img = image
        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBAscale
            img = image[:, :, :3]

    return img


def normalize(
        img: np.ndarray, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Normalize the image with mean and variance.

    Args:
        img: image to be normalized
        mean: mean value
        variance: variance value

    Returns:
        normalized image
    """
    # should be RGB order
    img = img.copy().astype(np.float32)

    img -= np.array(
        [mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32
    )
    img /= np.array(
        [variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0],
        dtype=np.float32,
    )
    return img


def denormalize(
        img: np.ndarray, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Inverse of normalize function.

    Args:
        img: image to be denormalized
        mean: mean value
        variance: variance value

    Returns:
        denormalized image
    """
    # should be RGB order
    img = img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def resize_aspect_ratio(img: np.ndarray, long_size: int, interpolation=cv2.INTER_LINEAR) -> Tuple[
    np.ndarray, float, Tuple[int, int]]:
    """
    Resize image with aspect ratio.

    Args:
        img: image to be resized
        long_size: target size for resizing
        interpolation: interpolation method

    Returns:
        resized image, ratio, size_heatmap
    """
    height, width, channel = img.shape

    # set target image size
    target_size = long_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w / 2), int(target_h / 2))

    return resized, ratio, size_heatmap


def resize_aspect_ratio_batch(imgs: List[np.ndarray], long_size: int, interpolation=cv2.INTER_LINEAR) -> Tuple[
    List[np.ndarray], List[float], List[Tuple[int, int]]]:
    """
    Resize a batch of images with aspect ratio.

    Args:
        imgs: List of images to be resized
        long_size: target size for resizing
        interpolation: interpolation method

    Returns:
        A tuple of three lists:
        - List of resized images
        - List of ratios
        - List of size_heatmaps
    """
    resized_images = []
    ratios = []
    size_heatmaps = []

    for img in imgs:
        resized_img, ratio, size_heatmap = resize_aspect_ratio(
            img, long_size, interpolation
        )
        resized_images.append(resized_img)
        ratios.append(ratio)
        size_heatmaps.append(size_heatmap)

    return resized_images, ratios, size_heatmaps


def img_2_heatmap(img: np.ndarray) -> np.ndarray:
    """
    Convert image to heatmap image.

    Args:
        img: image to be converted

    Returns:
        heatmap image
    """
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img
