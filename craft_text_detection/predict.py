from typing import List
import os
import time
import torch

import cv2
import numpy as np

import craft_text_detection.craft_utils as craft_utils
import craft_text_detection.image_utils as image_utils
import craft_text_detection.torch_utils as torch_utils
import concurrent.futures


def get_prediction(
        image,
        craft_net,
        refine_net=None,
        text_threshold: float = 0.7,
        link_threshold: float = 0.4,
        low_text: float = 0.4,
        device: str = 'cpu',
        long_size: int = 1280,
        poly: bool = True,
):
    """
    Predicts character/bounding box regions of given image.

    Args:
        image: file path or numpy-array or a byte stream object of an image.
        craft_net: craft net model
        refine_net: refine net model
        text_threshold: text confidence threshold
        link_threshold: link confidence threshold
        low_text: text low-bound score
        device: device to run inference
        long_size: desired longest image size for inference
        poly: enable polygon type result

    Returns:
        dict: result information with keys "masks", "boxes", "boxes_as_ratios", "polys_as_ratios", "heatmaps", "times".
            masks: lists of predicted masks 2d as bool array,
            boxes: list of coords of points of predicted boxes,
            boxes_as_ratios: list of coords of points of predicted boxes as ratios of image size,
            polys_as_ratios: list of coords of points of predicted polys as ratios of image size,
            heatmaps: visualizations of the detected characters/links,
            times: elapsed times of the submodules, in seconds

    """
    t0 = time.time()

    # read/convert image
    image = image_utils.read_image(image)

    # resize
    img_resized, target_ratio, size_heatmap = image_utils.resize_aspect_ratio(
        image, long_size, interpolation=cv2.INTER_LINEAR
    )
    ratio_h = ratio_w = 1 / target_ratio
    resize_time = time.time() - t0
    t0 = time.time()

    # preprocessing
    x = image_utils.normalize(img_resized)
    x = torch_utils.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = torch_utils.Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if device != 'cpu':
        x = x.to(device)
    preprocessing_time = time.time() - t0
    t0 = time.time()

    # forward pass
    with torch_utils.no_grad():
        y, feature = craft_net(x)
    craftnet_time = time.time() - t0
    t0 = time.time()

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch_utils.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()
    refinenet_time = time.time() - t0
    t0 = time.time()

    # Post-processing
    boxes, polys = craft_utils.get_det_boxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly
    )

    # coordinate adjustment
    boxes = craft_utils.adjust_result_coordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjust_result_coordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    # get image size
    img_height = image.shape[0]
    img_width = image.shape[1]

    # calculate box coords as ratios to image size
    boxes_as_ratio = []
    for box in boxes:
        boxes_as_ratio.append(box / [img_width, img_height])

    # calculate poly coords as ratios to image size
    polys_as_ratio = []
    for poly in polys:
        polys_as_ratio.append(poly / [img_width, img_height])

    text_score_heatmap = image_utils.img_2_heatmap(score_text)
    link_score_heatmap = image_utils.img_2_heatmap(score_link)

    postprocess_time = time.time() - t0

    times = {
        "resize_time": resize_time,
        "preprocessing_time": preprocessing_time,
        "craftnet_time": craftnet_time,
        "refinenet_time": refinenet_time,
        "postprocess_time": postprocess_time,
    }

    return {
        "boxes": boxes,
        "boxes_as_ratios": boxes_as_ratio,
        "polys": polys,
        "polys_as_ratios": polys_as_ratio,
        "heatmaps": {
            "text_score_heatmap": text_score_heatmap,
            "link_score_heatmap": link_score_heatmap,
        },
        "times": times,
    }


def process_image_postprocessing(i, score_text, score_link, text_threshold, link_threshold, low_text, poly,
                                 target_ratios, images):
    # Function to handle post-processing of a single image in the batch

    # Post-processing for each image
    boxes, polys = craft_utils.get_det_boxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly
    )

    ratio = 1 / target_ratios[i]
    boxes = craft_utils.adjust_result_coordinates(boxes, ratio, ratio)
    polys = craft_utils.adjust_result_coordinates(polys, ratio, ratio)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    img_height, img_width, _ = images[i].shape
    boxes_as_ratio = [box / [img_width, img_height] for box in boxes]
    polys_as_ratio = [poly / [img_width, img_height] for poly in polys]

    text_score_heatmap = image_utils.img_2_heatmap(score_text)
    link_score_heatmap = image_utils.img_2_heatmap(score_link)

    return {
        "boxes": boxes,
        "boxes_as_ratios": boxes_as_ratio,
        "polys": polys,
        "polys_as_ratios": polys_as_ratio,
        "heatmaps": {
            "text_score_heatmap": text_score_heatmap,
            "link_score_heatmap": link_score_heatmap
        }
    }


def get_prediction_batch(
        images: List[np.ndarray],
        craft_net,
        refine_net=None,
        text_threshold: float = 0.7,
        link_threshold: float = 0.4,
        low_text: float = 0.4,
        device: str = 'cpu',
        long_size: int = 1280,
        poly: bool = True,
):
    # Function to process a batch of images

    t0 = time.time()

    # Resize and preprocess each image
    img_resized_batch, target_ratios, size_heatmaps = \
        image_utils.resize_aspect_ratio_batch(images, long_size,
                                              interpolation=cv2.INTER_LINEAR)

    # Convert to tensor batch
    x_batch = [torch_utils.from_numpy(image_utils.normalize(img)).permute(2, 0, 1).unsqueeze(0) for img in
               img_resized_batch]
    x_batch = torch.cat(x_batch, axis=0)

    if device != 'cpu':
        x_batch = x_batch.to(device)

    preprocessing_time = time.time() - t0
    t0 = time.time()

    # Forward pass for the batch
    with torch_utils.no_grad():
        y_batch, feature_batch = craft_net(x_batch)

    craftnet_time = time.time() - t0
    t0 = time.time()

    refinenet_time = None
    if refine_net is not None:
        with torch_utils.no_grad():
            y_refined_batch = [refine_net(y.unsqueeze(0), feature.unsqueeze(0))[0, :, :, 0]
                               for y, feature in zip(y_batch, feature_batch)]
        y_refined_batch = torch.stack(y_refined_batch)  # Stacking the refined outputs into a batch
        refinenet_time = time.time() - t0
        t0 = time.time()
    # Initialize batch results
    batch_results = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(y_batch.shape[0]):
            score_text = y_batch[i, :, :, 0].cpu().data.numpy()
            score_link = y_refined_batch[i].cpu().data.numpy() if refine_net is not None else y_batch[i, :, :,
                                                                                              1].cpu().data.numpy()

            future = executor.submit(process_image_postprocessing, i, score_text, score_link, text_threshold,
                                     link_threshold, low_text,
                                     poly, target_ratios, images)
            futures.append(future)

        # Wait for all threads to complete
        for future in concurrent.futures.as_completed(futures):
            batch_results.append(future.result())

    postprocess_time = time.time() - t0
    times = {
        "preprocessing_time": preprocessing_time,
        "craftnet_time": craftnet_time,
        "refinenet_time": refinenet_time,
        "postprocess_time": postprocess_time,
    }

    return batch_results, times
