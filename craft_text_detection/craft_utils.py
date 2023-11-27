from typing import List, Tuple
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from craft_text_detection.models.craftnet import CraftNet
from craft_text_detection.models.refinenet import RefineNet
import craft_text_detection.file_utils as file_utils
import craft_text_detection.torch_utils as torch_utils

CRAFT_GDRIVE_URL = "https://drive.google.com/uc?id=1bupFXqT-VU6Jjeul13XP7yx2Sg5IHr4J"
REFINENET_GDRIVE_URL = (
    "https://drive.google.com/uc?id=1xcE9qpJXp4ofINwXWVhhQIh9S8Z7cuGj"
)


def warp_coord(inv_p_m: np.ndarray, pt: np.ndarray) -> np.ndarray:
    """
    Warp coordinates based on inverse perspective matrix.

    Args:
        inv_p_m: inverse perspective matrix
        pt: point to warp

    Returns:
        warped point
    """
    out = np.matmul(inv_p_m, (pt[0], pt[1], 1))
    return np.array([out[0] / out[2], out[1] / out[2]])


def copy_state_dict(state_dict: OrderedDict) -> OrderedDict:
    """
    Copies state dict.

    Args:
        state_dict: state dict to copy

    Returns:
        copied state dict
    """
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def load_craftnet_model(
        device: str = 'cpu',
        weight_path: Optional[Union[str, Path]] = None
) -> CraftNet:
    """
    Loads craftnet model.

    Args:
        device: device to use
        weight_path: path to weight file

    Returns:
        loaded craftnet model
    """
    if weight_path is None:
        home_path = str(Path.home())
        weight_path = Path(
            home_path,
            ".craft_text_detection",
            "weights",
            "craft_mlt_25k.pth"
        )
    weight_path = Path(weight_path).resolve()
    weight_path.parent.mkdir(exist_ok=True, parents=True)
    weight_path = str(weight_path)

    craft_net = CraftNet()

    url = CRAFT_GDRIVE_URL
    if not os.path.isfile(weight_path):
        print("Craft text detector weight will be downloaded to {}".format(weight_path))

        file_utils.download(url=url, save_path=weight_path)

    if 'cuda' in device:
        craft_net.load_state_dict(copy_state_dict(torch_utils.load(weight_path)))

        craft_net = craft_net.to(device)
        torch_utils.cudnn_benchmark = False
    else:
        craft_net.load_state_dict(
            copy_state_dict(torch_utils.load(weight_path, map_location="cpu"))
        )

    craft_net.eval()

    return craft_net


def load_refinenet_model(
        device: str = 'cpu',
        weight_path: Optional[Union[str, Path]] = None
) -> RefineNet:
    if weight_path is None:
        home_path = Path.home()
        weight_path = Path(
            home_path,
            ".craft_text_detection",
            "weights",
            "craft_refiner_CTW1500.pth"
        )
    weight_path = Path(weight_path).resolve()
    weight_path.parent.mkdir(exist_ok=True, parents=True)
    weight_path = str(weight_path)

    refine_net = RefineNet()

    url = REFINENET_GDRIVE_URL
    if not os.path.isfile(weight_path):
        print("Craft text refiner weight will be downloaded to {}".format(weight_path))

        file_utils.download(url=url, save_path=weight_path)

    if 'cuda' in device:
        refine_net.load_state_dict(copy_state_dict(torch_utils.load(weight_path)))

        refine_net = refine_net.to(device)
        torch_utils.cudnn_benchmark = False
    else:
        refine_net.load_state_dict(
            copy_state_dict(torch_utils.load(weight_path, map_location="cpu"))
        )

    refine_net.eval()

    return refine_net


def get_det_boxes_core(textmap: np.ndarray,
                       linkmap: np.ndarray,
                       text_threshold: float,
                       link_threshold: float,
                       low_text_threshold: float) -> Tuple[List[np.ndarray], np.ndarray, List[int]]:
    """
    Extracts bounding boxes from textmap and linkmap.

    Args:
        textmap: textmap, shape (H, W)
        linkmap: linkmap, shape (H, W)
        text_threshold: text threshold, text above this value will be considered text
        link_threshold: link threshold, link above this value will be considered link
        low_text_threshold: low text threshold, text below this value will be ignored

    Returns:
        bounding boxes, labels, mapper
    """
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    # labeling method
    ret, text_score = cv2.threshold(textmap, low_text_threshold, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text_score_comb.astype(np.uint8), connectivity=4
    )

    det = []
    mapper = []
    for k in range(1, n_labels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # thresholding
        if np.max(textmap[labels == k]) < text_threshold:
            continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255

        # remove link area
        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0

        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = (x - niter, x + w + niter + 1, y - niter, y + h + niter + 1)
        # boundary check
        if sx < 0:
            sx = 0
        if sy < 0:
            sy = 0
        if ex >= img_w:
            ex = img_w
        if ey >= img_h:
            ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_temp = np.roll(np.array(np.where(segmap != 0)), 1, axis=0)
        np_contours = np_temp.transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # boundary check due to minAreaRect may have out of range values
        # (see https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga3d476a3417130ae5154aea421ca7ead9)
        for p in box:
            if p[0] < 0:
                p[0] = 0
            if p[1] < 0:
                p[1] = 0
            if p[0] >= img_w:
                p[0] = img_w
            if p[1] >= img_h:
                p[1] = img_h

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        start_idx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - start_idx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)

    return det, labels, mapper


def get_poly_core(boxes: List[np.ndarray],
                  labels: np.ndarray,
                  mapper: List[int]) -> List[np.ndarray]:
    """
    Generates polygon from boxes.

    Args:
        boxes: boxes
        labels: labels
        mapper: mapper, maps boxes to labels

    Returns:
        polygons
    """
    num_cp = 5
    max_len_ratio = 0.7
    expand_ratio = 1.45
    max_r = 2.0
    step_r = 0.2

    polys = []
    for k, box in enumerate(boxes):
        # size filter for small instance
        w, h = (
            int(np.linalg.norm(box[0] - box[1]) + 1),
            int(np.linalg.norm(box[1] - box[2]) + 1),
        )
        if w < 10 or h < 10:
            polys.append(None)
            continue

        # warp image
        tar = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        M = cv2.getPerspectiveTransform(box, tar)
        word_label = cv2.warpPerspective(labels, M, (w, h), flags=cv2.INTER_NEAREST)
        try:
            # inverse perspective matrix
            inv_p_m = np.linalg.inv(M)
        except Exception as e:
            polys.append(None)
            continue

        # binarization for selected label
        cur_label = mapper[k]
        word_label[word_label != cur_label] = 0
        word_label[word_label > 0] = 1

        # Polygon generation
        # find top/bottom contours
        cp = []
        max_len = -1
        for i in range(w):
            region = np.where(word_label[:, i] != 0)[0]
            if len(region) < 2:
                continue
            cp.append((i, region[0], region[-1]))
            length = region[-1] - region[0] + 1
            if length > max_len:
                max_len = length

        # pass if max_len is similar to h
        if h * max_len_ratio < max_len:
            polys.append(None)
            continue

        # get pivot points with fixed length
        tot_seg = num_cp * 2 + 1
        seg_w = w / tot_seg  # segment width
        pp = [None] * num_cp  # init pivot points
        cp_section = [[0, 0]] * tot_seg
        seg_height = [0] * num_cp
        seg_num = 0
        num_sec = 0
        prev_h = -1
        for i in range(0, len(cp)):
            (x, sy, ey) = cp[i]
            if (seg_num + 1) * seg_w <= x and seg_num <= tot_seg:
                # average previous segment
                if num_sec == 0:
                    break
                cp_section[seg_num] = [
                    cp_section[seg_num][0] / num_sec,
                    cp_section[seg_num][1] / num_sec,
                ]
                num_sec = 0

                # reset variables
                seg_num += 1
                prev_h = -1

            # accumulate center points
            cy = (sy + ey) * 0.5
            cur_h = ey - sy + 1
            cp_section[seg_num] = [
                cp_section[seg_num][0] + x,
                cp_section[seg_num][1] + cy,
            ]
            num_sec += 1

            if seg_num % 2 == 0:
                continue  # No polygon area

            if prev_h < cur_h:
                pp[int((seg_num - 1) / 2)] = (x, cy)
                seg_height[int((seg_num - 1) / 2)] = cur_h
                prev_h = cur_h

        # processing last segment
        if num_sec != 0:
            cp_section[-1] = [cp_section[-1][0] / num_sec, cp_section[-1][1] / num_sec]

        # pass if num of pivots is not sufficient or segment widh
        # is smaller than character height
        if None in pp or seg_w < np.max(seg_height) * 0.25:
            polys.append(None)
            continue

        # calc median maximum of pivot points
        half_char_h = np.median(seg_height) * expand_ratio / 2

        # calc gradiant and apply to make horizontal pivots
        new_pp = []
        for i, (x, cy) in enumerate(pp):
            dx = cp_section[i * 2 + 2][0] - cp_section[i * 2][0]
            dy = cp_section[i * 2 + 2][1] - cp_section[i * 2][1]
            if dx == 0:  # gradient if zero
                new_pp.append([x, cy - half_char_h, x, cy + half_char_h])
                continue
            rad = -math.atan2(dy, dx)
            c, s = half_char_h * math.cos(rad), half_char_h * math.sin(rad)
            new_pp.append([x - s, cy - c, x + s, cy + c])

        # get edge points to cover character heatmaps
        is_spp_found, is_epp_found = False, False
        grad_s = (pp[1][1] - pp[0][1]) / (pp[1][0] - pp[0][0]) + (
                pp[2][1] - pp[1][1]
        ) / (pp[2][0] - pp[1][0])
        grad_e = (pp[-2][1] - pp[-1][1]) / (pp[-2][0] - pp[-1][0]) + (
                pp[-3][1] - pp[-2][1]
        ) / (pp[-3][0] - pp[-2][0])
        for r in np.arange(0.5, max_r, step_r):
            dx = 2 * half_char_h * r
            if not is_spp_found:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_s * dx
                p = np.array(new_pp[0]) - np.array([dx, dy, dx, dy])
                cv2.line(
                    line_img,
                    (int(p[0]), int(p[1])),
                    (int(p[2]), int(p[3])),
                    1,
                    thickness=1,
                )
                if (
                        np.sum(np.logical_and(word_label, line_img)) == 0
                        or r + 2 * step_r >= max_r
                ):
                    spp = p
                    is_spp_found = True
            if not is_epp_found:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_e * dx
                p = np.array(new_pp[-1]) + np.array([dx, dy, dx, dy])
                cv2.line(
                    line_img,
                    (int(p[0]), int(p[1])),
                    (int(p[2]), int(p[3])),
                    1,
                    thickness=1,
                )
                if (
                        np.sum(np.logical_and(word_label, line_img)) == 0
                        or r + 2 * step_r >= max_r
                ):
                    epp = p
                    is_epp_found = True
            if is_spp_found and is_epp_found:
                break

        # pass if boundary of polygon is not found
        if not (is_spp_found and is_epp_found):
            polys.append(None)
            continue

        # make final polygon
        poly = []
        poly.append(warp_coord(inv_p_m, (spp[0], spp[1])))
        for p in new_pp:
            poly.append(warp_coord(inv_p_m, (p[0], p[1])))
        poly.append(warp_coord(inv_p_m, (epp[0], epp[1])))
        poly.append(warp_coord(inv_p_m, (epp[2], epp[3])))
        for p in reversed(new_pp):
            poly.append(warp_coord(inv_p_m, (p[2], p[3])))
        poly.append(warp_coord(inv_p_m, (spp[2], spp[3])))

        # add to final result
        polys.append(np.array(poly))

    return polys


def get_det_boxes(
        textmap: np.ndarray,
        linkmap: np.ndarray,
        text_threshold: float,
        link_threshold: float,
        low_text: float,
        poly: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extracts bounding boxes from textmap and linkmap.

    Args:
        textmap: textmap, shape (H, W)
        linkmap: linkmap, shape (H, W)
        text_threshold: text threshold, text above this value will be considered text
        link_threshold: link threshold, link above this value will be considered link
        low_text: low text threshold, text below this value will be ignored
        poly: whether to return polygons

    Returns:
        bounding boxes, polygons
    """
    boxes, labels, mapper = get_det_boxes_core(
        textmap, linkmap, text_threshold, link_threshold, low_text
    )

    if poly:
        polys = get_poly_core(boxes, labels, mapper)
    else:
        polys = [None] * len(boxes)

    return boxes, polys


def adjust_result_coordinates(polys: List[np.ndarray],
                              ratio_w: float,
                              ratio_h: float,
                              ratio_net: float = 2) -> List[np.ndarray]:
    """
    Adjusts result coordinates, i.e. scales them.

    Args:
        polys: polygons
        ratio_w: width ratio
        ratio_h: height ratio
        ratio_net: net ratio

    Returns:
        adjusted polygons
    """
    if len(polys) > 0:
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys
