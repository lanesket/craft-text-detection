from typing import List, Tuple, NoReturn
import copy
import os

import cv2
import gdown
import numpy as np

from craft_text_detection.image_utils import read_image


def download(url: str, save_path: str) -> NoReturn:
    """
    Downloads file from given url and saves it to given path.

    Args:
        url: url to download file from
        save_path: path to save downloaded file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    gdown.download(url, save_path, quiet=False)


def get_files(img_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Get image, mask and ground truth files from given directory.

    Args:
        img_dir: input directory path

    Returns:
        imgs: list of image files
        masks: list of mask files
        xmls: list of ground truth files
    """
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls


def list_files(path: str) -> Tuple[List[str], List[str], List[str]]:
    """
    List files in given directory.

    Args:
        path: input directory path

    Returns:
        img_files: list of image files
        mask_files: list of mask files
        gt_files: list of ground truth files
    """
    img_files = []
    mask_files = []
    gt_files = []
    for (dir_path, dir_names, filenames) in os.walk(path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if (
                    ext == ".jpg"
                    or ext == ".jpeg"
                    or ext == ".gif"
                    or ext == ".png"
                    or ext == ".pgm"
            ):
                img_files.append(os.path.join(dir_path, file))
            elif ext == ".bmp":
                mask_files.append(os.path.join(dir_path, file))
            elif ext == ".xml" or ext == ".gt" or ext == ".txt":
                gt_files.append(os.path.join(dir_path, file))
            elif ext == ".zip":
                continue

    return img_files, mask_files, gt_files


def rectify_poly(img: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """
    Rectify detected polygon by affine transform.
    
    Args:
        img: numpy array of image
        poly: detected region polygon
        
    Returns:
        output_img: rectified region
    """
    # Use Affine transform
    n = int(len(poly) / 2) - 1
    width = 0
    height = 0
    for k in range(n):
        box = np.float32([poly[k], poly[k + 1], poly[-k - 2], poly[-k - 1]])
        width += int(
            (np.linalg.norm(box[0] - box[1]) + np.linalg.norm(box[2] - box[3])) / 2
        )
        height += np.linalg.norm(box[1] - box[2])
    width = int(width)
    height = int(height / n)

    output_img = np.zeros((height, width, 3), dtype=np.uint8)
    width_step = 0
    for k in range(n):
        box = np.float32([poly[k], poly[k + 1], poly[-k - 2], poly[-k - 1]])
        w = int((np.linalg.norm(box[0] - box[1]) + np.linalg.norm(box[2] - box[3])) / 2)

        # Top triangle
        pts1 = box[:3]
        pts2 = np.float32(
            [[width_step, 0], [width_step + w - 1, 0], [width_step + w - 1, height - 1]]
        )
        M = cv2.getAffineTransform(pts1, pts2)
        warped_img = cv2.warpAffine(
            img, M, (width, height), borderMode=cv2.BORDER_REPLICATE
        )
        warped_mask = np.zeros((height, width, 3), dtype=np.uint8)
        warped_mask = cv2.fillConvexPoly(warped_mask, np.int32(pts2), (1, 1, 1))
        output_img[warped_mask == 1] = warped_img[warped_mask == 1]

        # Bottom triangle
        pts1 = np.vstack((box[0], box[2:]))
        pts2 = np.float32(
            [
                [width_step, 0],
                [width_step + w - 1, height - 1],
                [width_step, height - 1],
            ]
        )
        M = cv2.getAffineTransform(pts1, pts2)
        warped_img = cv2.warpAffine(
            img, M, (width, height), borderMode=cv2.BORDER_REPLICATE
        )
        warped_mask = np.zeros((height, width, 3), dtype=np.uint8)
        warped_mask = cv2.fillConvexPoly(warped_mask, np.int32(pts2), (1, 1, 1))
        cv2.line(
            warped_mask, (width_step, 0), (width_step + w - 1, height - 1), (0, 0, 0), 1
        )
        output_img[warped_mask == 1] = warped_img[warped_mask == 1]

        width_step += w
        
    return output_img


def crop_poly(image: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """
    Crop detected polygon from image.
    
    Args:
        image: numpy array of image
        poly: detected region polygon
        
    Returns:
        cropped: cropped region
    """
    # points should have 1*x*2  shape
    if len(poly.shape) == 2:
        poly = np.array([np.array(poly).astype(np.int32)])

    # create mask with shape of image
    mask = np.zeros(image.shape[0:2], dtype=np.uint8)

    # method 1 smooth region
    cv2.drawContours(mask, [poly], -1, (255, 255, 255), -1, cv2.LINE_AA)
    # method 2 not so smooth region
    # cv2.fillPoly(mask, points, (255))

    # crop around poly
    res = cv2.bitwise_and(image, image, mask=mask)
    rect = cv2.boundingRect(poly)
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

    return cropped


def export_detected_region(image, poly, file_path, rectify=True) -> NoReturn:
    """
    Args:
        image: path to the image to be processed or numpy array or PIL image
        poly: detected region polygon
        file_path: path to export image
        rectify: rectify detected polygon by affine transform
    """
    if rectify:
        # rectify poly region
        result_rgb = rectify_poly(image, poly)
    else:
        result_rgb = crop_poly(image, poly)

    # export corpped region
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, result_bgr)


def export_detected_regions(
        image,
        regions,
        file_name: str = "image",
        output_dir: str = "output/",
        rectify: bool = False,
) -> List[str]:
    """
    Args:
        image: path to the image to be processed or numpy array or PIL image
        regions: detected regions
        file_name: export image file name
        output_dir: output directory
        rectify: rectify detected polygon by affine transform
    """

    # read/convert image
    image = read_image(image)

    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)

    # create crops dir
    crops_dir = os.path.join(output_dir, file_name + "_crops")
    os.makedirs(crops_dir, exist_ok=True)

    # init exported file paths
    exported_file_paths = []

    # export regions
    for ind, region in enumerate(regions):
        # get export path
        file_path = os.path.join(crops_dir, "crop_" + str(ind) + ".png")
        # export region
        export_detected_region(image, poly=region, file_path=file_path, rectify=rectify)
        # note exported file path
        exported_file_paths.append(file_path)

    return exported_file_paths


def export_extra_results(
        image,
        regions,
        heatmaps,
        file_name: str = "image",
        output_dir="output/",
        verticals=None,
        texts=None,
) -> NoReturn:
    """
    Save results of text detection.

    Args:
        image: path to the image to be processed or numpy array or PIL image
        regions: detected regions
        heatmaps: text and link score heatmaps
        file_name: export image file name
        output_dir: output directory
        verticals: verticals of detected regions
        texts: texts of detected regions
    """
    image = read_image(image)

    res_file = os.path.join(output_dir, file_name + "_text_detection.txt")
    res_img_file = os.path.join(output_dir, file_name + "_text_detection.png")
    text_heatmap_file = os.path.join(output_dir, file_name + "_text_score_heatmap.png")
    link_heatmap_file = os.path.join(output_dir, file_name + "_link_score_heatmap.png")

    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(text_heatmap_file, heatmaps["text_score_heatmap"])
    cv2.imwrite(link_heatmap_file, heatmaps["link_score_heatmap"])

    with open(res_file, "w") as f:
        for i, region in enumerate(regions):
            region = np.array(region).astype(np.int32).reshape((-1))
            strResult = ",".join([str(r) for r in region]) + "\r\n"
            f.write(strResult)

            region = region.reshape(-1, 2)
            cv2.polylines(
                image,
                [region.reshape((-1, 1, 2))],
                True,
                color=(0, 0, 255),
                thickness=2,
            )

            if texts is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.putText(
                    image,
                    "{}".format(texts[i]),
                    (region[0][0] + 1, region[0][1] + 1),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness=1,
                )
                cv2.putText(
                    image,
                    "{}".format(texts[i]),
                    tuple(region[0]),
                    font,
                    font_scale,
                    (0, 255, 255),
                    thickness=1,
                )

    cv2.imwrite(res_img_file, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
