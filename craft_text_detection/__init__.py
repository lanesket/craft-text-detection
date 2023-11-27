from __future__ import absolute_import

import os
from typing import Optional, List, Tuple, NoReturn, Union
import numpy as np
import craft_text_detection.craft_utils as craft_utils
import craft_text_detection.file_utils as file_utils
import craft_text_detection.image_utils as image_utils
import craft_text_detection.predict as predict
import craft_text_detection.torch_utils as torch_utils

__version__ = "0.4.3"

__all__ = [
    "read_image",
    "load_craftnet_model",
    "load_refinenet_model",
    "get_prediction",
    "export_detected_regions",
    "export_extra_results",
    "empty_cuda_cache",
    "Craft",
]

read_image = image_utils.read_image
load_craftnet_model = craft_utils.load_craftnet_model
load_refinenet_model = craft_utils.load_refinenet_model
get_prediction = predict.get_prediction
get_prediction_batch = predict.get_prediction_batch
export_detected_regions = file_utils.export_detected_regions
export_extra_results = file_utils.export_extra_results
empty_cuda_cache = torch_utils.empty_cuda_cache


class Craft:
    def __init__(
            self,
            text_threshold=0.7,
            link_threshold=0.4,
            low_text=0.4,
            device='cpu',
            long_size=1280,
            refiner=True,
            weight_path_craft_net: Optional[str] = None,
            weight_path_refine_net: Optional[str] = None,
    ):
        """
        Arguments:
            text_threshold: text confidence threshold
            link_threshold: link confidence threshold
            low_text: text low-bound score
            device: device for inference
            long_size: desired longest image size for inference
            refiner: enable link refiner
        """
        self.craft_net = None
        self.refine_net = None
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.device = device
        self.long_size = long_size
        self.refiner = refiner

        if refiner:
            self._load_refinenet_model(weight_path_refine_net)

        self._load_craftnet_model(weight_path_craft_net)

    def _load_craftnet_model(self, weight_path: Optional[str] = None) -> NoReturn:
        """
        Loads craftnet model
        """
        self.craft_net = load_craftnet_model(self.device, weight_path=weight_path)

    def _load_refinenet_model(self, weight_path: Optional[str] = None) -> NoReturn:
        """
        Loads refinenet model
        """
        self.refine_net = load_refinenet_model(self.device, weight_path=weight_path)

    def unload_craftnet_model(self) -> NoReturn:
        """
        Unloads craftnet model
        """
        self.craft_net = None
        empty_cuda_cache()

    def unload_refinenet_model(self) -> NoReturn:
        """
        Unloads refinenet model
        """
        self.refine_net = None
        empty_cuda_cache()

    def detect_text(self, image: Union[np.ndarray, List[np.ndarray]]) -> dict:
        """
        Args:
            image: file path or numpy-array or a byte stream object of an image.

        Returns:
            dict: result information with keys "masks", "boxes", "boxes_as_ratios", "polys_as_ratios", "heatmaps", "times".
                masks: lists of predicted masks 2d as bool array,
                boxes: list of coords of points of predicted boxes,
                boxes_as_ratios: list of coords of points of predicted boxes as ratios of image size,
                polys_as_ratios: list of coords of points of predicted polys as ratios of image size,
                heatmaps: visualizations of the detected characters/links,
                times: elapsed times of the submodules, in seconds
        """
        if type(image) not in [list, np.ndarray]:
            raise TypeError("image must be a list or numpy array")

        if type(image) == np.ndarray:
            image = [image]

        prediction_result = get_prediction_batch(
            images=image,
            craft_net=self.craft_net,
            refine_net=self.refine_net,
            text_threshold=self.text_threshold,
            link_threshold=self.link_threshold,
            low_text=self.low_text,
            device=self.device,
            long_size=self.long_size,
        )

        return prediction_result
