"""This module is used to provide the edge model for text detection."""
from typing import List
import numpy as np
import cv2
import time
from src.edgeinferencing.runtime import Engine
from src.edgeinferencing.config import EdgeModelConfig, EdgeInferencingPreProcessConfig
from src.edgeinferencing.common.preprocess_operator import create_operators, transform
from src.edgeinferencing.common.postprocess_db import DBPostProcess

EDGE_MODEL_DB_THRESHOLD = 0.3
EDGE_MODEL_DB_BOX_THRESHOLD = 0.5
EDGE_MODEL_DB_MAX_CANDIDATE = 1000
EDGE_MODEL_DB_UNCLIP_RATIO = 2
EDGE_MODEL_DB_USE_DILATION = 0
EDGE_MODEL_DB_SCORE_MODE = "fast"


class TextDetection:
    """
    Class TextDetection.
    """

    def __init__(
        self,
        model_config: EdgeModelConfig,
    ) -> None:
        """
        Initialize the TextDetection class.

        @param
        """
        self.pre_processors = create_operators(EdgeInferencingPreProcessConfig())
        self.post_process_op = DBPostProcess(
            thresh=EDGE_MODEL_DB_THRESHOLD,
            box_thresh=EDGE_MODEL_DB_BOX_THRESHOLD,
            max_candidates=EDGE_MODEL_DB_MAX_CANDIDATE,
            score_mode=EDGE_MODEL_DB_SCORE_MODE,
            unclip_ratio=EDGE_MODEL_DB_UNCLIP_RATIO,
            use_dilation=EDGE_MODEL_DB_USE_DILATION,
        )
        self.engine = Engine(model_config)
        self.model_config = model_config

    def initialize(self):
        """
        Initialize the text detection edge engine as per platform
        """
        print("Initializing Engine")
        self.engine.initialize()
        print("Finished Initializing Engine")

    def _pre_process(self, image: np.ndarray) -> np.ndarray:
        """
        Pre process image.

        @param
            image (np.ndarray): image to be pre-processed
        @return
            image (np.ndarray): pre-processed image
        """
        return cv2.resize(image, self.model_config.image_size)

    def _process(self, image: np.ndarray) -> List:
        """
        Run inference on image.

        @param
            image (np.ndarray): image to be processed
        @return
            boxes (List): list of bounding boxes
        """
        data = {"image": image}
        data = transform(data, self.pre_processors)
        img, shape_list = data
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        test_in = img.copy()
        start_time = time.time()
        output_buffer = self.engine.inference_single(test_in, profiling=False)
        time_taken = time.time() - start_time
        output_shape = list(test_in.shape)
        output_shape[1] = 1
        results = np.reshape(output_buffer[0], output_shape)
        post_proc_results = self.post_process_op(results, shape_list)
        dt_boxes = post_proc_results[0]["points"]

        print(f"Bounding boxes detected: {len(dt_boxes)}, time taken: {time_taken}")

        return dt_boxes

    def run(self, image):
        """
        Run inference on image.

        @param
            request (TextDetectionRequest): request input
        @return
            result (TextDetectionResult): result output
        """
        image = self._pre_process(image)
        boxes = self._process(image)
        return boxes
