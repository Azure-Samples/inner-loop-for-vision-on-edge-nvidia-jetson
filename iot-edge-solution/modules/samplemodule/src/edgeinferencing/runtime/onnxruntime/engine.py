"""This module is used to provide the local text detection engine without CUDA and TensorRT and it can run on AMD."""
from pathlib import Path
from typing import List
from src.edgeinferencing.config import EdgeModelConfig
from onnxruntime import InferenceSession, SessionOptions

import numpy as np


class Engine:
    """
    Engine class using ONNX Runtime
    """

    def __init__(self, config: EdgeModelConfig) -> None:
        """
        Initialize Engine

        @param
            config (EdgeModelConfig): configuration for the engine
            logger (Logger): logger for the engine
        """
        self._original_model_path = config.original_model_path

    def initialize(self):
        """
        initialize the engine
        """
        self.get_engine()

    def get_engine(self, profiling=False) -> None:
        """
        get inference engine

        @param
            profiling (bool): profiling flag
        """
        if not Path(self._original_model_path).exists():
            print(
                "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(
                    self._original_model_path
                )
            )
            raise ValueError(f"ONNX file not found at {self._original_model_path}")
        print("Loading ONNX file from path {}...".format(self._original_model_path))

        so = SessionOptions()
        so.enable_profiling = profiling
        self.sess = InferenceSession(
            self._original_model_path, providers=["CPUExecutionProvider"]
        )

        self.outputs = self.sess.get_outputs()
        self.output_names = list(map(lambda output: output.name, self.outputs))
        self.input_name = self.sess.get_inputs()[0].name

    def inference_single(self, input_data: np.ndarray, profiling: bool = False) -> List:
        """
        perform inference on the given input data
        note the input data is not a list

        @param
            input_data (np.ndarray): input data to be predict on (after preprocessing)
            profiling (bool): profiling flag
        @return
            List: list of prediction output data
        """

        if not self.sess:
            self.get_engine(profiling)

        detections = self.sess.run(self.output_names, {self.input_name: input_data})

        return detections
