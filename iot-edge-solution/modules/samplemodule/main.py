# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for
# full license information.


from src.frameprovider.frame_provider import VideoCapture
from src.edgeinferencing.edge_model import TextDetection
from src.edgeinferencing.config import EdgeModelConfig
from src.common.utils import get_parent_dir_path, get_camera_path

ONNX_MODEL_FILE_NAME = "ch_pp_inf_dynamic.onnx"
ENGINE_FILE_NAME = "ch_pp_inf_dynamic_fp16.engine"


def main():
    print("Starting Application..")
    cur_dir = get_parent_dir_path()
    onnx_file_path = cur_dir + "/local_data/" + ONNX_MODEL_FILE_NAME
    engine_file_path = cur_dir + "/local_data/" + ENGINE_FILE_NAME
    config = EdgeModelConfig(onnx_file_path, engine_file_path)
    text_detection = TextDetection(config)
    text_detection.initialize()
    camera_path = get_camera_path()
    VideoCapture(camera_path, text_detection)


if __name__ == "__main__":
    main()
