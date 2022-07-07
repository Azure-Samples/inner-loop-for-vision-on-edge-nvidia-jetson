"""This module is used to provide the configurations for edge text detection."""


class EdgeInferencingPreProcessConfig:
    """
    Configurations class for edge inferencing pre-processing pipeline.
    """

    def __init__(self) -> None:
        """
        Initialize the configuration.
        """
        self.det_resize_for_test = {}
        self.normalize_image = {
            "std": [0.229, 0.224, 0.225],
            "mean": [0.485, 0.456, 0.406],
            "scale": 1.0 / 255.0,
            "order": "hwc",
        }
        self.to_chw_image = None
        self.keep_keys = {"keep_keys": ["image", "shape"]}


class EdgeModelConfig:
    """
    Class for edge model configurations.
    """

    def __init__(self, original_model_path: str, engine_path: str) -> None:
        """
        Initialize the configuration.

        @param
            original_model_path (str): path to the ONNX model
            engine_path (str): path to the engine
        """
        self.original_model_path = original_model_path
        self.engine_path = engine_path
        self.image_size = (960, 960)
        self.fp16 = True
        self.dynamic_shape = True
        self.profile_config = [
            {"x": [(1, 3, 960, 960), (1, 3, 1280, 1280), (1, 3, 1536, 1536)]}
        ]
