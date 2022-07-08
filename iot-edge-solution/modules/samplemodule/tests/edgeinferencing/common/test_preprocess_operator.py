import unittest
from unittest.mock import MagicMock

import numpy as np
from src.edgeinferencing.config import EdgeInferencingPreProcessConfig
from src.edgeinferencing.common.preprocess_operator import (
    NormalizeImage,
    ToCHWImage,
    KeepKeys,
    DetResizeForTest,
    create_operators,
    transform,
)
from PIL import Image


class TestNormalizeImage(unittest.TestCase):
    def test_normalize_image_call_returns_normalized_image(self):
        config = {
            "scale": 1.0 / 255.0,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
        normalize_image = NormalizeImage(**config)
        image = np.array([[0, 0, 0], [0, 255, 0], [0, 0, 0]]).astype("uint8")
        image = Image.fromarray(image)
        data = {"image": image}
        normalize_image(data)
        self.assertAlmostEqual(data["image"][0, 0, 0], -2.117904, places=2)
        self.assertEqual(data["image"].shape, (3, 3, 3))
        self.assertEqual(data["image"].dtype, "float32")


class TestToCHWImage(unittest.TestCase):
    def test_to_chw_image_call_returns_chw_image(self):
        to_chw_image = ToCHWImage()
        image = np.ones((1, 2, 3)).astype("uint8")
        image = Image.fromarray(image)
        data = {"image": image}
        to_chw_image(data)
        self.assertEqual(data["image"].dtype, "uint8")
        self.assertEqual(data["image"].shape, (3, 1, 2))


class TestKeepKeys(unittest.TestCase):
    def test_keep_keys_call_returns_list_of_data(self):
        keep_keys = ["image", "shape"]
        keep_keys_operator = KeepKeys(keep_keys)
        image = np.ones((1, 2, 3)).astype("uint8")
        image = Image.fromarray(image)
        data = {"image": image, "shape": (1, 2, 3)}
        keep_keys_operator(data)
        self.assertEqual(data["shape"], (1, 2, 3))


class TestDetResizeForTest(unittest.TestCase):
    def test_det_resize_for_test_call_returns_resized_image_type0_min(self):
        det_resize_for_test = DetResizeForTest(
            target_size=300,
            max_size=600,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        image = np.ones((1, 2, 3)).astype("uint8")
        # image = Image.fromarray(image)
        data = {"image": image}
        det_resize_for_test(data)
        self.assertEqual(data["image"].dtype, "uint8")
        self.assertEqual(data["image"].shape, (736, 1472, 3))

    def test_det_resize_for_test_call_returns_resized_image_type0_max(self):
        det_resize_for_test = DetResizeForTest(
            target_size=300,
            max_size=600,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            limit_side_len=736,
            limit_type="max",
        )
        image = np.ones((1, 2, 3)).astype("uint8")
        # image = Image.fromarray(image)
        data = {"image": image}
        det_resize_for_test(data)
        self.assertEqual(data["image"].dtype, "uint8")
        self.assertEqual(data["image"].shape, (32, 32, 3))

    def test_det_resize_for_test_call_returns_resized_image_type1(self):
        det_resize_for_test = DetResizeForTest(
            target_size=300,
            max_size=600,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            image_shape=(300, 600),
        )
        image = np.ones((1, 2, 3)).astype("uint8")
        # image = Image.fromarray(image)
        data = {"image": image}
        det_resize_for_test(data)
        self.assertEqual(data["image"].dtype, "uint8")
        self.assertEqual(data["image"].shape, (300, 600, 3))

    def test_det_resize_for_test_call_returns_resized_image_type2(self):
        det_resize_for_test = DetResizeForTest(
            target_size=300,
            max_size=600,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            resize_long=960,
        )
        image = np.ones((1, 2, 3)).astype("uint8")
        # image = Image.fromarray(image)
        data = {"image": image}
        det_resize_for_test(data)
        self.assertEqual(data["image"].dtype, "uint8")
        self.assertEqual(data["image"].shape, (512, 1024, 3))


class TestPreProcessOperators(unittest.TestCase):
    def test_create_operators_if_created_successfully(self):
        edge_inferencing_pre_processing_config = EdgeInferencingPreProcessConfig()
        operators = create_operators(edge_inferencing_pre_processing_config)
        self.assertEqual(len(operators), 4)
        self.assertEqual(operators[0].__class__.__name__, "DetResizeForTest")
        self.assertEqual(operators[1].__class__.__name__, "NormalizeImage")
        self.assertEqual(operators[2].__class__.__name__, "ToCHWImage")
        self.assertEqual(operators[3].__class__.__name__, "KeepKeys")

    def test_create_operators_if_created_successfully_with_not_none_to_chw_image(self):
        edge_inferencing_pre_processing_config = EdgeInferencingPreProcessConfig()
        edge_inferencing_pre_processing_config.to_chw_image = MagicMock()
        operators = create_operators(edge_inferencing_pre_processing_config)
        self.assertEqual(len(operators), 4)
        self.assertEqual(operators[0].__class__.__name__, "DetResizeForTest")
        self.assertEqual(operators[1].__class__.__name__, "NormalizeImage")
        self.assertEqual(operators[2].__class__.__name__, "ToCHWImage")
        self.assertEqual(operators[3].__class__.__name__, "KeepKeys")

    def test_transform_if_the_image_is_transformed(self):
        edge_inferencing_pre_processing_config = EdgeInferencingPreProcessConfig()
        operators = create_operators(edge_inferencing_pre_processing_config)
        image = np.ones((1, 2, 3)).astype("uint8")
        # image = Image.fromarray(image)
        data = {"image": image}
        transform(data, operators)
        self.assertEqual(data["image"].dtype, "float32")
        self.assertEqual(data["image"].shape, (3, 736, 1472))
