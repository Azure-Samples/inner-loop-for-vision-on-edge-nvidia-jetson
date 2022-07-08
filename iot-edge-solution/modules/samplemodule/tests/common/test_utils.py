import unittest
from unittest.mock import patch

from src.common.utils import get_parent_dir_path, get_camera_path


class TestUtils(unittest.TestCase):
    def test_get_parent_dir_path(self):
        with patch("os.getcwd", return_value="test_dir"):
            assert get_parent_dir_path() == "test_dir"

    def test_get_camera_path(self):
        with patch("os.environ.get", return_value="test_dir"):
            assert get_camera_path() == "test_dir"
