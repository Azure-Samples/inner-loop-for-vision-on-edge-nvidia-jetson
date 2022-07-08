import unittest

import numpy as np

from src.edgeinferencing.common.postprocess_db import DBPostProcess


class TestDBPostProcess(unittest.TestCase):
    def test_get_mini_boxes_returns_boxes_list_of_areas(self):
        dp_process = DBPostProcess()
        contour = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        box, area = dp_process.get_mini_boxes(contour)
        self.assertAlmostEqual(box[0][0], 0)
        self.assertAlmostEqual(box[0][1], 0)
        self.assertAlmostEqual(box[3][0], 0)
        self.assertAlmostEqual(box[3][1], 1)
        self.assertEqual(area, 1.0)

    def test_box_score_fast_returns_score(self):
        dp_process = DBPostProcess()
        pred = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        points = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        score = dp_process.box_score_fast(pred, points)
        self.assertAlmostEqual(score, 0.25)

    def test_box_score_slow_returns_score(self):
        dp_process = DBPostProcess()
        pred = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        points = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        score = dp_process.box_score_slow(pred, points)
        self.assertAlmostEqual(score, 0.25)

    def test_unclip_returns_expanded_box(self):
        dp_process = DBPostProcess()
        box = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        expanded_expected = np.array(
            [[2, 0], [2, 1], [1, 2], [0, 2], [-1, 1], [-1, 0], [0, -1], [1, -1]]
        )
        expanded = dp_process.unclip(box)
        self.assert_(np.all(expanded == expanded_expected))

    def test_boxes_from_bitmap_returns_boxes_list_of_areas(self):
        dp_process = DBPostProcess()
        dp_process.min_size = 0.5
        dp_process.box_thresh = 0.5
        bitmap = np.array(
            [[2, 0], [2, 1], [1, 2], [0, 2], [-1, 1], [-1, 0], [0, -1], [1, -1]]
        )
        pred = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        dest_width, dest_height = 100, 100
        boxes, areas = dp_process.boxes_from_bitmap(
            pred, bitmap, dest_width, dest_height
        )
        expected_boxes = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        self.assertEqual(len(boxes), 1)
        self.assertEqual(areas[0], 0.5)
        self.assert_(np.all(boxes[0] == expected_boxes))
