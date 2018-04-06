import unittest

import numpy as np

from tensortools import ssd


class TestSSD(unittest.TestCase):
    def test_generate_anchors(self):
        annotations = [[0, 0, 10, 10], [0, 0, 10, 20]] / np.array([20, 20, 20, 20])
        anchors = ssd.generate_anchors(annotations, 2)
        self.assertEqual(2, len(anchors))


class TestSSDDistance(unittest.TestCase):
    def test_distance(self):
        ssd_distance = ssd.SSDDistance([[10, 10], [10, 5]])

        annoation = [1, 1]
        centroid = [0.1, 0.3]
        distance = ssd_distance.calculate(annoation, centroid)
        self.assertAlmostEqual(0.5, distance)
