import json
import unittest

import numpy as np

from tensortools import ssd, logging
from tensortools import utils

logger = logging.get_logger(__name__)


class TestSSD(unittest.TestCase):
    def test_generate_anchors(self):
        annotations = [[0, 0, 10, 10], [0, 0, 10, 20]] / np.array([20, 20, 20, 20])
        anchors = ssd.generate_anchors(annotations, [[10, 10]], 2)
        self.assertEqual(2, len(anchors))

    def test_generate_anchors_with_data(self):
        with open('../resources/annotations.json') as f:
            annotations = json.load(f)

        annotations = np.reshape(annotations, [-1, 4])
        annotations = annotations / ([1600, 640] * 2)
        annotations = annotations[:, 2] - annotations[:, 0], annotations[:, 3] - annotations[:, 1]
        annotations = np.stack(annotations, axis=1)

        fm_sizes = [[10, 10], [5, 5]]
        num_clusters = 2
        anchors = ssd.generate_anchors(annotations, fm_sizes, num_clusters)
        self.assertEqual(len(fm_sizes), len(anchors))

        self.assertAlmostEqual(0.35, utils.avg_iou(annotations, anchors[0]), places=2)
        self.assertAlmostEqual(0.43, utils.avg_iou(annotations, anchors[1]), places=2)

    def test_generate_anchors_with_voc_data(self):
        with open('../resources/voc_annotations.txt') as f:
            lines = f.read().splitlines()
            annotations = [list(map(float, line.split(','))) for line in lines]

        fm_sizes = [[10, 10], [5, 5]]
        num_clusters = 5
        anchors = ssd.generate_anchors(annotations, fm_sizes, num_clusters)
        self.assertAlmostEqual(2, len(anchors))
        self.assertEqual(num_clusters, len(anchors[0]))
        self.assertEqual(num_clusters, len(anchors[1]))

        logger.info(utils.avg_iou(annotations, anchors[0]))
        logger.info(utils.avg_iou(annotations, anchors[1]))


class TestSSDDistance(unittest.TestCase):
    def test_distance(self):
        ssd_distance = ssd.SSDDistance([[10, 10], [10, 5]])

        annotation = [1, 1]
        centroid = [0.1, 0.3]
        distance = ssd_distance.calculate(annotation, centroid)
        self.assertAlmostEqual(0.5, distance)
