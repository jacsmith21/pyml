import json
import unittest

import numpy as np

from tensortools import ssd, logging
from tensortools import utils

logger = logging.get_logger(__name__)


class TestSSD(unittest.TestCase):
    def test_generate_anchors(self):
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

        self.assertAlmostEqual(0.35, utils.avg_iou(annotations, anchors[0]), places=1)
        self.assertAlmostEqual(0.43, utils.avg_iou(annotations, anchors[1]), places=1)

    def test_generate_anchors_with_voc_data(self):
        with open('../resources/voc_annotations.txt') as f:
            lines = f.read().splitlines()
            annotations = [list(map(float, line.split(','))) for line in lines]

        fm_sizes = [[10, 10], [5, 5]]
        num_clusters = 2
        anchors = ssd.generate_anchors(annotations, fm_sizes, num_clusters)
        self.assertAlmostEqual(len(fm_sizes), len(anchors))
        self.assertEqual(num_clusters, len(anchors[0]))
        self.assertEqual(num_clusters, len(anchors[1]))
