import unittest

from pyml import utils
from pyml.yolo import generate_anchors


class TestYolo(unittest.TestCase):
    def test_generate_anchors(self):
        anchors = generate_anchors([[1, 1], [1, 0.9], [0.5, 0.5], [0.45, 0.45]], 2)
        self.assertEqual(2, len(anchors))

    def test_generate_voc_anchors(self):
        with open('../resources/voc_annotations.txt') as f:
            lines = f.read().splitlines()
            annotations = [list(map(float, line.split(','))) for line in lines]

        anchors = generate_anchors(annotations, 5)
        self.assertEqual(5, len(anchors))

        self.assertAlmostEqual(0.615890, utils.avg_iou(annotations, anchors), places=4)
