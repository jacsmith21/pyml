import unittest

from yolo import generate_anchors


class TestYolo(unittest.TestCase):
    def test_generate_anchors(self):
        anchors = generate_anchors([[1, 1], [1, 0.9], [0.5, 0.5], [0.45, 0.45]], 2)
        self.assertEqual(2, len(anchors))
