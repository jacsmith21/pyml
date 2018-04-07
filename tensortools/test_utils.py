import unittest

from tensortools import utils


class TestUtils(unittest.TestCase):
    def test_count(self):
        arr = [1, 2, 3, 4, 3, 4, 3, 2, 1]
        self.assertEqual(2, utils.count(arr, 1))
        self.assertEqual(3, utils.count(arr, 3))

        arr = [[1, 2], [2, 2]]
        self.assertEqual(3, utils.count(arr, 2))
