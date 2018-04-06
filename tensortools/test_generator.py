import unittest

from tensortools.generator import Generator


class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = Generator()

    def test_generates_color_images_with_correct_shape(self):
        images, labels = self.generator.generate(100, 128, 128, max_shapes=1)

        self.assertEqual(100, len(labels))
        self.assertEqual(100, len(images))
        [self.assertSequenceEqual([128, 128, 3], image.shape) for image in images]

    def _test_shape_generate(self, shape):
        image, labels = self.generator.generate(
            n_images=1,
            width=128,
            height=128,
            max_shapes=1,
            shape=shape)

        self.assertEqual(1, len(labels))
        _, roi = labels[0]
        crop = image[roi[0]:roi[2], roi[1]:roi[3]]

        # The crop is filled.
        self.assertTrue((crop < 255).any())

        # The crop is complete.
        image[roi[0]:roi[2], roi[1]:roi[3]] = 255
        self.assertTrue((image == 255).all())

    def test_rectangle_generate(self):
        self._test_shape_generate(Generator.RectangleGenerator())

    def test_circle_generate(self):
        self._test_shape_generate(Generator.CircleGenerator())

    def test_triangle_generate(self):
        self._test_shape_generate(Generator.TriangleGenerator())
