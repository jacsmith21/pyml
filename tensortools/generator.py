import logging

import numpy as np
import skimage.draw

from tensortools import abc


class Generator:
    class ShapeGenerator:
        def generate(self, height, width, min_size, max_size):
            while True:
                try:
                    x = np.random.randint(height)
                    y = np.random.randint(width)
                    color = self._random_color()
                    results = self._generate(x, y, height, width, min_size, max_size, color)
                except ArithmeticError:
                    continue
                else:
                    break

            return results

        @staticmethod
        def _random_color():
            return np.random.randint(0, 255, size=3)

        @abc.abstract
        def _generate(self, r, y, height, width, min_size, max_size, color):
            pass

        @classmethod
        def random_generator(cls):
            subclasses = cls.__subclasses__()
            subclass = np.random.choice(subclasses)
            return subclass()

    class RectangleGenerator(ShapeGenerator):
        def _generate(self, r, c, height, width, min_size, max_size, color):
            available_height = height - r
            if available_height < min_size:
                raise ArithmeticError

            available_width = width - c
            if available_width < min_size:
                raise ArithmeticError

            h, w = [np.random.randint(min_size, min(size, max_size)+1) for size in [available_height, available_width]]

            mask = np.zeros([height, width, len(color)], np.uint8)
            mask[r:r + h, c:c + w] = color

            return mask, ('rectangle', [r, c, r + h, c + w])

    class CircleGenerator(ShapeGenerator):
        def _generate(self, r, c, height, width, min_size, max_size, color):
            available_radius = min(r, c, height - r, width - c)
            if available_radius < min_size:
                raise ArithmeticError

            radius = np.random.randint(min_size, min(available_radius, max_size) + 1)

            mask = np.zeros([height, width, len(color)], np.uint8)
            indices = skimage.draw.circle(r, c, radius)
            mask[indices] = color

            return mask, ('circle', (r - radius + 1, c - radius + 1, r + radius, c + radius))

    class TriangleGenerator(ShapeGenerator):
        def _generate(self, r, c, height, width, min_size, max_size, color):
            # (r, c) is the bottom left corner.
            # We're making an equilateral triangle.
            available_side = min(width - c, r + 1)
            if available_side < min_size:
                raise ArithmeticError

            side = np.random.randint(min_size, min(available_side, max_size) + 1)
            triangle_height = int(np.ceil(np.sqrt(3 / 4) * side))

            mask = np.zeros([height, width, len(color)], dtype=np.uint8)
            indices = skimage.draw.polygon([r, r - triangle_height, r],
                                           [c, c + side // 2, c + side])
            mask[indices] = color

            return mask, ('triangle', (r - triangle_height, c, r, c + side))

    def generate(self,
                 n_images,
                 height,
                 width,
                 max_shapes,
                 min_shapes=1,
                 min_size=2,
                 max_size=None,
                 shape=None,
                 allow_overlap=False):
        """
        Generates a fake object detection dataset of squares, triangles & circles!

        :param n_images: The amount images to generate.
        :param height: The height of the desired images.
        :param width: The width of the desired images.
        :param max_shapes: The max amount of shapes per image.
        :param min_shapes: The min amount of shapes per image.
        :param min_size: The min size of the shapes.
        :param max_size: The max size of the shapes.
        :param shape: The type of shape. If None, a shape is randomly chosen each time.
        :param allow_overlap: Whether or not to allow overlap.
        :return:
            images: The generates images, shape [n_images, height, width, 3].
        """
        max_size = max_size or max(height, width)

        images = []
        labels = []
        for _ in range(n_images):
            n_shapes = np.random.randint(min_shapes, max_shapes + 1)

            for _ in range(n_shapes):
                image, image_labels = self._generate_image(
                    n_shapes, height, width, min_size, max_size, shape, allow_overlap)
                images.append(image)
                labels.append(image_labels)

        if n_images == 1:
            return images[0], labels[0]
        else:
            return images, labels

    def _generate_image(self, n_shapes, width, height, min_size, max_size, shape, allow_overlap):
        image = np.ones([height, width, 3], dtype=np.uint8) * 255

        labels = []
        for _ in range(n_shapes):
            shape_generator = shape or self.ShapeGenerator.random_generator()
            mask, label = shape_generator.generate(height, width, min_size, max_size)

            if not allow_overlap and (image[mask.nonzero()] < 255).any():
                logging.info('Overlap detected. Skipping shape.')
                continue

            image += mask
            labels.append(label)

        return image, labels
