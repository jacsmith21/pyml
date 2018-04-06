import os

import scipy.misc

from common.data import Label
from common.generator import Generator
from common.templates.preprocessor import BasePreprocessor
from common.utils import split_and_shuffle


class ShapePreprocessor(BasePreprocessor):
    def get_datasets(self):
        generator = Generator()
        images, labels = generator.generate(self.config.n_images,
                                            *self.config.shape,
                                            self.config.max_shapes,
                                            min_shapes=self.config.min_shapes,
                                            min_size=self.config.min_size,
                                            max_size=self.config.max_size,
                                            allow_overlap=self.config.allow_overlap)

        data = self.save_images(images, labels)
        return split_and_shuffle(data, self.config.train_valid_test_ratio)

    def save_images(self, images, labels):
        data = []
        for i, (image, image_labels) in enumerate(zip(images, labels)):
            path = os.path.join(self.config.scratch_path, '{}.png'.format(i))
            scipy.misc.imsave(path, image)
            image_labels = \
                [{'name': name, 'roi': roi} for name, roi in image_labels]

            data.append({'path': path, 'labels': image_labels})

        return data

    def extract_inputs(self, data):
        image = scipy.misc.imread(data['path'])
        return {
            'images': image
        }

    def extract_outputs(self, data):
        one_hot_labels = []
        for label_info in data['labels']:
            label = self.config.outputs['labels'][label_info['name']]
            one_hot_labels.append(label.one_hot(self.config.n_labels))

        return {
            'labels': one_hot_labels
        }
