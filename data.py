from enum import Enum

import tensorflow as tf
import numpy as np


class Data:
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'

    def __init__(self, dtype, raw_shape):
        self.dtype = dtype
        self.raw_shape = raw_shape

    def encode(self, value):
        if self.dtype != tf.string:
            if np.prod(value.shape) != np.prod(self.raw_shape):
                raise ValueError('Shape mismatch. Given {} expected {}'.format(value.shape, self.raw_shape))
            value = np.array(value).flatten().tolist()

        if self.dtype == tf.float32:
            return self._float32_feature_list(value)
        else:
            raise NotImplementedError('Encoding for {} not supported.'.format(self.dtype))

    def decode(self):
        return tf.FixedLenFeature(self.raw_shape, self.dtype)

    @staticmethod
    def _float32_feature_list(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class Label(Data):
    def __init__(self, labels):
        super().__init__(tf.float32, [len(labels)])


class Stream(Data):
    def __init__(self, length):
        super().__init__(tf.float32, [length])
