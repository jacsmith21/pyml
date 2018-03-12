import logging
import time
import os

import numpy as np
import tensorflow as tf


def get_logger(log_name, logging_level=logging.DEBUG):
    logger = logging.getLogger(log_name)
    formatter = logging.Formatter('[%(levelname)s/%(asctime)s/%(module)s:%(lineno)s] - %(message)s')
    formatter.converter = time.gmtime

    logger.setLevel(logging_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def make_summaries(*scalars):
    for key in scalars:
        tf.summary.scalar(key, scalars[key])


def one_hot(labels):
    one_hot_encoding = np.zeros(labels.size, labels.max + 1)
    return one_hot_encoding[np.arange(labels.size), labels]


def create_tfrecord(meta_dataset, samples, datatype, directory):
    tfrecord_path = os.path.join(directory, '{}.tfrecord'.format(datatype))

    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    for sample in samples:

        features = {}
        for key in sample:
            features[key] = meta_dataset[key].encode(sample[key])

        example = tf.train.Example(features=tf.train.Features(feature=features))

        writer.write(example.SerializeToString())

    writer.close()


def cast(arr, dtype):
    return arr.astype(dtype)


def split_dataset(dataset, train_valid_test_ratio):
    num_samples = len(dataset[dataset.keys()[0]])

    shuffled_indices = np.arange(num_samples)
    np.random.shuffle(num_samples)

    num_per_set = cast(train_valid_test_ratio * num_samples, np.int64)
    train_indices, valid_indices, test_indices = \
        [shuffled_indices[np.sum(num_per_set[:i]):np.sum(num_per_set[:i+1])] for i in [0, 1, 2]]

    return dataset[train_indices], dataset[valid_indices], dataset[train_indices]
