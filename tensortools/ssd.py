import random

import numpy as np

from tensortools import stats
from tensortools import utils


class SSDDistance:
    def __init__(self, fm_sizes):
        self.fm_sizes = fm_sizes

    def calculate(self, centroid, anchor):
        fm_sizes = np.array(self.fm_sizes)

        ious = []
        for fm in fm_sizes:
            anchor_norm = anchor / fm
            ious.append(utils.iou(anchor_norm, centroid))
        return 1 - np.mean(ious)


def generate_anchors(annotations, fm_sizes, num_clusters):
    """
    Generate anchors for an SSD model.

    :param annotations: The annotations, shape [n_samples, 4]. Each annotation should be in the following form:
    [r1, c1, r2, c2] where the first point is the ULC and the second point is the LRC. Each value should be in the range
    [0, 1]
    :param fm_sizes:
    :param num_clusters: The number of clusters (aka the number of desired bounding boxes).
    :return: The anchors boxes, shape [num_clusters, 2].
    """
    annotations = np.array(annotations)
    if annotations.min() < 0 or annotations.max() > 1:
        raise ValueError('\'annotations\' should be between 0 and 1')

    indices = [random.randrange(len(annotations)) for _ in range(num_clusters)]
    initial_centroids = annotations[indices]
    return stats.k_means(annotations, initial_centroids, SSDDistance(fm_sizes))


