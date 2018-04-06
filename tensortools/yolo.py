import numpy as np

from tensortools import stats
from tensortools import utils


class YoloDistance:
    @staticmethod
    def calculate(annotation, centroid):
        return 1 - utils.iou(annotation, centroid)


def generate_anchors(annotations, num_clusters):
    annotations = np.array(annotations)

    indices = np.random.choice(range(len(annotations)), num_clusters, replace=False)
    initial_centroids = annotations[indices]
    return stats.k_means(annotations, initial_centroids, YoloDistance())
