import numpy as np

from tensortools import stats


def generate_anchors(annotations, num_clusters):
    annotations = np.array(annotations)
    return stats.k_means(annotations, num_clusters)
