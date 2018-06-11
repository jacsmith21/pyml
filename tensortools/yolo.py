import numpy as np

from tensortools import stats


def generate_anchors(annotations, num_clusters):
    """
    Generate anchors for YOLO V2 & V3 using the algorithm described in the paper (KMeans).

    :param annotations: The annotations, shape `[n_annotations, 2]`.
    :param num_clusters: The amount of clusters to create.
    :return: The centroids (annotations) that best cluster the given annotations, shape `[num_clusters, 2]`.
    """
    annotations = np.array(annotations)
    return stats.k_means(annotations, num_clusters)
