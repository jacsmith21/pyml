import numpy as np

from tensortools import logging
from tensortools import utils

logger = logging.get_logger(__name__)


def distance(annotation, centroid):
    return 1 - utils.iou(annotation, centroid)


def k_means(annotations, n_clusters):
    """

    :param annotations:
    :param n_clusters:
    :return:
    """
    prev_assignments = None
    indices = np.random.choice(range(len(annotations)), n_clusters, replace=False)
    centroids = annotations[indices]

    count = 0
    while True:
        distances = []
        for annotation in annotations:
            distances.append([distance(annotation, centroid) for centroid in centroids])

        # assign samples to centroids
        # distances have a shape of (n_annotations, k)
        # assign the annotations to the closest cluster
        new_assignments = np.argmin(distances, axis=1)

        logger.info("iter {}: mean = {}".format(count, np.mean(np.min(distances, axis=-1))))

        if (new_assignments == prev_assignments).all():
            logger.info('Centroids have not changed since the last iteration. Finished searching!')
            logger.info("Centroids = {}".format(centroids))
            return centroids

        # calculate new centroids
        centroid_annotations = [list() for _ in range(len(centroids))]
        for cluster, annotation in zip(new_assignments, annotations):
            centroid_annotations[cluster].append(annotation)

        for i, centroid_annotation in enumerate(centroid_annotations):
            centroids[i] = np.mean(centroid_annotation, axis=0)

        prev_assignments = new_assignments
        count += 1
