import random

import numpy as np

from pyml import utils


def _k_means(annotation_dims, centroids):
    num_annotations = annotation_dims.shape[0]
    prev_assignments = None

    count = 0
    while True:
        distances = []
        for i in range(num_annotations):
            distance = 1 - utils.iou(annotation_dims[i], centroids)
            distances.append(distance)

        # TODO: Check if this is even the correct summary to give!
        print("iter {}: mean = {}".format(count, np.mean(distances)))

        # assign samples to centroids
        # distances have a shape of (n_annotations, k)
        # assign the annotations to the closest cluster
        new_assignments = np.argmin(distances, axis=1)

        if (new_assignments == prev_assignments).all():
            print('Centroids have not changed since the last iteration. Finished searching!')
            print("Centroids = ", centroids)
            return centroids

        # calculate new centroids
        centroid_annotations = [list() for _ in range(len(centroids))]
        for cluster, annotation in zip(new_assignments, annotation_dims):
            centroid_annotations[cluster].append(annotation)

        for i, annotations in enumerate(centroid_annotations):
            centroids[i] = np.mean(annotations, axis=0)

        prev_assignments = np.copy(new_assignments)
        count += 1


def generate_anchors(annotations, num_clusters):
    annotations = np.array(annotations)

    indices = [random.randrange(len(annotations)) for _ in range(num_clusters)]
    initial_centroids = annotations[indices]
    return _k_means(annotations, initial_centroids)
