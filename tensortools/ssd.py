import numpy as np

from tensortools import stats, logging
from tensortools import utils

logger = logging.get_logger(__name__)


def generate_anchors(annotations, fm_sizes, n_clusters):
    """
    Generate anchors for an SSD model using KMeans. First, the algorithm matches each annotation with the appropriate
    feature map. It does this by calculating the highest IOU between the annotation and one feature map cell. For
    example, a feature map of [10, 10] and annotation of [0.1, 0.1] would have an IOU of 1. Next, it runs a KMeans
    algorithm for each set of annotations.

    :param annotations: The annotations, shape `[n_samples, 4]`. Each annotation should be in the following form:
    `[r1, c1, r2, c2]` where the first point is the ULC and the second point is the LRC. Each value should be in the
    range `[0, 1]`
    :param fm_sizes: The feature map sizes, shape `[n_fm, 2]`. For example, `[[13, 13], [5, 5]]`
    :param n_clusters: The number of clusters (aka the number of desired bounding boxes for each feature map).
    :return: The anchors boxes, shape `[n_fm, n_clusters, 2]`. Each value is in its normalized form in the range of
    [0, 1].
    """
    annotations = np.array(annotations)
    if annotations.min() < 0 or annotations.max() > 1:
        logger.warn('\'annotations\' are not between 0 and 1. This may or not be a problem.')

    fm_sizes = np.array(fm_sizes)
    cell_sizes = 1 / fm_sizes

    ious = [[] for _ in range(len(annotations))]
    for annotation_ious, annotation in zip(ious, annotations):
        for cell in cell_sizes:
            iou = utils.iou(cell, annotation)
            annotation_ious.append(iou)

    closest = np.argmax(ious, axis=1)
    anchors = []
    for i, fm in enumerate(fm_sizes):
        count = utils.count(closest, i)
        logger.info('{} has {} that are the closets'.format(fm, count))
        if count <= n_clusters:
            logger.warn('Feature map {} only has {} matching annotations. No anchors created.'.format(fm, count))
            anchors.append([cell_sizes[i] for _ in range(n_clusters)])
            continue

        fm_annotations = annotations[closest == i]
        logger.info('Average default box size, IOU for feature map {}: {}, {}'.format(
            fm, np.mean(fm_annotations, axis=0), utils.avg_iou(fm_annotations, [1 / fm])))
        fm_anchors = stats.k_means(fm_annotations, n_clusters)
        anchors.append(fm_anchors)

    return anchors