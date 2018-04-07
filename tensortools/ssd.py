import numpy as np

from tensortools import stats, logging
from tensortools import utils
from tensortools.yolo import YoloDistance

logger = logging.get_logger(__name__)


class SSDDistance:
    def __init__(self, fm_sizes):
        self.fm_sizes = np.array(fm_sizes)

    def calculate(self, annotation, centroid):
        ious = []
        for fm in self.fm_sizes:
            anchor_norm = annotation / fm
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

    fm_sizes = np.array(fm_sizes)
    anchors = [np.array([1, 1]) / fm for fm in fm_sizes]

    ious = [[] for _ in range(len(annotations))]
    for annotation_ious, annotation in zip(ious, annotations):
        for anchor in anchors:
            iou = utils.iou(anchor, annotation)
            annotation_ious.append(iou)

    closest = np.argmax(ious, axis=1)
    logger.info(utils.count(closest, 0))
    logger.info(utils.count(closest, 1))

    anchors = []
    for i, fm in enumerate(fm_sizes):
        logger.info(fm)
        fm_annotations = annotations[closest == i]
        logger.info('average: {}'.format(np.mean(fm_annotations, axis=0)))
        indices = np.random.choice(range(len(fm_annotations)), num_clusters, replace=False)
        initial_centroids = fm_annotations[indices]
        fm_anchors = stats.k_means(fm_annotations, initial_centroids, YoloDistance())
        anchors.append(fm_anchors)

    return anchors
