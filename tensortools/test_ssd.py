import numpy as np

from tensortools import ssd, logging
from tensortools import utils
from tensortools import testing

logger = logging.get_logger(__name__)


def test_generate_anchors(random_annotations):
    annotations = np.reshape(random_annotations, [-1, 4])
    annotations = annotations / ([1600, 640] * 2)
    annotations = annotations[:, 2] - annotations[:, 0], annotations[:, 3] - annotations[:, 1]
    annotations = np.stack(annotations, axis=1)

    fm_sizes = [[10, 10], [5, 5]]
    num_clusters = 2
    anchors = ssd.generate_anchors(annotations, fm_sizes, num_clusters)

    assert len(fm_sizes) == len(anchors)
    assert testing.approx(0.35, utils.avg_iou(annotations, anchors[0]), places=1)
    assert testing.approx(0.43, utils.avg_iou(annotations, anchors[1]), places=1)


def test_generate_anchors_with_voc_data(voc_annotations):
    fm_sizes = [[10, 10], [5, 5]]
    num_clusters = 2
    anchors = ssd.generate_anchors(voc_annotations, fm_sizes, num_clusters)

    assert len(fm_sizes) == len(anchors)
    assert num_clusters == len(anchors[0])
    assert num_clusters == len(anchors[1])
