from tensortools import utils
from tensortools import testing
from tensortools.yolo import generate_anchors


def test_generate_anchors():
    anchors = generate_anchors([[1, 1], [1, 0.9], [0.5, 0.5], [0.45, 0.45]], 2)
    assert 2 == len(anchors)


def test_generate_voc_anchors(voc_annotations):
    anchors = generate_anchors(voc_annotations, 5)
    assert 5 == len(anchors)

    assert testing.approx(0.615890, utils.avg_iou(voc_annotations, anchors), places=4)
