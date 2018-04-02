import numpy as np


def iou(x, centroids):
    ious = []
    for centroid in centroids:
        c_w, c_h = centroid
        w, h = x

        if c_w >= w and c_h >= h:
            intersection, union = w * h, c_w * c_h
        elif c_w >= w and c_h <= h:
            intersection, union = w * c_h, w * h + (c_w-w) * c_h
        elif c_w <= w and c_h >= h:
            intersection, union = c_w * h, w * h + c_w * (c_h-h)
        else:
            intersection, union = c_w * c_h, w * h

        ious.append(intersection / union)

    return np.array(ious)


def avg_iou(annotation_dims, centroids):
    """

    :param annotation_dims:
    :param centroids:
    :return:
    """
    total = 0
    for annotation in annotation_dims:
        # note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective,
        # but I am too lazy
        total += max(iou(annotation, centroids))

    n_annotations, _ = annotation_dims.shape
    return total / n_annotations
