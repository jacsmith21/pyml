import os
import sys

import numpy as np
import requests


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


def avg_iou(annotations, centroids):
    """

    :param annotations:
    :param centroids:
    :return:
    """
    annotations = np.array(annotations)

    total = 0
    for annotation in annotations:
        # note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective,
        # but I am too lazy
        total += max(iou(annotation, centroids))

    n_annotations, _ = annotations.shape
    return total / n_annotations


def download_file(url, dst):
    if os.path.isfile(dst):
        return

    with open(dst, "wb") as f:
        print("Downloading %s to %s" % (url, dst))
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                sys.stdout.flush()
