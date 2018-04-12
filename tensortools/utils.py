import os
import sys

import numpy as np
import requests

from tensortools import logging

logger = logging.get_logger(__name__)


def standardize(arr):
    """
    Normalizes an array by subtracting the mean and dividing by the standard deviation. See
    https://en.wikipedia.org/wiki/Feature_scaling.

    >>> standardize([1, 2, 3])
    array([-1.22474487,  0.        ,  1.22474487])

    :param arr: The array to normalize.
    :return: The normalized array.
    """
    arr -= np.mean(arr)
    arr /= np.std(arr)
    return arr


def iou(box_a, box_b):
    """
    Calculates the IOU between two boxes.

    For example:

    >>> iou([0.5, 0.5], [1, 1])
    0.25

    :param box_a:
    :param box_b:
    :return:
    """
    c_w, c_h = box_b
    w, h = box_a

    if c_w >= w and c_h >= h:
        intersection, union = w * h, c_w * c_h
    elif c_w >= w and c_h <= h:
        intersection, union = w * c_h, w * h + (c_w-w) * c_h
    elif c_w <= w and c_h >= h:
        intersection, union = c_w * h, w * h + c_w * (c_h-h)
    else:
        intersection, union = c_w * c_h, w * h

    return intersection / union


def avg_iou(annotations, anchors):
    """
    Calculates the average iou between the given anchors. Only the max IOU between each annotation and anchor is used
    to calculate the average.


    :param annotations: The annotations, shape [n_annotations, 2].
    :param anchors: The anchors, shape [n_anchors, 2].
    :return: The average IOU.
    """
    annotations = np.array(annotations)

    total = 0
    for annotation in annotations:
        # we calculate the iou for each annotation with each anchor
        # however we only keep the largest iou as each annotation only
        # belongs to one anchor
        total += max([iou(annotation, anchor) for anchor in anchors])

    n_annotations, _ = annotations.shape
    return total / n_annotations


def count(arr, value):
    """
    Counts the occurrences of the value in the given iterable.

    :param arr: The iterable.
    :param value: The value to count.
    :return: The number of occurrences.
    """
    total = 0
    for element in arr:
        try:
            iter(element)
            total += count(element, value)
        except TypeError:
            pass

        total += 1 if element == value else 0

    return total


def download_file(url, dst):
    """
    Downloads a file from a url to the given destination with % finished bar.

    :param url: The url.
    :param dst: The destination.
    """
    if os.path.isfile(dst):
        return

    with open(dst, "wb") as f:
        logger.info("Downloading %s to %s" % (url, dst))
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
