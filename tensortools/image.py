import numpy as np
import scipy.misc


def show(image):
    """
    Displays the image in a popup window.

    :param image: The image.
    """
    image = np.array(image).astype(np.uint8)
    scipy.misc.imshow(image)
