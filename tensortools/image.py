import numpy as np
import scipy.misc


def show(image):
    image = np.array(image).astype(np.uint8)
    scipy.misc.imshow(image)
