import tensorflow as tf
import numpy as np


def flatten(tensor, has_batch=False):
    """
    Flattens a tensor.

    :param tensor: Tensor of shape [batch_size, dim1, dim2 ... dimn] or [dim1, dim2 ... dimn]
    :param has_batch: Whether or not the tensor has a batch dimension. If so, it is preserved.
    :return: Tensor of shape [batch_size, dim1*dim2*...*dimn] or [dim1*dim2* ... *dimn]
    """
    tensor = tf.convert_to_tensor(tensor)

    if has_batch:
        shape = tensor.shape.as_list()[1:]
        size = np.prod(shape)
        new_tensor = tf.reshape(tensor, [-1, size])
    else:
        new_tensor = tf.reshape(tensor, [-1])

    return new_tensor

