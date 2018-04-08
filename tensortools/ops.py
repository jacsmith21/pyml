import tensorflow as tf
import numpy as np


def fully_connected(inputs,
                    units,
                    use_bias=True,
                    activation=None,
                    keep_prob=tf.constant(1.0, name='default_keep_prob'),
                    initializer=None):
    """
    A fully connected layer with dropout included.

    :param inputs: The inputs in the shape of [batch_size, previous_layer_units]
    :param units: The number of units for the layer.
    :param use_bias: Whether to use bias.
    :param activation: The TensorFlow activation function to use.
    :param keep_prob: The keep probability to use for the dropout layer.
    :param initializer: The kernal initializer.
    :return:
    """
    a = tf.layers.dense(inputs, units, use_bias=use_bias, activation=activation, kernel_initializer=initializer)
    return tf.nn.dropout(a, keep_prob)


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

