import numpy as np

from tensortools import ops


def test_flatten(sess):
    f1 = ops.flatten([[8, 7, 6], [1, 2, 4]], has_batch=False)
    f2 = ops.flatten([[[1, 2], [3, 4]]], has_batch=True)

    np.testing.assert_array_equal([8, 7, 6, 1, 2, 4], sess.run(f1))
    np.testing.assert_array_equal([[1, 2, 3, 4]], sess.run(f2))
