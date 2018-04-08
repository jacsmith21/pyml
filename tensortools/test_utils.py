from tensortools import utils


def test_count():
    arr = [1, 2, 3, 4, 3, 4, 3, 2, 1]
    assert 2 == utils.count(arr, 1)
    assert 3 == utils.count(arr, 3)

    arr = [[1, 2], [2, 2]]
    assert 3 == utils.count(arr, 2)
