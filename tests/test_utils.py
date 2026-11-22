import numpy as np
from hxensemble import find_elbow

def test_original_order_is_preserved():
    x = np.array([3., 1., 2., 4.])
    y = np.array([3., 1., 4., 4.])
    idx, xx, yy = find_elbow(x, y)
    assert (idx, xx, yy) == (2, 2., 4.)
