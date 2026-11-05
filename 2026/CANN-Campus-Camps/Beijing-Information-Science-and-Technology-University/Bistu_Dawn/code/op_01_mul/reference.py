import numpy as np


def impl(x, y):
    dtype = x.dtype
    return (x.astype(np.float64) * y.astype(np.float64)).astype(dtype)
