import numpy as np


def to_dense_array(data, fill_value):
    if np.ma.isMaskedArray(data):
        return np.ma.filled(data, fill_value)

    return np.asarray(data)
