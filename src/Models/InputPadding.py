import numpy as np


def pad_to_length(X: list[np.ndarray], length: int) -> list[np.ndarray]:
    """Given a list of 2D arrays with the same first dimension, pad them to the same size"""
    if len(X) == 0:
        return []

    if any(len(x.shape) != 2 for x in X):
        raise ValueError("All input arrays must be 2D")

    width = X[0].shape[0]
    if any(x.shape[0] != width for x in X):
        raise ValueError("All input arrays must have the same width")

    return [np.pad(x, ((0, 0), (0, length - x.shape[1])), mode='constant') for x in X]
