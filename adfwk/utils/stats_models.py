import numpy as np
from sklearn.utils.validation import check_array
from numba import njit


def pairwise_distances(X, Y):
    X = check_array(X)
    Y = check_array(Y)

    if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
        raise ValueError(
            "pairwise_distances function receive matrix with different shapes {0} and {1}".format(X.shape, Y.shape))

    return _pairwise_distances(X, Y)


@njit
def _pairwise_distances(X, Y):
    euclidean_sq = np.square(Y - X)
    return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()
