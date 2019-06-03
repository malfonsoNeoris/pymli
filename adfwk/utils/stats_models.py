import numpy as np
import keras.backend as K
from sklearn.utils.validation import check_array


def pairwise_distances(X, Y):
    X = check_array(X)
    Y = check_array(Y)

    if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
        raise ValueError(
            "pairwise_distances function receive matrix with different shapes {0} and {1}".format(X.shape, Y.shape))

    euclidean_sq = np.square(Y - X)
    return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()


def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
