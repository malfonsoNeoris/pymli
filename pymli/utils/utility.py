import random
import warnings
import numpy as np
from sklearn.utils import column_or_1d


def invert_order(scores, method='multiplication'):
    scores = column_or_1d(scores)

    if method == 'multiplication':
        return scores.ravel() * -1
    elif method == 'subtraction':
        return (scores.max() - scores).ravel()


def zeroed_observation(observation):
    if hasattr(observation, 'shape'):
        return np.zeros(observation.shape)
    elif hasattr(observation, '__iter__'):
        out = []
        for x in observation:
            out.append(zeroed_observation(x))
        return out
    else:
        return 0.


def sample_batch_indexes(low, high, size):
    if high - low >= size:
        r = range(low, high)
        batch_idxs = random.sample(r, size)
    else:
        warnings.warn('Not enough entries to sample without replacement.'
                      'Consider increasing your warm-up phase to avoid oversampling!')
        batch_idxs = np.random.random_integers(low, high - 1, size=size)
    return batch_idxs
