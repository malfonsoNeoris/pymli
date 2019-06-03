from sklearn.utils import column_or_1d


def invert_order(scores, method='multiplication'):
    scores = column_or_1d(scores)

    if method == 'multiplication':
        return scores.ravel() * -1
    elif method == 'subtraction':
        return (scores.max() - scores).ravel()
