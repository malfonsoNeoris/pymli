import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array


def average(scores, estimator_weights=None):
    scores = check_array(scores)

    if estimator_weights is not None:
        scores = np.sum(np.multiply(scores, estimator_weights), axis=1) / np.sum(estimator_weights)
        return scores.ravel()
    else:
        return np.mean(scores, axis=1).ravel()


def maximization(scores):
    scores = check_array(scores)
    return np.max(scores, axis=1).ravel()


class CombinationModels(object):
    def __init__(self, data, models,
                 method=average,
                 method_params=None,
                 preprocessing=False):

        self.preprocessing = preprocessing
        self.method = method
        self.method_params = method_params
        self.models = models
        self._train_scores = np.zeros([data.shape[0], len(models)])
        self._thresholds = np.zeros(len(models))
        self._models_scalers = []
        self.scaler_ = None

        for i in range(len(models)):
            model = self.models[i]
            ss = StandardScaler()

            self._train_scores[:, i] = ss.fit_transform(model.decision_function(data).reshape(-1, 1)).flatten()
            self._thresholds[i] = ss.transform([[model.threshold_]])
            self._models_scalers.insert(i, ss)

    def decision_function(self, X):
        X = check_array(X)

        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        scores = np.zeros([X.shape[0], len(self.models)])

        for i in range(len(self.models)):
            model = self.models[i]
            model_scaler = self._models_scalers[i]
            scores[:, i] = model_scaler.transform(model.decision_function(X_norm).reshape(-1, 1)).flatten()

        if self.method_params is not None:
            return self.method(scores, **self.method_params).reshape(-1, 1)
        else:
            return self.method(scores).reshape(-1, 1)

    @property
    def threshold_(self):
        if self.method_params is not None:
            return self.method([self._thresholds], **self.method_params)[0]
        else:
            return self.method([self._thresholds])[0]
