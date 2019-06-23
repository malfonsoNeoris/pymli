import numpy as np
from sklearn.neighbors import NearestNeighbors

from adfwk.models.base import BaseDetector
from adfwk.models.mixins import SKlearnSaveModelMixin
from adfwk.utils.decorators import only_fitted


class KNN(BaseDetector, SKlearnSaveModelMixin):
    def __init__(self, n_neighbors=5, method='largest',
                 radius=1.0, algorithm='auto', leaf_size=30,
                 metric='minkowski', p=2, metric_params=None,
                 contamination=0.1, n_jobs=1, random_state=None,
                 preprocessing=False, verbose=0):
        super(KNN, self).__init__(contamination=contamination,
                                  preprocessing=preprocessing,
                                  random_state=random_state)
        self.n_neighbors = n_neighbors
        self.method = method
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _build_model(self):
        return NearestNeighbors(n_neighbors=self.n_neighbors,
                                radius=self.radius,
                                algorithm=self.algorithm,
                                leaf_size=self.leaf_size,
                                metric=self.metric,
                                p=self.p,
                                metric_params=self.metric_params,
                                n_jobs=self.n_jobs)

    def _build_and_fit_model(self, X, y=None):
        self.model_ = self._build_model()
        self.history_ = self.model_.fit(X=X, y=y)
        self.decision_scores_ = self.decision_function(X)
        return self

    @only_fitted(['model_', 'history_'])
    def decision_function(self, X):
        dist_array, _ = self.model_.kneighbors(X=X,
                                               n_neighbors=self.n_neighbors,
                                               return_distance=True)
        dist = self._get_dist_by_method(dist_array)
        return dist.ravel()

    def _get_dist_by_method(self, dist_array):
        if self.method == 'largest':
            return dist_array[:, -1]
        elif self.method == 'mean':
            return np.mean(dist_array, axis=1)
        elif self.method == 'median':
            return np.median(dist_array, axis=1)
        else:
            raise ValueError(self.method, 'not valid method')
