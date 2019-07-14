from sklearn.ensemble import IsolationForest as SKLearnIsolationForest

from .base import BaseDetector
from .mixins import SKlearnSaveModelMixin
from ..utils.utility import invert_order
from ..utils.decorators import only_fitted


class IsolationForest(BaseDetector, SKlearnSaveModelMixin):
    def __init__(self, n_estimators=100,
                 max_samples='auto',
                 contamination=0.1,
                 max_features=1.,
                 bootstrap=False,
                 n_jobs=1,
                 behaviour='new',
                 random_state=None,
                 preprocessing=False,
                 verbose=0):
        super(IsolationForest, self).__init__(contamination=contamination,
                                              preprocessing=preprocessing,
                                              random_state=random_state)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.behaviour = behaviour
        self.verbose = verbose

    def _build_model(self):
        return SKLearnIsolationForest(n_estimators=self.n_estimators,
                                      max_samples=self.max_samples,
                                      contamination=self.contamination,
                                      max_features=self.max_features,
                                      bootstrap=self.bootstrap,
                                      n_jobs=self.n_jobs,
                                      random_state=self.random_state,
                                      verbose=self.verbose,
                                      behaviour=self.behaviour)

    def _fit_model(self, X, y=None):
        return self.model_.fit(X=X, y=y)

    @only_fitted(['model_', 'history_'])
    def _decision_function(self, X):
        return invert_order(self.model_.decision_function(X))
