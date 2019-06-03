from adfwk.models.base import BaseDetector
from adfwk.utils.decorators import only_fitted
from adfwk.utils.utility import invert_order

from sklearn.ensemble import IsolationForest as SKIsolationForest


class IsolationForest(BaseDetector):
    def __init__(self, n_estimators=100,
                 max_samples='auto',
                 contamination=0.1,
                 max_features=1.,
                 bootstrap=False,
                 n_jobs=1,
                 behaviour='old',
                 random_state=None,
                 preprocessing=True,
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
        return SKIsolationForest(n_estimators=self.n_estimators,
                                 max_samples=self.max_samples,
                                 contamination=self.contamination,
                                 max_features=self.max_features,
                                 bootstrap=self.bootstrap,
                                 n_jobs=self.n_jobs,
                                 random_state=self.random_state,
                                 verbose=self.verbose,
                                 behaviour='new')

    def _build_and_fit_model(self, X, y=None):
        self.model_ = self._build_model()
        self.history_ = self.model_.fit(X=X, y=y)
        self.decision_scores_ = self.decision_function(X)
        return self

    @only_fitted(['model_', 'history_'])
    def decision_function(self, X):
        return invert_order(self.model_.decision_function(X))
