from sklearn.svm import OneClassSVM as SKLearnOneClassSVM
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array

from .base import BaseDetector
from ..utils.utility import invert_order


class OneClassSVM(BaseDetector):

    def __init__(self, kernel='rbf', degree=3, gamma='auto', coef0=0.0,
                 tol=1e-3, nu=0.5, shrinking=True, cache_size=200,
                 verbose=False, max_iter=-1, contamination=0.1,
                 preprocessing=False, random_state=None):
        super(OneClassSVM, self).__init__(contamination=contamination,
                                          preprocessing=preprocessing,
                                          random_state=random_state)
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.nu = nu
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter

    def _build_model(self):
        return SKLearnOneClassSVM(kernel=self.kernel,
                                  degree=self.degree,
                                  gamma=self.gamma,
                                  coef0=self.coef0,
                                  tol=self.tol,
                                  nu=self.nu,
                                  shrinking=self.shrinking,
                                  cache_size=self.cache_size,
                                  verbose=self.verbose,
                                  max_iter=self.max_iter)

    def _fit_model(self, X, y=None):
        self.model_.fit(X, y)

    def _decision_function(self, X):
        return invert_order(self.model_.decision_function(X))
