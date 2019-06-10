import abc
import pickle
import numpy as np

from scipy import special

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler

from adfwk.utils.decorators import only_fitted


class BaseDetector(abc.ABC):

    @abc.abstractmethod
    def __init__(self, contamination=0.1, preprocessing=True, random_state=None):
        self.contamination = contamination
        self.preprocessing = preprocessing
        self.random_state = random_state
        self.decision_scores_ = None
        self.threshold_ = None
        self.n_features_ = None
        self.n_samples_ = None
        self.scaler_ = None
        self.model_ = None
        self.history_ = None
        self._mu = None
        self._sigma = None

        if self.random_state is not None:
            np.random.seed(self.random_state)
            try:
                from tensorflow import set_random_seed
                set_random_seed(self.random_state)
            except ImportError:
                pass

    @abc.abstractmethod
    def _build_and_fit_model(self, X, y=None):
        pass

    def fit(self, X, y=None):
        X = check_array(X)
        self._set_n_classes(y)

        self.n_samples_, self.n_features_ = X.shape[0], X.shape[1]

        if self.preprocessing:
            self.scaler_ = StandardScaler()
            X_norm = self.scaler_.fit_transform(X)
        else:
            X_norm = np.copy(X)

        np.random.shuffle(X_norm)

        self._build_and_fit_model(X_norm, y)
        self._process_decision_scores()

        return self

    @abc.abstractmethod
    def decision_function(self, X):
        pass

    @only_fitted(['decision_scores_', 'threshold_', 'labels_'])
    def predict(self, X):
        pred_score = self.decision_function(X)
        return (pred_score > self.threshold_).astype('int').ravel()

    @only_fitted(['decision_scores_', 'threshold_', 'labels_'])
    def predict_proba(self, X, method='linear'):
        train_scores = self.decision_scores_
        test_scores = self.decision_function(X)

        probs = np.zeros([X.shape[0], int(self._classes)])

        if method == 'linear':
            scaler = MinMaxScaler().fit(train_scores.reshape(-1, 1))

            probs[:, 1] = scaler.transform(test_scores.reshape(-1, 1)).ravel().clip(0, 1)
            probs[:, 0] = 1 - probs[:, 1]

            return probs
        elif method == 'unify':
            erf_score = special.erf((test_scores - self._mu) / (self._sigma * np.sqrt(2)))

            probs[:, 1] = erf_score.clip(0, 1).ravel()
            probs[:, 0] = 1 - probs[:, 1]

            return probs
        else:
            raise ValueError(method, 'not valid method')

    def _set_n_classes(self, y=None):
        self._classes = 2

        if y is not None:
            check_classification_targets(y)
            self._classes = len(np.unique(y))

        return self

    def _process_decision_scores(self):
        self.threshold_ = np.percentile(self.decision_scores_, 100 * (1 - self.contamination))
        self.labels_ = (self.decision_scores_ > self.threshold_).astype('int').ravel()

        self._mu = np.mean(self.decision_scores_)
        self._sigma = np.std(self.decision_scores_)

        return self

    @only_fitted(['model_', 'history_'])
    def save(self, path):
        model_ = self.model_
        self.model_ = None
        with open(path + '/model.pkl', 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
        self.model_ = model_
        self.save_model(path)

    @classmethod
    def load(cls, path):
        with open(path + '/model.pkl', 'rb') as file:
            model = pickle.load(file)
        model.load_model(path)
        return model
