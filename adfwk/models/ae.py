import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.losses import mean_squared_error

from sklearn.utils import check_array

from adfwk.models.base import BaseDetector
from adfwk.models.mixins import KerasSaveModelMixin
from adfwk.utils.stats_models import pairwise_distances
from adfwk.utils.decorators import only_fitted


class AutoEncoder(BaseDetector, KerasSaveModelMixin):
    def __init__(self, hidden_neurons=None,
                 hidden_activation='relu', output_activation='sigmoid',
                 loss=mean_squared_error, optimizer='adam',
                 epochs=100, batch_size=32, dropout_rate=0.2,
                 l2_regularizer=0.1, validation_size=0.1, preprocessing=True,
                 verbose=1, contamination=0.1, random_state=None):
        super(AutoEncoder, self).__init__(contamination=contamination,
                                          preprocessing=preprocessing,
                                          random_state=random_state)
        self.hidden_neurons = hidden_neurons
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l2_regularizer = l2_regularizer
        self.validation_size = validation_size
        self.preprocessing = preprocessing
        self.verbose = verbose

        self.n_samples_ = None
        self.n_features_ = None
        self.scaler_ = None
        self.encoding_dim_ = None
        self.compression_rate_ = None
        self.model_ = None
        self.history_ = None

        # default values
        if self.hidden_neurons is None:
            self.hidden_neurons = [64, 32, 32, 64]

        # Verify the network design is valid
        if not self.hidden_neurons == self.hidden_neurons[::-1]:
            raise ValueError(self.hidden_neurons, "Hidden units should be symmetric")

        self.hidden_neurons_ = self.hidden_neurons

    def _build_model(self):
        model = Sequential()
        # Input layer
        model.add(Dense(
            self.hidden_neurons_[0], activation=self.hidden_activation,
            input_shape=(self.n_features_,),
            activity_regularizer=l2(self.l2_regularizer)))
        model.add(Dropout(self.dropout_rate))

        # Additional layers
        for i, hidden_neurons in enumerate(self.hidden_neurons_, 1):
            model.add(Dense(
                hidden_neurons,
                activation=self.hidden_activation,
                activity_regularizer=l2(self.l2_regularizer)))
            model.add(Dropout(self.dropout_rate))

        # Output layers
        model.add(Dense(self.n_features_, activation=self.output_activation,
                        activity_regularizer=l2(self.l2_regularizer)))

        # Compile model
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def _build_and_fit_model(self, X, y=None):
        # Validate and complete the number of hidden neurons
        if np.min(self.hidden_neurons) > self.n_features_:
            raise ValueError("The number of neurons should not exceed "
                             "the number of features")
        self.hidden_neurons_.insert(0, self.n_features_)

        # Calculate the dimension of the encoding layer & compression rate
        self.encoding_dim_ = np.median(self.hidden_neurons)
        self.compression_rate_ = self.n_features_ // self.encoding_dim_

        # Build AE model & fit with X
        self.model_ = self._build_model()
        self.history_ = self.model_.fit(X, X,
                                        epochs=self.epochs,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        validation_split=self.validation_size,
                                        verbose=self.verbose).history
        # Reverse the operation for consistency
        self.hidden_neurons_.pop(0)
        # Predict on X itself and calculate the reconstruction error as
        # the outlier scores. Noted X_norm was shuffled has to recreate
        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        pred_scores = self.model_.predict(X_norm)
        self.decision_scores_ = pairwise_distances(X_norm, pred_scores)

        return self

    @only_fitted(['model_', 'history_'])
    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.
        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        X = check_array(X)

        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        pred_scores = self.model_.predict(X_norm)
        return pairwise_distances(X_norm, pred_scores)
