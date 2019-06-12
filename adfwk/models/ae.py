import numpy as np
from keras import Input, Model

from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.utils import check_array

from adfwk.models.base import BaseDetector
from adfwk.utils.stats_models import pairwise_distances
from adfwk.utils.decorators import only_fitted


class AutoEncoder(BaseDetector):
    def __init__(self, intermediate_dim=64, latent_dim=32,
                 hidden_activation='relu', output_activation='sigmoid',
                 loss='binary_crossentropy', optimizer='rmsprop',
                 epochs=100, batch_size=32, dropout_rate=0.2,
                 l2_regularizer=0.1, validation_size=0.1, preprocessing=True,
                 verbose=1, contamination=0.1, random_state=None):
        super(AutoEncoder, self).__init__(contamination=contamination,
                                          preprocessing=preprocessing,
                                          random_state=random_state)
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
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

        self.encoder_ = None
        self.decoder_ = None

    def _build_model(self):
        inputs = Input(shape=(self.n_features_,))
        encoded_intermediate = Dense(self.intermediate_dim, activation=self.hidden_activation)(inputs)
        encoded_intermediate = Dropout(0.2)(encoded_intermediate)
        encoded_latent = Dense(self.latent_dim, activation=self.hidden_activation)(encoded_intermediate)

        decoder = Sequential([
            Dense(self.intermediate_dim, input_dim=self.latent_dim, activation=self.hidden_activation),
            Dropout(0.2),
            Dense(self.n_features_, activation=self.output_activation)
        ])

        outputs = decoder(encoded_latent)

        self.encoder_ = Model(inputs, encoded_latent)
        self.decoder_ = decoder

        model = Model(inputs, outputs)
        model.compile(loss=self.loss, optimizer=self.optimizer)

        print(model.summary())

        return model

    def _build_and_fit_model(self, X, y=None):
        if self.intermediate_dim > self.n_features_:
            raise ValueError("The number of neurons should not exceed the number of features")

        self.compression_rate_ = self.n_features_ // self.latent_dim

        self.model_ = self._build_model()
        self.history_ = self.model_.fit(X, X,
                                        epochs=self.epochs,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        validation_split=self.validation_size,
                                        verbose=self.verbose).history

        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        pred_scores = self.model_.predict(X_norm)
        self.decision_scores_ = pairwise_distances(X_norm, pred_scores)

        return self

    @only_fitted(['model_', 'history_'])
    def decision_function(self, X):
        X = check_array(X)

        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        pred_scores = self.model_.predict(X_norm)
        return pairwise_distances(X_norm, pred_scores)
