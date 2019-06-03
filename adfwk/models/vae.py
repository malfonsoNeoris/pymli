import numpy as np

import keras.backend as K
from keras.layers import Input, Dense, Lambda, Add, Multiply
from keras.models import Model, Sequential

from sklearn.utils import check_array

from adfwk.models.layers.kldivergence import KLDivergenceLayer
from adfwk.models.base import BaseDetector
from adfwk.utils.stats_models import pairwise_distances, nll
from adfwk.utils.decorators import only_fitted


class VariationalAutoEncoder(BaseDetector):
    def __init__(self, intermediate_dim=64, latent_dim=32,
                 hidden_activation='relu', output_activation='sigmoid',
                 optimizer='rmsprop',
                 epochs=100, batch_size=32,
                 validation_size=0.1, preprocessing=True,
                 verbose=1, contamination=0.1, random_state=None):
        super(VariationalAutoEncoder, self).__init__(contamination=contamination,
                                                     preprocessing=preprocessing,
                                                     random_state=random_state)
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.preprocessing = preprocessing
        self.verbose = verbose

    def _build_model(self):
        x = Input(shape=(self.n_features_,))
        h = Dense(self.intermediate_dim, activation=self.hidden_activation)(x)

        z_mu = Dense(self.latent_dim)(h)
        z_log_var = Dense(self.latent_dim)(h)

        z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
        z_sigma = Lambda(lambda t: K.exp(.5 * t))(z_log_var)

        eps = Input(tensor=K.random_normal(shape=(K.shape(x)[0], self.latent_dim), seed=self.random_state))
        z_eps = Multiply()([z_sigma, eps])
        z = Add()([z_mu, z_eps])

        decoder = Sequential([
            Dense(self.intermediate_dim, input_dim=self.latent_dim, activation=self.hidden_activation),
            Dense(self.n_features_, activation=self.output_activation)
        ])

        x_pred = decoder(z)
        vae = Model(inputs=[x, eps], outputs=x_pred, name='vae')

        vae.compile(optimizer=self.optimizer, loss=nll)

        return vae

    def _build_and_fit_model(self, X, y=None):
        if self.intermediate_dim > self.n_features_:
            raise ValueError("The number of neurons should not exceed the number of features")

        self.model_ = self._build_model()
        self.history_ = self.model_.fit(X, X,
                                        epochs=self.epochs,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        validation_split=self.validation_size,
                                        verbose=self.verbose).history

        self.decision_scores_ = self.decision_function(X)
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
