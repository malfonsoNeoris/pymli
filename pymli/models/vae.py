import pickle
import keras.backend as K

from keras import Sequential
from keras.layers import Input, Dense, Lambda, Multiply, Add
from keras.models import Model, model_from_json
from keras.utils.vis_utils import plot_model
from keras.regularizers import l2

from .base import BaseDetector
from .layers.kldivergence import KLDivergenceLayer
from ..utils.stats_models import pairwise_distances
from ..utils.decorators import only_fitted


class VariationalAutoEncoder(BaseDetector):
    def __init__(self, intermediate_dim=64, latent_dim=32,
                 hidden_activation='relu', output_activation='sigmoid',
                 l2_regularizer=0.1, optimizer='rmsprop',
                 epochs=100, batch_size=32,
                 validation_size=0.1, preprocessing=False,
                 verbose=1, contamination=0.1, random_state=None):
        super(VariationalAutoEncoder, self).__init__(contamination=contamination,
                                                     preprocessing=preprocessing,
                                                     random_state=random_state)
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.l2_regularizer = l2_regularizer
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.preprocessing = preprocessing
        self.verbose = verbose

        self.encoder_ = None
        self.decoder_ = None

    def _build_model(self):
        if self.intermediate_dim > self.n_features_:
            raise ValueError("The number of neurons should not exceed the number of features")

        x = Input(shape=(self.n_features_,))
        h = Dense(self.intermediate_dim,
                  activation=self.hidden_activation,
                  activity_regularizer=l2(self.l2_regularizer))(x)

        z_mu = Dense(self.latent_dim, activity_regularizer=l2(self.l2_regularizer))(h)
        z_log_var = Dense(self.latent_dim, activity_regularizer=l2(self.l2_regularizer))(h)

        z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
        z_sigma = Lambda(lambda t: K.exp(.5 * t))(z_log_var)

        eps = Input(tensor=K.random_normal(shape=(K.shape(x)[0], self.latent_dim), seed=self.random_state))
        z_eps = Multiply()([z_sigma, eps])
        z = Add()([z_mu, z_eps])

        decoder = Sequential([
            Dense(self.intermediate_dim,
                  input_dim=self.latent_dim,
                  activation=self.hidden_activation,
                  activity_regularizer=l2(self.l2_regularizer)),
            Dense(self.n_features_,
                  activation=self.output_activation,
                  activity_regularizer=l2(self.l2_regularizer))
        ])

        x_pred = decoder(z)

        def nll(y_true, y_pred):
            return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

        vae = Model(inputs=[x, eps], outputs=x_pred, name='vae')
        vae.compile(optimizer=self.optimizer, loss=nll)

        if self.verbose > 0:
            print(vae.summary())

        self.decoder_ = decoder
        self.encoder_ = Model(x, z_mu)

        return vae

    def _fit_model(self, X, y=None):
        return self.model_.fit(X, X,
                               epochs=self.epochs,
                               batch_size=self.batch_size,
                               validation_split=self.validation_size,
                               verbose=self.verbose).history

    @only_fitted(['model_', 'history_'])
    def _decision_function(self, X):
        pred_scores = self.decoder_.predict(self.encoder_.predict(X, batch_size=self.batch_size),
                                            batch_size=self.batch_size)
        return pairwise_distances(X, pred_scores)

    @only_fitted(['model_', 'history_'])
    def save(self, path):
        model_ = self.model_
        encoder_ = self.encoder_
        decoder_ = self.decoder_
        self.model_ = None
        self.encoder_ = None
        self.decoder_ = None
        with open(path + '/model.pkl', 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
        self.model_ = model_
        self.encoder_ = encoder_
        self.decoder_ = decoder_
        self.save_model(path)

    @only_fitted(['model_', 'history_'])
    def save_model(self, path):
        plot_model(model=self.model_, show_shapes=True, to_file=path + '/model_.svg')
        plot_model(model=self.encoder_, show_shapes=True, to_file=path + '/encoder_.svg')
        plot_model(model=self.decoder_, show_shapes=True, to_file=path + '/decoder_.svg')
        with open(path + '/architecture_model_.json', 'w') as file:
            file.write(self.model_.to_json())
        with open(path + '/architecture_encoder_.json', 'w') as file:
            file.write(self.encoder_.to_json())
        with open(path + '/architecture_decoder_.json', 'w') as file:
            file.write(self.decoder_.to_json())
        self.model_.save_weights(path + '/model_.h5')
        self.encoder_.save_weights(path + '/encoder_.h5')
        self.decoder_.save_weights(path + '/decoder_.h5')

    def load_model(self, path):
        with open(path + '/architecture_model_.json', 'r') as file:
            self.model_ = model_from_json(file.read(), custom_objects={'KLDivergenceLayer': KLDivergenceLayer})
        with open(path + '/architecture_encoder_.json', 'r') as file:
            self.encoder_ = model_from_json(file.read(), custom_objects={'KLDivergenceLayer': KLDivergenceLayer})
        with open(path + '/architecture_decoder_.json', 'r') as file:
            self.decoder_ = model_from_json(file.read())
        self.model_.load_weights(path + '/model_.h5')
        self.encoder_.load_weights(path + '/encoder_.h5')
        self.decoder_.load_weights(path + '/decoder_.h5')
