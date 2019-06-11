import abc

from joblib import dump, load
from keras.models import model_from_json
from keras.utils.vis_utils import plot_model

from adfwk.models.layers.kldivergence import KLDivergenceLayer
from adfwk.utils.decorators import only_fitted


class SaveModelMixin(abc.ABC):
    def __init__(self):
        self.model_ = None

    @abc.abstractmethod
    def save_model(self, path):
        pass

    @abc.abstractmethod
    def load_model(self, path):
        pass


class KerasSaveModelMixin(SaveModelMixin):

    @only_fitted(['model_', 'history_'])
    def save_model(self, path):
        plot_model(model=self.model_, show_shapes=True, to_file=path + '/model.svg')
        with open(path + '/architecture.json', 'w') as file:
            file.write(self.model_.to_json())
        self.model_.save_weights(path + '/model_.h5')

    def load_model(self, path):
        with open(path + '/architecture.json', 'r') as file:
            self.model_ = model_from_json(file.read())
        self.model_.load_weights(path + '/model_.h5')


class SKlearnSaveModelMixin(SaveModelMixin):

    @only_fitted(['model_', 'history_'])
    def save_model(self, path):
        dump(self.model_, path + '/model_.joblib')

    def load_model(self, path):
        self.model_ = load(path + '/model_.joblib')
