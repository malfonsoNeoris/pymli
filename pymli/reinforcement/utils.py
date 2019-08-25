import keras.backend as K
import numpy as np
from keras.models import model_from_config


def get_object_config(o):
    if o is None:
        return None

    config = {
        'class_name': o.__class__.__name__,
        'config': o.get_config()
    }
    return config


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


def huber_loss(y_true, y_pred, clip_value):
    x = y_true - y_pred
    if np.isinf(clip_value):
        return .5 * K.square(x)

    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)

    if K.backend() == 'tensorflow':
        import tensorflow as tf
        return tf.where(condition, squared_loss, linear_loss)


def clone_model(model, custom_objects=None):
    if custom_objects is None:
        custom_objects = {}
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone
