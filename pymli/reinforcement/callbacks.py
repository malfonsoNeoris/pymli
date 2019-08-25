from keras.callbacks import CallbackList as KerasCallbackList
from keras.callbacks import Callback as KerasCallBack


class Callback(KerasCallBack):
    def _set_env(self, env):
        self.env = env

    def on_episode_begin(self, episode, logs=None):
        pass

    def on_episode_end(self, episode, logs=None):
        pass

    def on_step_begin(self, step, logs=None):
        pass

    def on_step_end(self, step, logs=None):
        pass

    def on_action_begin(self, action, logs=None):
        pass

    def on_action_end(self, action, logs=None):
        pass


class CallbackList(KerasCallbackList):
    def set_env(self, env):
        for callback in self.callbacks:
            if callable(getattr(callback, '_set_env', None)):
                callback._set_env(env)

    def on_episode_begin(self, episode, logs=None):
        if logs is None:
            logs = {}
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_episode_begin', None)):
                callback.on_episode_begin(episode, logs=logs)
            elif callable(getattr(callback, 'on_epoch_begin', None)):
                callback.on_epoch_begin(episode, logs=logs)

    def on_episode_end(self, episode, logs=None):
        if logs is None:
            logs = {}
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_episode_end', None)):
                callback.on_episode_end(episode, logs=logs)
            elif callable(getattr(callback, 'on_epoch_end', None)):
                callback.on_epoch_end(episode, logs=logs)

    def on_step_begin(self, step, logs=None):
        if logs is None:
            logs = {}
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_step_begin', None)):
                callback.on_step_begin(step, logs=logs)
            elif callable(getattr(callback, 'on_batch_begin', None)):
                callback.on_batch_begin(step, logs=logs)

    def on_step_end(self, step, logs=None):
        if logs is None:
            logs = {}
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_step_end', None)):
                callback.on_step_end(step, logs=logs)
            elif callable(getattr(callback, 'on_batch_end', None)):
                callback.on_batch_end(step, logs=logs)

    def on_action_begin(self, action, logs=None):
        if logs is None:
            logs = {}
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_action_begin', None)):
                callback.on_action_begin(action, logs=logs)

    def on_action_end(self, action, logs=None):
        if logs is None:
            logs = {}
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_action_end', None)):
                callback.on_action_end(action, logs=logs)


class Visualizer(Callback):
    def on_action_end(self, action, logs=None):
        self.env.render(mode='human')
