import abc
import numpy as np
from copy import deepcopy
from keras.callbacks import History

from .callbacks import CallbackList, Visualizer


class Agent(abc.ABC):
    def __init__(self, processor=None):
        self.processor = processor if processor is not None else Processor()
        self.training = False
        self.compiled = False
        self.step = 0

    def fit(self, env, n_steps, action_repetition=1, n_max_episode_steps=None, callbacks=None, visualize=True):
        if not self.compiled:
            raise RuntimeError('Your tried to fit your agent but it hasn\'t been compiled yet. '
                               'Please call `compile()` before `fit()`.')

        self.training = True
        history = History()

        callbacks = self._configure_callbacks(callbacks, env, history, n_steps, visualize)
        callbacks.on_train_begin()

        self.step = np.int16(0)
        episode = np.int16(0)
        observation = None
        episode_step = np.int16(0)
        episode_reward = np.float32(0)

        while self.step < n_steps:
            if observation is None:
                callbacks.on_episode_begin(episode)
                episode_step = np.int16(0)
                episode_reward = np.float32(0)

                self.reset_states()

                observation = deepcopy(env.reset())
                observation = self.processor.process_observation(observation)

            callbacks.on_step_begin(episode_step)
            action = self.forward(observation)
            action = self.processor.process_action(action)

            reward = np.float32(0)
            accumulated_info = {}
            done = False
            for _ in range(action_repetition):
                callbacks.on_action_begin(action)
                observation, r, done, info = env.step(action)
                observation = deepcopy(observation)
                observation, r, done, info = self.processor.process_step(observation, r, done, info)

                for key, value in info.items():
                    if not np.isreal(value):
                        continue
                    if key not in accumulated_info:
                        accumulated_info[key] = np.zeros_like(value)
                    accumulated_info[key] += value
                callbacks.on_action_end(action)
                reward += r
                if done:
                    break

            if n_max_episode_steps and episode_step >= n_max_episode_steps - 1:
                done = True

            metrics = self.backward(reward, terminal=done)
            episode_reward += reward

            step_logs = {
                'action': action,
                'observation': observation,
                'reward': reward,
                'metrics': metrics,
                'episode': episode,
                'info': accumulated_info,
            }
            callbacks.on_step_end(episode_step, step_logs)

            episode_step += 1
            self.step += 1

            if done:
                self.forward(observation)
                self.backward(0., terminal=False)

                episode_logs = {
                    'episode_reward': episode_reward,
                    'n_episode_steps': episode_step,
                    'n_steps': self.step,
                }
                callbacks.on_episode_end(episode, episode_logs)

                episode += 1
                observation = None
                episode_step = None
                episode_reward = None

        callbacks.on_train_end()

        return history

    def test(self, env, n_episodes=1, action_repetition=1, callbacks=None, visualize=True, n_max_episode_steps=None):
        if not self.compiled:
            raise RuntimeError('Your tried to test your agent but it hasn\'t been compiled yet. '
                               'Please call `compile()` before `test()`.')

        self.training = False
        self.step = 0

        history = History()
        callbacks = self._configure_callbacks(callbacks, env, history, n_episodes, visualize)        

        callbacks.on_train_begin()
        
        for episode in range(n_episodes):
            callbacks.on_episode_begin(episode)
            
            episode_reward = 0.
            episode_step = 0

            self.reset_states()
            observation = deepcopy(env.reset())
            observation = self.processor.process_observation(observation)

            done = False
            while not done:
                callbacks.on_step_begin(episode_step)

                action = self.forward(observation)
                action = self.processor.process_action(action)
                
                reward = 0.
                accumulated_info = {}
                for _ in range(action_repetition):
                    callbacks.on_action_begin(action)
                    observation, r, d, info = env.step(action)
                    observation = deepcopy(observation)
                    observation, r, d, info = self.processor.process_step(observation, r, d, info)
                    callbacks.on_action_end(action)
                    reward += r
                    for key, value in info.items():
                        if not np.isreal(value):
                            continue
                        if key not in accumulated_info:
                            accumulated_info[key] = np.zeros_like(value)
                        accumulated_info[key] += value
                    if d:
                        done = True
                        break
                if n_max_episode_steps and episode_step >= n_max_episode_steps - 1:
                    done = True
                self.backward(reward, terminal=done)
                episode_reward += reward

                step_logs = {
                    'action': action,
                    'observation': observation,
                    'reward': reward,
                    'episode': episode,
                    'info': accumulated_info,
                }
                callbacks.on_step_end(episode_step, step_logs)
                episode_step += 1
                self.step += 1

            self.forward(observation)
            self.backward(0., terminal=False)

            episode_logs = {
                'episode_reward': episode_reward,
                'n_steps': episode_step,
            }
            callbacks.on_episode_end(episode, episode_logs)
        callbacks.on_train_end()

        return history

    def _configure_callbacks(self, callbacks, env, history, n_steps, visualize):
        callbacks = [] if not callbacks else callbacks[:]
        callbacks += [history]
        if visualize:
            callbacks += [Visualizer()]
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.set_env(env)
        callbacks.set_params({'n_steps': n_steps})
        return callbacks

    def reset_states(self):
        """Resets all internally kept states after an episode is completed.
        """
        pass

    def get_config(self):
        """Configuration of the agent for serialization.
        # Returns
            Dictionnary with agent configuration
        """
        return {}

    @abc.abstractmethod
    def forward(self, observation):
        """Takes the an observation from the environment and returns the action to be taken next.
        If the policy is implemented by a neural network, this corresponds to a forward (inference) pass.
        # Argument
            observation (object): The current observation from the environment.
        # Returns
            The next action to be executed in the environment.
        """
        pass

    @abc.abstractmethod
    def backward(self, reward, terminal):
        """Updates the agent after having executed the action returned by `forward`.
        If the policy is implemented by a neural network, this corresponds to a weight update using back-prop.
        # Argument
            reward (float): The observed reward after executing the action returned by `forward`.
            terminal (boolean): `True` if the new state of the environment is terminal.
        # Returns
            List of metrics values
        """
        pass

    @abc.abstractmethod
    def compile(self, optimizer, metrics=None):
        """Compiles an agent and the underlaying models to be used for training and testing.
        # Arguments
            optimizer (`keras.optimizers.Optimizer` instance): The optimizer to be used during training.
            metrics (list of functions `lambda y_true, y_pred: metric`): The metrics to run during training.
        """
        pass

    @abc.abstractmethod
    def load_weights(self, filepath):
        """Loads the weights of an agent from an HDF5 file.
        # Arguments
            filepath (str): The path to the HDF5 file.
        """
        pass

    @abc.abstractmethod
    def save_weights(self, filepath, overwrite=False):
        """Saves the weights of an agent as an HDF5 file.
        # Arguments
            filepath (str): The path to where the weights should be saved.
            overwrite (boolean): If `False` and `filepath` already exists, raises an error.
        """
        pass

    @property
    def layers(self):
        """Returns all layers of the underlying model(s).
        If the concrete implementation uses multiple internal models,
        this method returns them in a concatenated list.
        # Returns
            A list of the model's layers
        """
        return []

    @property
    def metrics_names(self):
        """The human-readable names of the agent's metrics. Must return as many names as there
        are metrics (see also `compile`).
        # Returns
            A list of metric's names (string)
        """
        return []


class Policy(abc.ABC):
    """Abstract base class for all implemented policies.
    Each policy helps with selection of action to take on an environment.
    Do not use this abstract base class directly but instead use one of the concrete policies implemented.
    # Arguments
        agent (rl.core.Agent): Agent used
    """
    def _set_agent(self, agent):
        self.agent = agent

    @property
    def metrics_names(self):
        return []

    @property
    def metrics(self):
        return []

    @abc.abstractmethod
    def select_action(self, **kwargs):
        pass

    def get_config(self):
        return {}


class Processor(object):
    """Abstract base class for implementing processors.
    A processor acts as a coupling mechanism between an `Agent` and its `Env`. This can
    be necessary if your agent has different requirements with respect to the form of the
    observations, actions, and rewards of the environment. By implementing a custom processor,
    you can effectively translate between the two without having to change the underlaying
    implementation of the agent or environment.
    """

    def process_step(self, observation, reward, done, info):
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        return observation, reward, done, info

    def process_observation(self, observation):
        return observation

    def process_reward(self, reward):
        return reward

    def process_info(self, info):
        return info

    def process_action(self, action):
        return action

    def process_state_batch(self, batch):
        return batch

    @property
    def metrics(self):
        return []

    @property
    def metrics_names(self):
        return []


class Env(abc.ABC):
    """The abstract environment class that is used by all agents. This class has the exact
    same API that OpenAI Gym uses so that integrating with it is trivial. In contrast to the
    OpenAI Gym implementation, this class only defines the abstract methods without any actual
    implementation.
    """
    reward_range = (-np.inf, np.inf)
    action_space = None
    observation_space = None

    @abc.abstractmethod
    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        # Arguments
            action (object): An action provided by the environment.
        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        pass

    @abc.abstractmethod
    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)
        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        """
        pass

    @abc.abstractmethod
    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    @abc.abstractmethod
    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        # Returns
            Returns the list of seeds used in this env's random number generators
        """
        pass

    @abc.abstractmethod
    def configure(self, *args, **kwargs):
        """Provides runtime configuration to the environment.
        This configuration should consist of data that tells your
        environment how to run (such as an address of a remote server,
        or path to your ImageNet data). It should not affect the
        semantics of the environment.
        """
        pass

    def __del__(self):
        self.close()

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)

