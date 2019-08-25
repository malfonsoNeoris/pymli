import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from pymli.reinforcement.agents.sarsa import SARSAAgent
from pymli.reinforcement.policy import EpsGreedyQPolicy


ENV_NAME = 'CartPole-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
n_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(n_actions))
model.add(Activation('linear'))
print(model.summary())

policy = EpsGreedyQPolicy()
sarsa = SARSAAgent(model=model, n_actions=n_actions, n_steps_warmup=10, policy=policy)
sarsa.compile(Adam(lr=1e-3), metrics=['mae'])

history = sarsa.fit(env, n_steps=50000, visualize=True)

# After training is done, we save the final weights.
sarsa.save_weights('sarsa_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
sarsa.test(env, n_episodes=5, visualize=True)