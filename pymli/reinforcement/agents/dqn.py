import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Lambda, Input
from ..core import Agent
from ..policy import EpsGreedyQPolicy, GreedyQPolicy
from ..utils import huber_loss, mean_q, clone_model


class DQNAgent(Agent):
    def __init__(self, model, n_actions, memory, policy=None, test_policy=None, gamma=.99, batch_size=32,
                 memory_interval=1, train_interval=1, n_steps_warmup=10, target_model_update=10000, 
                 custom_model_objects=None, delta_clip=np.inf, *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs)

        self.model = model
        self.n_actions = n_actions
        self.memory = memory
        self.policy = policy if policy is not None else EpsGreedyQPolicy()
        self.test_policy = test_policy if test_policy is not None else GreedyQPolicy()
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_interval = memory_interval
        self.train_interval = train_interval
        self.n_steps_warmup = n_steps_warmup
        self.target_model_update = target_model_update
        self.custom_model_objects = custom_model_objects
        self.delta_clip = delta_clip

        self.recent_observation = None
        self.recent_action = None
        self.rewards = None
        self.trainable_model = None
        self.target_model = None

    def process_state_batch(self, batch):
        batch = np.array(batch)
        return self.processor.process_state_batch(batch)

    def compute_batch_q_values(self, state_batch):
        batch = self.process_state_batch(state_batch)
        q_values = self.model.predict_on_batch(batch)
        return q_values

    def compute_q_values(self, state):
        q_values = self.compute_batch_q_values([state]).flatten()
        return q_values

    def update_target_model_hard(self):
        self.target_model.set_weights(self.model.get_weights())

    def forward(self, observation):
        q_values = self.compute_q_values([observation])

        if self.training:
            action = self.policy.select_action(q_values=q_values)
        else:
            action = self.test_policy.select_action(q_values=q_values)

        self.recent_observation = observation
        self.recent_action = action

        return action

    def backward(self, reward, terminal):
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal, training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            return metrics

        if self.step > self.n_steps_warmup and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)

            # Compute Q values for mini-batch update.
            target_q_values = self.target_model.predict_on_batch(state1_batch)
            q_batch = np.max(target_q_values, axis=1).flatten()

            targets = np.zeros((self.batch_size, self.n_actions))
            dummy_targets = np.zeros((self.batch_size,))
            masks = np.zeros((self.batch_size, self.n_actions))

            discounted_reward_batch = self.gamma * q_batch
            discounted_reward_batch *= terminal1_batch

            Rs = reward_batch + discounted_reward_batch
            for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
                target[action] = R  # update action with estimated accumulated reward
                dummy_targets[idx] = R
                mask[action] = 1.  # enable loss for this specific action
            targets = np.array(targets).astype('float32')
            masks = np.array(masks).astype('float32')

            # Finally, perform a single update on the entire batch. We use a dummy target since
            # the actual loss is computed in a Lambda layer that needs more complex input. However,
            # it is still useful to know the actual target to compute metrics properly.
            ins = [state0_batch] if type(self.model.input) is not list else state0_batch
            metrics = self.trainable_model.train_on_batch(ins + [targets, masks], [dummy_targets, targets])
            # throw away individual losses
            metrics = [metric for idx, metric in enumerate(metrics) if idx not in (1, 2)]
            metrics += self.policy.metrics
            if self.processor is not None:
                metrics += self.processor.metrics

        if self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics

    def compile(self, optimizer, metrics=None):
        if metrics is None:
            metrics = []

        metrics += [mean_q]

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        self.target_model = clone_model(self.model, self.custom_model_objects)
        self.target_model.compile(optimizer='sgd', loss='mse')
        self.model.compile(optimizer='sgd', loss='mse')

        # Compile model.
        def clipped_masked_error(args):
            y_true, y_pred, mask = args
            loss = huber_loss(y_true, y_pred, self.delta_clip)
            loss *= mask  # apply element-wise mask
            return K.sum(loss, axis=-1)

        y_pred = self.model.output
        y_true = Input(name='y_true', shape=(self.n_actions,))
        mask = Input(name='mask', shape=(self.n_actions,))
        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_true, y_pred, mask])
        ins = [self.model.input] if type(self.model.input) is not list else self.model.input
        trainable_model = Model(inputs=ins + [y_true, mask], outputs=[loss_out, y_pred])
        assert len(trainable_model.output_names) == 2
        combined_metrics = {trainable_model.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
        self.trainable_model = trainable_model

        self.compiled = True

    def load_weights(self, filepath):
        pass

    def save_weights(self, filepath, overwrite=False):
        pass
