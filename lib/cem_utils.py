import argparse
import time
import tensorflow as tf

import numpy as np


class RandomAgent(object):

    def __init__(self, env):
        self.env = env

    def act(self, _):
        action = self.env.action_space.sample()
        return action


def episode_rollout(env, actor, max_timesteps=100, render=False, fps=100):
    state = env.reset()
    if render:
        env.render(mode="human")

    states = []
    actions = []
    rewards = []

    for t in range(max_timesteps):  # t = 0, 1, 2, ..., T-1
        states.append(state)  # s_t

        action = actor.act(state)

        actions.append(action)  # a_t

        next_state, reward, done, _ = env.step(action)  # s_t+1, r_t+1
        # if done:
        #     reward = -reward

        rewards.append(reward)  # r_t+1

        # (state, action, reward, next_states)  (s_t, a_t, r_t+1, s_t+1)
        state = next_state

        if render:
            time.sleep(1.0 / fps)
            env.render()

        if done:
            break

    # states: s_0, s_1, s_2, ..., s_T-1
    # actions: a_0, a_1, ..., a_T-1
    # rewards: r_1, r_2, ..., r_T
    return np.array(states), np.array(actions), np.array(rewards)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_iters", default=100, type=int, help="Number of training iterations."
    )
    parser.add_argument(
        "--n_samples",
        default=100,
        type=int,
        help="Number of samples CEM algorithm chooses from on each iteration.",
    )
    parser.add_argument(
        "--max_timesteps", default=500, type=int, help="Maximum episode length."
    )
    parser.add_argument(
        "--best_frac",
        default=0.2,
        type=float,
        help="Fraction of top samples used to calculate mean and variance of next iteration",
    )
    parser.add_argument("--fps", default=100, type=int, help="Rendering FPS.")
    return parser.parse_args()


class CEMAgentR(object):
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions

        # Initialize linear policy parameters
        self.mean, self.variance = None, None
        self.W, self.b = None, None
        self.initialize_policy()

    def initialize_policy(self):
        """
            Initialize policy weights W and b.
        """
        self.mean = np.random.randn(self.n_states + 1)
        self.variance = np.square(0.1 * np.ones_like(self.mean))
        Wb = np.random.multivariate_normal(
            self.mean, np.diag(self.variance)
        )  # (n_states + 1, )

        self.set_policy(Wb)

    def act(self, state):
        """
            Linear policy.

            a = sign(s^T W + b)
        """
        logits = np.matmul(state, self.W) + self.b  # (None, )
        a = np.greater_equal(logits, 0).astype(np.int)  # (None, )
        return a

    def set_policy(self, Wb):
        """
            Update policy weights given using a weight sample.
        """
        self.W = Wb[:-1]  # (n_states, )
        self.b = Wb[-1]  # ()


class CEMAgentG(tf.keras.Model):
    def __init__(self, n_states, n_actions, n_hidden, alpha):
        super(CEMAgentG, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions

        # Actor
        self.layer_input = tf.keras.layers.Dense(
            n_hidden[0],
            input_shape=(n_states, ),
            activation="relu",
            name="actor_layer_input",
            trainable=True,
        )

        self.layers_hidden = []
        for l, n in enumerate(n_hidden[1:]):
            self.layers_hidden.append(
                tf.keras.layers.Dense(
                    n, activation="relu", name=f"actor_layer_hidden_{l}", trainable=True
                )
            )

        self.layer_output = tf.keras.layers.Dense(
            n_actions, activation=None, name="actor_layer_output", trainable=True
        )

        # Training
        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=alpha),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

        # Initialize variables
        _ = self.act(np.zeros((10, n_states)).astype(np.float32))

        for var in self.trainable_variables:
            print(var.name, var.shape)

    @tf.function(autograph=False)
    def call(self, state):
        a = self.layer_input(state)  # (None , n_hidden_l1)
        for layer in self.layers_hidden:
            a = layer(a)  # (None, n_hidden)

        logits = self.layer_output(a)  # (None, n_actions)
        return logits

    def act(self, state):
        state = np.reshape(state, (-1, self.n_states)).astype(np.float32)

        logits = self.call(state)
        return np.squeeze(tf.argmax(logits, axis=1).numpy())  # (None, )
