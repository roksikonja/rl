import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from gym import envs

from rl_utils import compute_returns

print(tf.__version__)


class ActorCritic(tf.keras.Model):
    def __init__(self, n_states, n_actions, n_hidden_l1, n_hidden_l2, alpha_a, alpha_c):
        super(ActorCritic, self).__init__()
        # Actor
        self.actor_dense_l1 = tf.keras.layers.Dense(n_hidden_l1, input_shape=(n_states,), activation="tanh",
                                                    name="actor_dense_l1", trainable=True)
        self.actor_dense_l2 = tf.keras.layers.Dense(n_hidden_l2, activation="tanh", name="actor_dense_l2",
                                                    trainable=True)
        self.actor_dense_l3 = tf.keras.layers.Dense(n_actions, activation="softmax", name="actor_dense_l3",
                                                    trainable=True)

        # Baseline
        self.critic_dense_l1 = tf.keras.layers.Dense(n_hidden_l1, input_shape=(n_states,), activation="tanh",
                                                     name="critic_dense_l1", trainable=True)
        self.critic_dense_l2 = tf.keras.layers.Dense(n_hidden_l2, activation="tanh", name="critic_dense_l2",
                                                     trainable=True)
        self.critic_dense_l3 = tf.keras.layers.Dense(1, activation=None, name="critic_dense_l3", trainable=True)

        self.probs, self.log_probs, = None, None
        self.action, self.action_log_probs, self.value = None, None, None

        # Training
        self.alpha_a, self.alpha_c, self.optimizer = alpha_a, alpha_c, tf.keras.optimizers.SGD()

        # Initialize variables
        _ = self.act(np.zeros((1, n_states)).astype(np.float32))
        _ = self.state_value(np.zeros((1, n_states)).astype(np.float32))

        self.actor_vars = None, None
        for var in self.trainable_variables:
            print(var.name, var.shape)

    @tf.function(autograph=False)
    def act(self, state):
        with tf.GradientTape() as gt:
            # Policy DNN
            a1 = self.actor_dense_l1(state)  # (n_batch, n_hidden_l1)
            a2 = self.actor_dense_l2(a1)  # (n_batch, n_hidden_l2)
            a3 = self.actor_dense_l3(a2)  # (n_batch, n_actions)

            # Probabilities
            self.probs = a3  # (n_batch, n_actions)
            self.log_probs = tf.math.log(a3)  # (n_batch, n_actions)

            # Sample action
            self.action = tfp.distributions.Categorical(probs=self.probs).sample(1)
            self.action = tf.squeeze(self.action)  # (n_batch, ) or ()

            self.action_log_probs = self.log_probs[:, self.action]

        grads = gt.gradient(self.action_log_probs, self.actor_variables())

        return self.action, grads

    @tf.function(autograph=False)
    def state_value(self, state):
        assert state.shape[0] == 1
        with tf.GradientTape() as gt:
            a1 = self.critic_dense_l1(state)  # (n_batch, n_hidden_l1)
            a2 = self.critic_dense_l2(a1)  # (n_batch, n_hidden_l2)
            a3 = self.critic_dense_l3(a2)  # (n_batch, 1)

            # Value function
            self.value = tf.squeeze(a3)  # (n_batch, ) or ()

        grads = gt.gradient(self.value, self.critic_variables())

        return self.value, grads

    @tf.function(autograph=False)
    def apply_gradients_c(self, grads, alpha_factor):
        # - for Gradient Ascent
        self.optimizer.learning_rate = - self.alpha_c * alpha_factor
        self.optimizer.apply_gradients(zip(grads, self.critic_variables()))

    @tf.function(autograph=False)
    def apply_gradients_a(self, grads, alpha_factor):
        # - for Gradient Ascent
        self.optimizer.learning_rate = - self.alpha_a * alpha_factor
        self.optimizer.apply_gradients(zip(grads, self.actor_variables()))

    def actor_variables(self):
        return [var for var in self.trainable_variables if "actor" in var.name]

    def critic_variables(self):
        return [var for var in self.trainable_variables if "critic" in var.name]


if __name__ == "__main__":
    start = time.time()

    MODE = "AC"
    MAX_STEPS = 500
    MAX_EPISODES = 5000
    GAMMA = 0.999
    FPS = 100
    DECAY_PERIOD = 2000
    ENV_NAME = "CartPole-v1"
    render = False

    # Environment
    env = envs.make(ENV_NAME)

    actor_critic = ActorCritic(env.observation_space.shape[0], env.action_space.n, 2 ** 6, 2 ** 6, 0.001, 0.001)
    # Training
    e, episodes, total_returns, alpha_decay = 1, [], [], 1.0
    for e in range(1, MAX_EPISODES + 1):
        # Generate episode
        state = np.reshape(env.reset(), (1, -1)).astype(np.float32)  # s_0
        action, grads = actor_critic.act(state)  # a_0, grads_0
        action = action.numpy()

        if render:
            env.render(mode='human')

        rewards = []
        done, t = False, 0
        while not done and t < MAX_STEPS:  # t = 0, 1, 2, ..., T-1

            t = t + 1
            next_state, reward, done, _ = env.step(action)  # s_t+1, r_t+1
            next_state = np.reshape(next_state, (1, -1)).astype(np.float32)
            rewards.append(reward)

            # Update weights
            state_value, v_grads = actor_critic.state_value(state)
            state_value = state_value.numpy()

            if done:
                next_state_value = 0.0
            else:
                next_state_value, _ = actor_critic.state_value(next_state)
                next_state_value = next_state_value.numpy()

            delta = reward + GAMMA * next_state_value - state_value

            actor_critic.apply_gradients_c(v_grads, tf.Variable(delta * alpha_decay))  # Update critic weights
            actor_critic.apply_gradients_a(grads,
                                           tf.Variable(
                                               delta * np.power(GAMMA, t) * alpha_decay))  # Update actor weights

            # Act
            next_action, next_grads = actor_critic.act(next_state)  # a_t+1, grads_t+1
            next_action = next_action.numpy()

            state, action, grads = next_state, next_action, next_grads

            if render:
                time.sleep(1.0 / FPS)
                env.render()

        e_length = t

        total_return, returns = compute_returns(np.array(rewards), GAMMA)
        print("e {:<20} return {:<20} length {:<20}".format(e, np.round(total_return, decimals=3), e_length))
        total_returns.append(total_return), episodes.append(e)

        alpha_decay = np.exp(- e / DECAY_PERIOD)

    env.close()
    episodes = np.array(episodes)
    total_returns = np.array(total_returns)
    average_returns = pd.Series(total_returns).rolling(100, min_periods=1).mean().values

    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    ax[0].plot(episodes, total_returns, label=f"{MODE}")
    ax[1].plot(episodes, average_returns, label=f"avg_{MODE}")
    ax[0].set_title("returns")
    ax[1].set_title("average returns")
    ax[0].set_ylim(bottom=0)
    ax[1].set_ylim(bottom=0)
    ax[0].legend()
    ax[1].legend()
    fig.savefig("./results/{}_ac_result.png".format(datetime.datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S")))
    fig.show()

    end = time.time()
    print(f"Finished in {end - start} s")
