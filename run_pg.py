import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gym import envs

print(tf.__version__)

MODE = "reinforce"
MAX_STEPS = 500
MAX_EPISODES = 1000
FPS = 40
BATCH_SIZE = 20
ENV_NAME = "CartPole-v1"

env = envs.make(ENV_NAME)
action_space = env.action_space


class Params(object):
    def __init__(self, n_states, n_actions, n_hidden_l1, n_hidden_l2, alpha, gamma):
        self.n_states, self.n_actions = n_states, n_actions

        self.n_hidden_l1, self.n_hidden_l2 = n_hidden_l1, n_hidden_l2

        self.alpha, self.gamma = alpha, gamma


class Actor(tf.keras.Model):
    def __init__(self, params: Params):
        super(Actor, self).__init__()
        # Policy
        self.dense_l1 = tf.keras.layers.Dense(params.n_hidden_l1, input_shape=(params.n_states,), activation="relu",
                                              name="dense_l1", trainable=True)
        self.dense_l2 = tf.keras.layers.Dense(params.n_hidden_l2, activation="relu", name="dense_l2", trainable=True)
        self.dense_l3 = tf.keras.layers.Dense(params.n_actions, activation="softmax", name="dense_l3", trainable=True)

        # Actor state
        self.state, self.probs, self.log_probs, self.action = None, None, None, None

        # Discounting factor
        self.gamma = params.gamma

        # Training
        self.alpha, self.optimizer = params.alpha, tf.keras.optimizers.SGD(lr=params.alpha)

    def __call__(self, state):
        """
        Works for batch_size = 1
        :param state:
        :return:
        """

        with tf.GradientTape() as gt:
            # DNN
            a1 = self.dense_l1(state)  # (batch_size, 24)
            a2 = self.dense_l2(a1)  # (batch_size, 24)
            a3 = self.dense_l3(a2)  # (batch_size, num_classes)

            # Probabilities
            self.probs = a3  # (batch_size, num_classes)
            self.log_probs = tf.math.log(a3)  # (batch_size, num_classes)

            # Sample action
            self.action = tfp.distributions.Categorical(probs=self.probs, name='action_sampling').sample(1)
            self.action = tf.squeeze(self.action)  # (batch_size, ) or ()

            self.action_log_probs = self.log_probs[:, self.action]

        grads = gt.gradient(self.action_log_probs, self.trainable_variables)

        return self.action.numpy(), grads

    def compute_returns(self, rewards):
        assert 0 <= self.gamma <= 1.0
        if self.gamma == 1.0:
            returns = np.cumsum(rewards)[::-1]
        elif self.gamma == 0:
            returns = rewards
        else:
            returns = np.zeros_like(rewards)
            g = 0  # G_T
            for t in reversed(range(returns.shape[0])):  # T-1, T-2, ..., 0
                r = rewards[t]  # r_t+1
                g = r + self.gamma * g  # G_t = r_t+1 + self.gamma * G_t+1
                returns[t] = g

        total_return = returns[0]
        return total_return, returns  # G_0, G_1, ..., G_T-1

    def apply_gradients(self, returns, grads):
        """
        Apply gradients to training variables. (Gradient Ascent)
        """
        for t in range(len(grads)):
            grads_t = grads[t]
            return_t = returns[t]
            # self.optimizer.learning_rate = - self.alpha * self.gamma ** t * return_t
            self.optimizer.learning_rate = - self.alpha * return_t  # - for Gradient Ascent
            self.optimizer.apply_gradients(zip(grads_t, self.trainable_variables))


def episode_rollout(env, actor, render=False):
    state = env.reset()
    if render:
        env.render(mode='human')

    memory = []
    states, next_states, actions, rewards, dones, gradients = [], [], [], [], [], []
    done, t = False, 0
    while not done and t < MAX_STEPS:  # t = 0, 1, 2, ..., T-1
        states.append(state)  # s_t

        state = np.reshape(state, (1, -1)).astype(np.float32)
        action, grads = actor(state)
        actions.append(action)  # a_t
        gradients.append(grads)

        t = t + 1
        next_state, reward, done, _ = env.step(action)  # s_t+1, r_t+1
        rewards.append(reward)  # r_t+1
        dones.append(done)
        next_states.append(next_state)

        memory.append((state, action, reward, next_states))  # (s_t, a_t, r_t+1, s_t+1)
        state = next_state
        if render:
            time.sleep(1.0 / FPS)
            env.render()

    # states: s_0, s_1, s_2, ..., s_T-1
    # actions: a_0, a_1, ..., a_T-1
    # rewards: r_1, r_2, ..., r_T
    return t, np.array(states), np.array(next_states), np.array(actions), np.array(rewards), np.array(dones), \
           gradients, memory


params = Params(env.observation_space.shape[0], env.action_space.n, 64, 64, 1e-5, 0.999)

Actor = Actor(params)
e = 1
episodes = []
total_returns = []
batch_gradients = []
render = False
while e < MAX_EPISODES:
    length, states, next_states, actions, rewards, dones, gradients, memory = episode_rollout(env, Actor, render=render)
    total_return, returns = Actor.compute_returns(rewards)

    print("e {:<20} return {:<20} length {:<20}".format(e, np.round(total_return, decimals=3), length))
    Actor.apply_gradients(returns, batch_gradients)

    # print(len(batch_gradients), len(gradients))
    # if not batch_gradients:
    #     batch_gradients = gradients
    #
    # if e % BATCH_SIZE:
    #     batch_gradients = [gradients[i] + batch_gradients[i] for i in range(len(gradients))]
    # else:
    #     batch_gradients = [grad / BATCH_SIZE for grad in batch_gradients]
    #     Actor.apply_gradients(returns, batch_gradients)
    #     batch_gradients = [grad * 0 for grad in batch_gradients]

    total_returns.append(total_return)
    episodes.append(e)
    e = e + 1

    # if length == MAX_STEPS and e > MAX_EPISODES // 10:
    #     render = True
    # else:
    #     render = False

    if e == MAX_EPISODES - 50:
        input(str(e))
        render = True

env.close()

episodes, total_returns = np.array(episodes), np.array(total_returns)

plt.figure(figsize=(16, 5))
plt.plot(episodes, total_returns, label="REINFORCE")
plt.legend()
plt.show()
