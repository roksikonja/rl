import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gym import envs

tf.keras.backend.set_floatx("float64")
print(tf.__version__)

MODE = "reinforce"
MAX_STEPS = 100
MAX_EPISODES = 5
FPS = 100
ENV_NAME = "CartPole-v1"

env = envs.make(ENV_NAME)
action_space = env.action_space


def policy(state):
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(24, input_shape=(4,), activation="relu", name="dense_1",
                                                              trainable=True),
                                        tf.keras.layers.Dense(24, activation="relu", name="dense_2", trainable=True),
                                        tf.keras.layers.Dense(2, activation="softmax", name="dense_3", trainable=True)])
    action_probs = model(state)  # (1, K)

    action = tfp.distributions.Categorical(probs=action_probs, name='action_sampling').sample(1)
    action = tf.reshape(action, ())  # ()

    # action = action_space.sample()  # a_t
    return action.numpy()


class Actor(tf.keras.Model):
    def __init__(self):
        super(Actor, self).__init__()
        self.dense_l1 = tf.keras.layers.Dense(24, input_shape=(4,), activation="relu", name="dense_l1", trainable=True)
        self.dense_l2 = tf.keras.layers.Dense(24, activation="relu", name="dense_l2", trainable=True)
        self.dense_l3 = tf.keras.layers.Dense(2, activation="softmax", name="dense_l3", trainable=True)

        self.grads = None

        self.probs, self.log_probs = None, None
        self.action = None

        self.learning_rate = 0.001

        self.optimizer = tf.keras.optimizers.SGD(lr=self.learning_rate)
        self.compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def __call__(self, state):
        """
        Works for batch_size = 1
        :param state:
        :return:
        """

        with tf.GradientTape(persistent=True) as gt:
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
            print(self.action)

            self.eligibility = self.log_probs[:, self.action]

        grads = gt.gradient(self.loss, self.trainable_variables)

        return self.action, grads

        # print(grads)
        # self.loss = tf.keras.losses.sparse_categorical_crossentropy(self.action, self.probs, from_logits=True)
        # print(self.loss.shape, self.loss)

    def initialize_gradients(self):
        self.grads = self.trainable_variables
        for i, variable in enumerate(self.grads):
            self.grads[i] = 0 * variable

    def apply_gradients(self, grads):
        """
        Apply gradients to training variables. (Gradient Ascent)
        """
        for i, grad in enumerate(grads):
            self.trainable_variable[i] = self.trainable_variable[i] + grad


def compute_returns(rewards, gamma=1.0):
    assert 0 <= gamma <= 1.0
    if gamma == 1.0:
        returns = np.cumsum(rewards)[::-1]
    elif gamma == 0:
        returns = rewards
    else:
        returns = np.zeros_like(rewards)
        g = 0  # G_T
        for t in reversed(range(returns.shape[0])):  # T-1, T-2, ..., 0
            r = rewards[t]  # r_t+1
            g = r + gamma * g  # G_t = r_t+1 + gamma * G_t+1
            returns[t] = g

    total_return = returns[0]
    return total_return, returns  # G_0, G_1, ..., G_T-1


def episode_rollout(env, policy, render=False):
    state = env.reset()
    if render:
        env.render(mode='human')

    memory = []
    states, next_states, actions, rewards, dones = [], [], [], [], []
    done, t = False, 0
    while not done and t < MAX_STEPS:  # t = 0, 1, 2, ..., T-1
        states.append(state)  # s_t

        state = np.reshape(state, (1, -1))
        action = policy(state)
        actions.append(action)  # a_t

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
    return t, np.array(states), np.array(next_states), np.array(actions), np.array(rewards), np.array(dones), memory


# e = 0
# while e < MAX_EPISODES:
#     length, states, next_states, actions, rewards, dones, memory = episode_rollout(env, policy, render=True)
#     total_return, returns = compute_returns(rewards)
#     print(length, states.shape, actions.shape, rewards.shape)
#     e = e + 1
#
# env.close()

# state = np.ones((1, 4))
# print(policy(state))
# print(compute_returns(np.ones(10), 0.0))
# print(compute_returns(np.ones(10), 0.999))
# print(compute_returns(np.ones(10), 1.0))

Actor = Actor()

s = np.ones((1, 4))
Actor(s)

# Actor.get_grads()
# print(Actor.grad_buffer)
