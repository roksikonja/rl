import gpflow
import numpy as np
import tensorflow as tf


def squash_sin(m, s, max_action=None):
    """
    Squashing function, passing the controls mean and variance
    through a sinus, as in gSin.m. The output is in [-max_action, max_action].
    IN: mean (m) and variance(s) of the control input, max_action
    OUT: mean (M) variance (S) and input-output (C) covariance of the squashed
         control input
    """
    k = tf.shape(m)[1]
    if max_action is None:
        max_action = tf.ones((1, k), dtype=np.float64)  # squashes in [-1,1] by default
    else:
        max_action = max_action * tf.ones((1, k), dtype=np.float64)

    M = max_action * tf.exp(-tf.diag_part(s) / 2) * tf.sin(m)

    lq = -(tf.diag_part(s)[:, None] + tf.diag_part(s)[None, :]) / 2
    q = tf.exp(lq)
    S = (tf.exp(lq + s) - q) * tf.cos(tf.transpose(m) - m) - (
        tf.exp(lq - s) - q
    ) * tf.cos(tf.transpose(m) + m)
    S = max_action * tf.transpose(max_action) * S / 2

    C = max_action * tf.diag(tf.exp(-tf.diag_part(s) / 2) * tf.cos(m))
    return M, S, tf.reshape(C, shape=[k, k])


class LinearController(gpflow.Module):
    def __init__(self, state_dim, control_dim, max_action=None):
        gpflow.Module.__init__(self)
        self.W = gpflow.Parameter(np.random.rand(control_dim, state_dim))
        self.b = gpflow.Parameter(np.random.rand(1, control_dim))
        self.max_action = max_action

    def compute_action(self, m, s, squash=False):
        """
        Perform a linear transform. Random variable x is normally distributed, i.e. x ~ N(m, s).

            pi(x) = W x + b ~ N(M, S)
            M = W m + b
            S = W s W^T

        :param m: Mean of x. (num_steps, state_dim + control_dim)
        :type m: ndarray
        :param s: Covariance matrix of x. (num_steps, state_dim + control_dim, state_dim + control_dim)
        :type s: ndarray
        :param squash: If squashing of the control output should be squashed.
        :type squash: bool
        :return:
            M output mean
            S output variance
            V input-output covariance
        """
        M = m @ tf.transpose(self.W) + self.b  # mean output
        S = self.W @ s @ tf.transpose(self.W)  # output variance
        V = tf.transpose(self.W)  # input output covariance
        if squash:
            M, S, V2 = squash_sin(M, S, self.max_action)
            V = V @ V2
        return M, S, V

    def randomize(self, mean=0, sigma=1.0):
        self.W.assign(mean + sigma * np.random.normal(size=self.W.shape))
        self.b.assign(mean + sigma * np.random.normal(size=self.b.shape))

    def policy(self, m, s, squash=True):
        return self.compute_action(m, s, squash)[0]


class RandomController(gpflow.Module):
    def __init__(self, env):
        self.env = env

    def compute_action(self):
        return self.env.action_space.sample(), None, None

    def policy(self, state):
        return self.compute_action()[0]
