import time

import gpflow
import numpy as np
import tensorflow as tf

from .mgpr import MGPR
from .. import controllers
from .. import rewards


class PILCO(gpflow.Module):
    def __init__(
        self, X, Y, horizon=30, controller=None, name=None,
    ):
        super(PILCO, self).__init__(name)
        self.mgpr = MGPR(X, Y)

        self.state_dim = Y.shape[1]
        self.control_dim = X.shape[1] - Y.shape[1]

        self.horizon = horizon

        if controller is None:
            self.controller = controllers.LinearController(
                self.state_dim, self.control_dim
            )
        else:
            self.controller = controller

        self.reward = rewards.ExponentialReward(self.state_dim)

        self.m_init = X[0:1, 0 : self.state_dim]
        self.S_init = np.diag(np.ones(self.state_dim) * 0.1)

        self.optimizer = gpflow.optimizers.Scipy()

    def _build_likelihood(self):
        reward = self.predict(self.m_init, self.S_init, self.horizon)[2]
        return reward

    def optimize_models(self, restarts=1):
        """
            Optimize GP models to data (X, Y) by evidence maximization.
        """
        self.mgpr.optimize(restarts=restarts)

    def optimize_policy(self, maxiter=50, restarts=1):
        """
            Optimize controller's parameter's, i.e. W and b in the case of linear controller.
        """
        self.optimizer.minimize(self.training_loss, maxiter=maxiter)
        print(f"Controller's optimization: done with reward={self.compute_reward()}.")

    def compute_action(self, x_m):
        return self.controller.compute_action(
            x_m, tf.zeros([self.state_dim, self.state_dim], np.float64)
        )[0]

    def predict(self, m_x, s_x, n):
        loop_vars = [tf.constant(0, tf.int32), m_x, s_x, tf.constant([[0]], np.float64)]

        _, m_x, s_x, reward = tf.while_loop(
            # Termination condition
            lambda j, m_x, s_x, reward: j < n,
            # Body function
            lambda j, m_x, s_x, reward: (
                j + 1,
                *self.propagate(m_x, s_x),
                tf.add(reward, self.reward.compute_reward(m_x, s_x)[0]),
            ),
            loop_vars,
        )

        return m_x, s_x, reward

    def propagate(self, m_x, s_x):
        m_u, s_u, c_xu = self.controller.compute_action(m_x, s_x)

        m = tf.concat([m_x, m_u], axis=1)
        s1 = tf.concat([s_x, s_x @ c_xu], axis=1)
        s2 = tf.concat([tf.transpose(s_x @ c_xu), s_u], axis=1)
        s = tf.concat([s1, s2], axis=0)

        M_dx, S_dx, C_dx = self.mgpr.predict_on_noisy_inputs(m, s)
        M_x = M_dx + m_x

        S_x = (
            S_dx
            + s_x
            + s1 @ C_dx
            + tf.matmul(C_dx, s1, transpose_a=True, transpose_b=True)
        )

        # While-loop requires the shapes of the outputs to be fixed
        M_x.set_shape([1, self.state_dim])
        S_x.set_shape([self.state_dim, self.state_dim])
        return M_x, S_x

    def compute_reward(self):
        return self._build_likelihood()
