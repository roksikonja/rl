import gym
import numpy as np
import tensorflow as tf
import gpflow

from pilco_v2.controllers import RandomController
from pilco_v2.controllers import LinearController
from pilco_v2.utils import pilco_rollout
from pilco_v2.utils import get_summary
from pilco_v2.models import PILCO


tf.keras.backend.set_floatx("float32")
gpflow.config.set_default_float("float32")

np.random.seed(10)
env = gym.make("InvertedPendulum-v2")

state_dim = env.observation_space.shape[0]
control_dim = env.action_space.shape[0]

"""
    Initialization using random controller.
    Data is collected to build a GP model.
"""

actor = RandomController(env=env)
print(get_summary(actor))
X, Y = pilco_rollout(
    env=env,
    actor=actor,
    max_timesteps=200,
    num_episodes=5,
    verbose=True,
    render=True,
    fps=30,
)
print(X.shape, Y.shape, X.dtype, Y.dtype)

"""
    Actual selection of the controller.
"""

actor = LinearController(state_dim=state_dim, control_dim=control_dim)
print(get_summary(actor))

pilco = PILCO(X, Y, controller=actor, horizon=40)
print(get_summary(pilco))

pilco.optimize_models()
print(get_summary(pilco))

pilco.optimize_policy()

pilco.mgpr.set_XY(X, Y)
