import gym
import numpy as np

from pilco.controllers import LinearController

# from pilco.controllers import RbfController
from pilco.models import PILCO
from rl_utils import rollout

np.random.seed(0)

env = gym.make("InvertedPendulum-v2")
# Initial random rollouts to generate a dataset
X, Y = rollout(env=env, pilco=None, random=True, max_timesteps=200, verbose=False)
print(X.shape, Y.shape)
for i in range(1, 3):
    X_, Y_ = rollout(env=env, pilco=None, random=True, max_timesteps=200, verbose=False)
    print(X_.shape, Y_.shape)
    X = np.vstack((X, X_))
    Y = np.vstack((Y, Y_))
print(X.shape, Y.shape)

state_dim = Y.shape[1]
control_dim = X.shape[1] - state_dim

controller = LinearController(state_dim=state_dim, control_dim=control_dim)

# controller = RbfController(
#     state_dim=state_dim, control_dim=control_dim, num_basis_functions=5
# )

pilco = PILCO(X, Y, controller=controller, horizon=40)

# for i in range(3):
#     print(i)
#     pilco.optimize_models()
#     pilco.optimize_policy()
#
#     X_new, Y_new = rollout(env=env, pilco=pilco, max_timesteps=100, verbose=False)
#     # Update dataset
#     X = np.vstack((X, X_new))
#     Y = np.vstack((Y, Y_new))
#     pilco.mgpr.set_XY(X, Y)
