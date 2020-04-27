import numpy as np
import tensorflow as tf
from gym import envs

from lib.pgt_utils import ActorReinforce, pgt_gradients
from lib.rl_utils import compute_returns, episode_rollout
from lib.visualizer import Visualizer

print(tf.__version__)
visualizer = Visualizer()

MAX_STEPS = 300
MAX_EPISODES = 2000
GAMMA = 0.995
FPS = 100
BATCH = 1
DECAY_PERIOD = 1000
ENV_NAME = "CartPole-v1"
render = False

# Environment
env = envs.make(ENV_NAME)
actor = ActorReinforce(
    env.observation_space.shape[0], env.action_space.n, (16, 16), 0.0001
)

# Training
e, episodes, total_returns, alpha_decay = 1, [], [], 1.0
t_grads = actor.initialize_gradients()  # Initialize zero gradients
while e < MAX_EPISODES:

    _, _, _, rewards, gradients = episode_rollout(env, actor, MAX_STEPS, render)
    total_return, returns = compute_returns(np.array(rewards), GAMMA)

    total_returns.append(total_return), episodes.append(e)

    print(
        "e {:<20} return {:<20} length {:<20}".format(
            e, np.round(total_return, decimals=3), len(rewards)
        )
    )

env.close()

visualizer.plot_signals(
    [(np.array(episodes), np.array(total_returns), "REINFORCE")], y_lim=[0, None]
).show()

visualizer.plot_signals_mean_std(
    [(np.array(episodes), np.array(total_returns), "REINFORCE", 100)], y_lim=[0, None]
).show()
