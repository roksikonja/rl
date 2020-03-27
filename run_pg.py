import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from gym import envs

from pg_utils import ActorReinforce, ActorBaseline, ActorRandom
from rl_utils import episode_rollout, compute_returns

print(tf.__version__)

MODE = "REINFORCE"
MAX_STEPS = 200
MAX_EPISODES = 5000
GAMMA = 0.8
ENV_NAME = "CartPole-v1"

env = envs.make(ENV_NAME)

if MODE == "REINFORCE":
    Actor = ActorReinforce(env.observation_space.shape[0], env.action_space.n, 64, 64, 0.00001)
elif MODE == "BASELINE":
    Actor = ActorBaseline(env.observation_space.shape[0], env.action_space.n, 64, 64, 0.00001)
else:
    Actor = ActorRandom(env.action_space.n)

e = 1
episodes, total_returns = [], []
render = False
while e < MAX_EPISODES:
    length, states, next_states, actions, rewards, dones, gradients, memory = episode_rollout(env, Actor,
                                                                                              max_steps=MAX_STEPS,
                                                                                              render=render)
    total_return, returns = compute_returns(rewards, GAMMA)

    print("e {:<20} return {:<20} length {:<20}".format(e, np.round(total_return, decimals=3), length))

    for t in range(len(gradients)):
        grads_t = gradients[t]
        return_t = returns[t]
        Actor.apply_gradients(grads_t, return_t * (np.power(GAMMA, t)))

    total_returns.append(total_return)
    episodes.append(e)
    e = e + 1

episodes = np.array(episodes)
total_returns = np.array(total_returns)
average_returns = pd.Series(total_returns).rolling(100, min_periods=1).mean().values

plt.figure(figsize=(16, 5))
plt.plot(episodes, total_returns, label="REINFORCE")
plt.plot(episodes, average_returns, label="avg_REINFORCE")
plt.legend()
plt.show()
