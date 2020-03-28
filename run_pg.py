import time
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from gym import envs

from pg_utils import ActorReinforce, ActorBaseline, ActorRandom, ActorConstant
from rl_utils import compute_returns

print(tf.__version__)

start = time.time()

MODE = "BASELINE"
MAX_STEPS = 300
MAX_EPISODES = 4000
GAMMA = 0.999
FPS = 100
DECAY_PERIOD = 2000
ENV_NAME = "CartPole-v1"
render = False

# Environment
env = envs.make(ENV_NAME)

results = []
for MODE in ["RANDOM", "REINFORCE", "BASELINE", "CONSTANT_BASELINE"]:
# for MODE in ["BASELINE"]:
    # Actor
    if MODE == "REINFORCE":
        actor = ActorReinforce(env.observation_space.shape[0], env.action_space.n, 2 ** 5, 2 ** 5, 0.00001)
    elif MODE == "BASELINE":
        actor = ActorBaseline(env.observation_space.shape[0], env.action_space.n, 2 ** 5, 2 ** 5, 0.00001, 0.00001)
    elif MODE == "CONSTANT_BASELINE":
        actor = ActorConstant(env.observation_space.shape[0], env.action_space.n, 2 ** 5, 2 ** 4, 0.00001, 0.00001)
    else:
        actor = ActorRandom(env.action_space.n)

    # Training
    e, episodes, total_returns, alpha_decay = 1, [], [], 1.0
    while e < MAX_EPISODES:

        # Generate episode
        state = np.reshape(env.reset(), (1, -1)).astype(np.float32)  # s_0
        action, grads = actor.act(state)  # a_0, grads_0
        action = action.numpy()

        if render:
            env.render(mode='human')

        states, next_states, actions, next_actions, rewards, gradients = [], [], [], [], [], []
        done, t = False, 0
        while not done and t < MAX_STEPS:  # t = 0, 1, 2, ..., T-1
            states.append(state)  # s_t
            actions.append(action)  # a_t
            gradients.append(grads)  # grads_t

            t = t + 1
            next_state, reward, done, _ = env.step(action)  # s_t+1, r_t+1
            next_state = np.reshape(next_state, (1, -1)).astype(np.float32)
            rewards.append(reward)  # r_t+1
            next_states.append(next_state)  # s_t+1

            next_action, next_grads = actor.act(next_state)  # a_t+1, grads_t+1
            next_action = next_action.numpy()
            next_actions.append(next_action)

            state, action, grads = next_state, next_action, next_grads

            if render:
                time.sleep(1.0 / FPS)
                env.render()

        e_length = t
        env.close()

        total_return, returns = compute_returns(np.array(rewards), GAMMA)

        # Update weights
        if MODE == "BASELINE" or MODE == "CONSTANT_BASELINE":
            for t in range(e_length):
                return_t, grads, state = returns[t], gradients[t], states[t]

                # Updates
                state_value, v_grads = actor.state_value(state)
                state_value = state_value.numpy()
                delta = return_t - state_value

                actor.apply_gradients_b(v_grads, tf.Variable(delta * alpha_decay))  # Update baseline weights
                actor.apply_gradients_a(grads,
                                        tf.Variable(delta * np.power(GAMMA, t) * alpha_decay))  # Update actor weights
        else:
            for t in range(e_length):
                return_t, grads_t = returns[t], gradients[t]
                actor.apply_gradients(grads_t, tf.Variable(return_t * np.power(GAMMA, t) * alpha_decay))

        print("e {:<20} return {:<20} length {:<20}".format(e, np.round(total_return, decimals=3), len(states)))
        total_returns.append(total_return), episodes.append(e)

        alpha_decay = np.exp(- e / DECAY_PERIOD)
        e = e + 1

    episodes = np.array(episodes)
    total_returns = np.array(total_returns)
    average_returns = pd.Series(total_returns).rolling(100, min_periods=1).mean().values

    results.append((episodes, total_returns, average_returns, MODE))

fig, ax = plt.subplots(1, 2, figsize=(16, 5))
for result in results:
    episodes, total_returns, average_returns, mode = result
    ax[0].plot(episodes, total_returns, label=f"{mode}")
    ax[1].plot(episodes, average_returns, label=f"avg_{mode}")

ax[0].set_title("returns")
ax[1].set_title("average returns")
ax[0].set_ylim(bottom=0)
ax[1].set_ylim(bottom=0)
ax[0].legend()
ax[1].legend()
fig.savefig("./results/{}_pg_result.png".format(datetime.datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S")))
fig.show()

end = time.time()
print(f"Finished in {end - start} s")
