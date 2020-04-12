import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from gym import envs

from lib.ac_utils import ActorCritic
from lib.rl_utils import compute_returns

print(tf.__version__)

if __name__ == "__main__":
    start = time.time()

    MODE = "AC"
    MAX_STEPS = 500
    MAX_EPISODES = 2000
    GAMMA = 0.999
    FPS = 100
    DECAY_PERIOD = 2000
    ENV_NAME = "CartPole-v1"
    render = False

    # Environment
    env = envs.make(ENV_NAME)

    actor_critic = ActorCritic(
        env.observation_space.shape[0],
        env.action_space.n,
        2 ** 4,
        2 ** 5,
        0.0001,
        0.001,
    )
    # Training
    e, episodes, total_returns, alpha_decay = 1, [], [], 1.0
    for e in range(1, MAX_EPISODES + 1):
        # Generate episode
        state = np.reshape(env.reset(), (1, -1)).astype(np.float32)  # s_0
        action, grads = actor_critic.act(state)  # a_0, grads_0
        action = action.numpy()

        if render:
            env.render(mode="human")

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

            actor_critic.apply_gradients_c(
                v_grads, tf.Variable(delta * alpha_decay)
            )  # Update critic weights
            actor_critic.apply_gradients_a(
                grads, tf.Variable(delta * np.power(GAMMA, t) * alpha_decay)
            )  # Update actor weights

            # Act
            next_action, next_grads = actor_critic.act(next_state)  # a_t+1, grads_t+1
            next_action = next_action.numpy()

            state, action, grads = next_state, next_action, next_grads

            if render:
                time.sleep(1.0 / FPS)
                env.render()

        e_length = t

        total_return, returns = compute_returns(np.array(rewards), GAMMA)
        print(
            "e {:<20} return {:<20} length {:<20}".format(
                e, np.round(total_return, decimals=3), e_length
            )
        )
        total_returns.append(total_return), episodes.append(e)

        alpha_decay = np.exp(-e / DECAY_PERIOD)

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
    fig.savefig(
        "./results/{}_ac_result.png".format(
            datetime.datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S")
        )
    )
    fig.show()

    end = time.time()
    print(f"Finished in {end - start} s")
