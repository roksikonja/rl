import time

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import envs

from lib.oslo_utils import CohenController

gym.logger.set_level(40)

if __name__ == "__main__":
    MAX_STEPS, MAX_EPISODES = 300, 10
    # ENV_NAME, GAMMA = "InvertedPendulum-v2", 0.999
    ENV_NAME, GAMMA = "Pendulum-v0", 0.999
    RENDER, FPS = True, 100

    # Environment
    env = envs.make(ENV_NAME)
    print(env.action_space, env.action_space.low, env.action_space.high)
    print(env.observation_space, env.observation_space.low, env.observation_space.high)

    d, k = env.observation_space.shape[0], env.action_space.shape[0]
    n = d + k

    sigma_w = 1 / d
    W = sigma_w ** 2 * np.identity(d)

    theta = 0.1
    alpha_0 = 1e1
    alpha_1 = 2e1
    nu = 10
    T = 1000
    delta = 0.1

    # Q = np.diag(np.linspace(alpha_0, alpha_1, d))
    # R = np.diag(np.linspace(alpha_0, alpha_1, k))
    Q, R = np.eye(d), np.eye(k)
    A0, B0 = 1e-12 * np.ones(shape=(d, d)), 1e-12 * np.ones(shape=(d, k))

    print(Q.shape, R.shape)
    controller = CohenController(
        Q=Q, R=R, system=env, sigma=sigma_w, theta=theta, nu=nu, time_horizon=3000
    )

    observation = controller.system.reset()

    if RENDER:
        controller.system.render(mode="human")

    for _ in range(controller.time_horizon):
        action = controller.step(observation=observation)
        observation, _, done, _ = controller.system.step(action)

        if RENDER:
            time.sleep(1.0 / FPS)
            controller.system.render()

        if done:
            print("The termination point was reached.")
            observation = controller.system.reset()

    # Plot COHEN
    plt.plot(np.flip(np.linalg.norm(controller._x_history, axis=1)), label="path")
    plt.plot(
        np.flip(np.linalg.norm(controller._z_history[:, d:], axis=1)), label="action"
    )
    plt.yscale("log")
    plt.legend()
    plt.show()
