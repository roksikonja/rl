import argparse
import time

import numpy as np


def episode_rollout(env, actor, max_timesteps, render=False, fps=100):
    state = env.reset()
    if render:
        env.render(mode="human")

    states = []
    actions = []
    rewards = []

    for t in range(max_timesteps):  # t = 0, 1, 2, ..., T-1
        states.append(state)  # s_t

        action = actor.act(state)
        actions.append(action)  # a_t

        next_state, reward, done, _ = env.step(action)  # s_t+1, r_t+1
        # if done:
        #     reward = -reward

        rewards.append(reward)  # r_t+1

        # (state, action, reward, next_states)  (s_t, a_t, r_t+1, s_t+1)
        state = next_state

        if render:
            time.sleep(1.0 / fps)
            env.render()

        if done:
            break

    # states: s_0, s_1, s_2, ..., s_T-1
    # actions: a_0, a_1, ..., a_T-1
    # rewards: r_1, r_2, ..., r_T
    return np.array(states), np.array(actions), np.array(rewards)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iters", default=100, type=int, help="Number of iterations.")
    parser.add_argument(
        "--n_samples",
        default=50,
        type=int,
        help="Number of samples CEM algorithm chooses from on each iteration.",
    )
    parser.add_argument(
        "--max_timesteps", default=500, type=int, help="Maximum episode length."
    )
    parser.add_argument(
        "--best_frac",
        default=0.2,
        type=float,
        help="Fraction of top samples used to calculate mean and variance of next iteration",
    )
    parser.add_argument(
        "--fps", default=100, type=int, help="Rendering FPS."
    )
    return parser.parse_args()


class CEMAgent(object):

    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions

        # Initialize linear policy parameters
        self.mean, self.variance = None, None
        self.W, self.b = None, None
        self.initialize_policy()

    def initialize_policy(self):
        """
            Initialize policy weights W and b.
        """
        self.mean = np.random.randn(self.n_states + 1)
        self.variance = np.square(0.1 * np.ones_like(self.mean))
        Wb = np.random.multivariate_normal(self.mean, np.diag(self.variance))  # (n_states + 1, )

        self.set_policy(Wb)

    def act(self, state):
        """
            Linear policy.

            a = sign(s^T W + b)
        """
        logits = np.matmul(state, self.W) + self.b  # (None, )
        a = np.greater_equal(logits, 0).astype(np.int)  # (None, )
        return a

    def set_policy(self, Wb):
        """
            Update policy weights given using a weight sample.
        """
        self.W = Wb[:-1]  # (n_states, )
        self.b = Wb[-1]  # ()
