import time

import numpy as np
import tensorflow as tf
from gpflow.utilities.utilities import tabulate_module_summary


def pilco_rollout(
    env, actor, max_timesteps, num_episodes=1, verbose=False, render=False, fps=100
):
    X = []
    Y = []
    for e in range(num_episodes):
        state = env.reset()
        if render:
            env.render(mode="human")

        state_actions = []  # X
        targets = []  # Y

        for t in range(max_timesteps):  # t = 0, 1, 2, ..., T-1
            action = actor.policy(state)

            next_state, reward, done, _ = env.step(action)  # s_t+1, r_t+1

            state_actions.append(np.hstack((state, action)))  # \tilde{x}_t
            targets.append(next_state - state)  # \Delta_t

            # if verbose:
            #     print("Action:\t", action, action.shape)
            #     print("Next state:\t", next_state, next_state.shape)

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
        state_actions = np.stack(state_actions)
        targets = np.stack(targets)

        if verbose:
            print(
                f"e = {e}\tX {str(state_actions.shape)}\ttargets {str(targets.shape)}"
            )

        X.append(state_actions)
        Y.append(targets)

    env.close()

    return np.vstack(X).astype(np.float32), np.vstack(Y).astype(np.float32)
    # return np.vstack(X), np.vstack(Y)


def get_summary(module: tf.Module):
    """
    Returns a summary of the parameters and variables contained in a tf.Module.
    """
    return tabulate_module_summary(module, None)
