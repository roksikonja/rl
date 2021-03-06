import time

import numpy as np


def compute_returns(rewards, gamma=1.0):
    assert 0 <= gamma <= 1.0
    if gamma == 1.0:
        returns = np.cumsum(rewards)[::-1]
    elif gamma == 0:
        returns = rewards
    else:
        returns = np.zeros_like(rewards)
        g = 0  # G_T
        for t in reversed(range(returns.shape[0])):  # T-1, T-2, ..., 0
            r = rewards[t]  # r_t+1
            g = r + gamma * g  # G_t = r_t+1 + gamma * G_t+1
            returns[t] = g

    total_return = returns[0]
    return total_return, returns  # G_0, G_1, ..., G_T-1


def episode_rollout(env, actor, max_timesteps, render=False, fps=100):
    state = env.reset()
    if render:
        env.render(mode="human")

    states = []
    next_states = []
    actions = []
    rewards = []
    gradients = []

    for t in range(max_timesteps):  # t = 0, 1, 2, ..., T-1
        states.append(state)  # s_t

        state = np.reshape(state, (1, -1)).astype(np.float32)
        action, grads = actor.act(state)
        action = action.numpy()

        actions.append(action)  # a_t
        gradients.append(grads)  # grad_t

        next_state, reward, done, _ = env.step(action)  # s_t+1, r_t+1
        if done:
            reward = -reward

        rewards.append(reward)  # r_t+1
        next_states.append(next_state)

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
    return (
        np.array(states),
        np.array(next_states),
        np.array(actions),
        np.array(rewards),
        gradients,
    )


def rollout(env, pilco, max_timesteps, verbose=True, random=False, SUBS=1, render=True):
    X = []
    Y = []
    x = env.reset()
    for timestep in range(max_timesteps):
        if render:
            env.render()
        u = policy(env, pilco, x, random)
        x_new, done = None, None
        for i in range(SUBS):
            x_new, _, done, _ = env.step(u)
            if done:
                break
            if render:
                time.sleep(1.0 / 50)
                env.render()
        if verbose:
            print("Action: ", u, u.shape)
            print("State : ", x_new, x_new.shape)
        X.append(np.hstack((x, u)))
        Y.append(x_new - x)
        x = x_new
        if done:
            break
    return np.stack(X), np.stack(Y)


def policy(env, pilco, x, random):
    if random:
        return env.action_space.sample()
    else:
        return pilco.compute_action(x[None, :])[0, :]
