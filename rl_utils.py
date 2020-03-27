import time

import numpy as np


def compute_returns(rewards, gamma):
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


def episode_rollout(env, actor, max_steps, render=False, fps=100):
    state = env.reset()
    if render:
        env.render(mode='human')

    states, next_states, actions, rewards, dones, gradients, memory = [], [], [], [], [], [], []
    done, t = False, 0
    while not done and t < max_steps:  # t = 0, 1, 2, ..., T-1
        states.append(state)  # s_t

        state = np.reshape(state, (1, -1)).astype(np.float32)
        action, grads = actor.act(state)
        actions.append(action)  # a_t
        gradients.append(grads)

        t = t + 1
        next_state, reward, done, _ = env.step(action)  # s_t+1, r_t+1
        rewards.append(reward)  # r_t+1
        dones.append(done)
        next_states.append(next_state)

        memory.append((state, action, reward, next_states))  # (s_t, a_t, r_t+1, s_t+1)
        state = next_state

        if render:
            time.sleep(1.0 / fps)
            env.render()

    env.close()
    # states: s_0, s_1, s_2, ..., s_T-1
    # actions: a_0, a_1, ..., a_T-1
    # rewards: r_1, r_2, ..., r_T
    return t, np.array(states), np.array(next_states), np.array(actions), np.array(rewards), np.array(dones), \
           gradients, memory
