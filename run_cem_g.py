"""
    CE performs random noisy evaluations, i.e. episode rollouts, and filters best performing trajectories.
    It then trains the model (NN) weights, mapping observations to actions, using state s_t as input and a_t as
    target and using the cross-entropy loss.
"""
import gym
import numpy as np

from lib.cem_utils import parse_arguments, episode_rollout, RandomAgent, CEMAgentG
from lib.rl_utils import compute_returns

env = gym.make("CartPole-v1")

actor = CEMAgentG(env.observation_space.shape[0], env.action_space.n, (5,), 0.01)
random_actor = RandomAgent(env)
args = parse_arguments()
best_n_samples = int(np.ceil(args.best_frac * args.n_samples))

print("TRAINING")
for i in range(args.n_iters):
    sample_returns = []
    sample_states = []
    sample_actions = []
    for j in range(args.n_samples):
        # Off- or on-policy? Better works in off-policy mode. Why?
        if j % 20 != 0 or True:
            sample_actor = random_actor
        else:
            sample_actor = actor

        states, actions, rewards = episode_rollout(env, sample_actor, args.max_timesteps)
        total_return, returns = compute_returns(rewards)

        sample_states.append(states)
        sample_actions.append(actions)
        sample_returns.append(total_return)

    sample_returns = np.array(sample_returns)
    best_indices = np.argsort(sample_returns)[::-1][:best_n_samples]

    data = np.vstack([sample_states[k] for k in best_indices])
    targets = np.vstack([np.reshape(sample_actions[k], (-1, 1)) for k in best_indices])

    # Learn policy
    actor.fit(data, targets, verbose=0, epochs=1)

    # Test actor
    _, _, rewards = episode_rollout(env, actor, args.max_timesteps, render=True, fps=args.fps)
    total_return, _ = compute_returns(rewards)
    print(f"{i}: {sample_returns.mean()} {total_return}")

print("\nTESTING")
for i in range(10):
    _, _, rewards = episode_rollout(
        env, actor, args.max_timesteps, render=True, fps=args.fps
    )
    total_return, _ = compute_returns(rewards)
    print(f"{i}: {total_return}")

env.close()
