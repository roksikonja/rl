import gym
import numpy as np

from lib.cem_utils import CEMAgent
from lib.cem_utils import parse_arguments, episode_rollout
from lib.rl_utils import compute_returns

if __name__ == "__main__":
    args = parse_arguments()

    # np.random.seed(0)
    env = gym.make("CartPole-v1")
    actor = CEMAgent(env.observation_space.shape[0], env.action_space.n)

    # states (T, 4)
    # actions (T, )
    # rewards (T, )
    # returns (T, )

    print("TRAINING")
    best_n_samples = int(np.round(args.best_frac * args.n_samples))
    for i in range(args.n_iters):
        samples_Wb = np.random.multivariate_normal(actor.mean, np.diag(actor.variance),
                                                   args.n_samples)  # (n_samples, n_states + 1)

        sample_returns = []
        for j in range(args.n_samples):
            sample_Wb = samples_Wb[j, :]
            actor.set_policy(sample_Wb)

            _, _, rewards = episode_rollout(env, actor, args.max_timesteps)
            total_return, _ = compute_returns(rewards)

            sample_returns.append(total_return)

        sample_returns = np.array(sample_returns)

        best_indices = np.argsort(sample_returns)[::-1][:best_n_samples]
        best_samples = samples_Wb[best_indices, :]

        # Noise is added - for exploration, similar role as in particle filter
        v = np.max([5 - i / 10, 0])

        mean = best_samples.mean(axis=0)  # (n_states + 1, )
        variance = best_samples.var(axis=0) + v  # (n_states + 1, )
        
        # Update actor's mean and variance
        actor.mean = mean
        actor.variance = variance + v

        # Test
        # _, _, rewards = episode_rollout(env, actor, args.max_timesteps, render=True, fps=100)
        # total_return, _ = compute_returns(rewards)
        total_return = 1
        print(f"{i}: {sample_returns.mean()} {total_return}")

    print("\nTESTING")
    for i in range(10):
        _, _, rewards = episode_rollout(env, actor, args.max_timesteps, render=True, fps=100)
        total_return, _ = compute_returns(rewards)
        print(f"{i}: {total_return}")

    env.close()
