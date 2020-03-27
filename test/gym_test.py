import argparse
import itertools
import time

import numpy as np

from gym import envs

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="random")
parser.add_argument("--max_steps", type=int, default=100)
parser.add_argument("--max_episodes", type=int, default=10)
parser.add_argument("--fps", type=int, default=100)
args = parser.parse_args()

# env_name = "MountainCar-v0"
env_name = "CartPole-v1"
# env_name = "RoadRunner-ram-v0"

env = envs.make(env_name)
action_space = env.action_space

print(env.action_space)
print(env.observation_space)

t = None
done = False
ep = 1

while ep != args.max_episodes:

    # Initial observation (state)
    observation = env.reset()
    env.render(mode='human')

    # Episode
    print("Starting a new trajectory")
    done = False
    for t in range(args.max_steps) if args.max_steps else itertools.count():
        time.sleep(1.0 / args.fps)

        # Random action
        action = action_space.sample()

        # Step
        # _, _, done, _ = env.step(a)
        observation_, reward, done, info = env.step(action)  # s', r, done, info

        if t % 10 == 0:
            print(f"t = {t}:\t\t\ts = {np.around(observation, decimals=2)}\t\t\ta = {action}\t\t\tr = {reward}\t\t\t"
                  f"s' = {np.around(observation_, decimals=2)}")
        env.render()
        observation = observation_  # s <- s'
        if done:
            print(f"Done after {t + 1} steps")
            break
    ep = ep + 1

env.close()