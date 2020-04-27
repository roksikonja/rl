import numpy as np
import tensorflow as tf
from gym import envs

from lib.pgt_utils import ActorReinforce, pgt_gradients
from lib.rl_utils import compute_returns, episode_rollout
from lib.visualizer import Visualizer

print(tf.__version__)
visualizer = Visualizer()

MAX_STEPS = 300
MAX_EPISODES = 2000
GAMMA = 0.995
FPS = 100
BATCH = 1
DECAY_PERIOD = 1000
ENV_NAME = "CartPole-v1"
render = False

# Environment
env = envs.make(ENV_NAME)
actor = ActorReinforce(
    env.observation_space.shape[0], env.action_space.n, (16, 16), 0.0001
)

# Training
e, episodes, total_returns, alpha_decay = 1, [], [], 1.0
t_grads = actor.initialize_gradients()  # Initialize zero gradients
while e < MAX_EPISODES:
    # if e == MAX_EPISODES - 50:
    #     render = True

    # Generate episode
    _, _, _, rewards, gradients = episode_rollout(env, actor, MAX_STEPS, render)
    total_return, returns = compute_returns(np.array(rewards), GAMMA)

    # REINFORCE I - PGT
    # for t in range(len(gradients)):
    #     return_t, grads_t = returns[t], gradients[t]
    #     actor.apply_gradients(
    #         grads_t, tf.Variable(return_t * np.power(GAMMA, t) * alpha_decay)
    #     )

    t_grads = pgt_gradients(
        t_grads, gradients, tf.Variable(returns, dtype=tf.float32), GAMMA
    )

    # REINFORCE II
    # for g in range(len(t_grads)):
    #     t_grads[g] = t_grads[g] + tf.add_n([grads[g] for grads in gradients]) * total_return

    # GPOMDP
    # for g in range(len(t_grads)):
    #     tmp = []
    #     for t in range(len(gradients)):
    #         tmp.append(tf.add_n([gradients[k][g] for k in range(t + 1)]) * np.power(GAMMA, t) * returns[t])
    #     t_grads[g] = t_grads[g] + tf.add_n(tmp)

    total_returns.append(total_return), episodes.append(e)

    print(
        "e {:<20} return {:<20} length {:<20}".format(
            e, np.round(total_return, decimals=3), len(rewards)
        )
    )

    if e % BATCH == 0:
        # print(sum([np.linalg.norm(grad.numpy()) for grad in t_grads]))

        actor.apply_gradients(t_grads, tf.Variable(1.0 / BATCH * alpha_decay))
        t_grads = actor.initialize_gradients()

    alpha_decay = np.exp(-e / DECAY_PERIOD)
    e = e + 1

env.close()

visualizer.plot_signals(
    [(np.array(episodes), np.array(total_returns), "REINFORCE")], y_lim=[0, None]
).show()

visualizer.plot_signals_mean_std(
    [(np.array(episodes), np.array(total_returns), "REINFORCE", 100)], y_lim=[0, None]
).show()
