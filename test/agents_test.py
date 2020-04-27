"""
    DQN, tf_agent tutorial. https://github.com/tensorflow/agents/blob/master/docs/tutorials/1_dqn_tutorial.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

print(tf.version.VERSION)

num_iterations = 10000
initial_collect_steps = 1000
collect_steps_per_iteration = 1
replay_buffer_max_length = 100000
batch_size = 64
learning_rate = 1e-3
log_interval = 200
num_eval_episodes = 10
eval_interval = 500

env_name = "CartPole-v1"
env = suite_gym.load(env_name)
env.reset()

"""
    Helpers.
"""


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step_ = environment.reset()
        episode_return = 0.0

        while not time_step_.is_last():
            action_step = policy.action(time_step_)
            time_step_ = environment.step(action_step.action)
            episode_return += time_step_.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def collect_step(environment, policy, buffer):
    time_step_ = environment.current_time_step()
    action_step = policy.action(time_step_)
    next_time_step_ = environment.step(action_step.action)
    traj_ = trajectory.from_transition(time_step_, action_step, next_time_step_)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj_)


def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)


print("Observation Spec:")
print(env.time_step_spec().observation)

print("Reward Spec:")
print(env.time_step_spec().reward)

print("Action Spec:")
print(env.action_spec())

time_step = env.reset()
print("Time step:")
print(time_step)

print(time_step.step_type)
print(time_step.reward)
print(time_step.discount)
print(time_step.observation)

action = np.array(1, dtype=np.int32)

next_time_step = env.step(action)
print("Next time step:")
print(next_time_step)

"""
    Environments.
"""
train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

"""
    Agent.
"""
q_net = q_network.QNetwork(
    train_env.observation_spec(), train_env.action_spec(), fc_layer_params=(100,),
)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter,
)

agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy
random_policy = random_tf_policy.RandomTFPolicy(
    train_env.time_step_spec(), train_env.action_spec()
)

example_environment = tf_py_environment.TFPyEnvironment(suite_gym.load("CartPole-v1"))

time_step = example_environment.reset()
random_policy.action(time_step)

compute_avg_return(eval_env, random_policy, num_eval_episodes)

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length,
)

print(agent.collect_data_spec)

collect_data(train_env, random_policy, replay_buffer, steps=100)

# Dataset generates trajectories with shape [Bx2x...]
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2
).prefetch(3)

iterator = iter(dataset)
print(next(iterator))

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

    # Collect a few steps using collect_policy and save to the replay buffer.
    for _ in range(collect_steps_per_iteration):
        collect_step(train_env, agent.collect_policy, replay_buffer)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print("step = {0}: loss = {1}".format(step, train_loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        print("step = {0}: Average Return = {1}".format(step, avg_return))
        returns.append(avg_return)

iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, returns)
plt.ylabel("Average Return")
plt.xlabel("Iterations")
plt.ylim(top=250)
plt.show()

action = np.array(1, dtype=np.int32)
time_step = env.reset()
print(time_step)
while not time_step.is_last():
    time_step = env.step(action)
    print(time_step)
