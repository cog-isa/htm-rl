from typing import List

import numpy as np
import matplotlib.pyplot as plt

from htm_rl.agent.agent import Agent
from htm_rl.agent.memory import Memory
from htm_rl.common.base_sar import Sar
from htm_rl.common.utils import trace


def train(agent: Agent, env, n_episodes: int, max_steps: int, verbose: bool):
    reward_reached = 0
    for episode in range(n_episodes):
        reward_reached += train_episode(agent, env, max_steps, verbose)
        trace(verbose, '')
    trace(True, f'Reward reached: {reward_reached} of {n_episodes}')


def train_episode(agent: Agent, env, max_steps: int, verbose: bool):
    agent.reset()
    state, reward, done = env.reset(), 0, False
    for step in range(max_steps + 1):
        action = agent.make_step(state, reward, verbose)
        if step == max_steps or done:
            break
        state, reward, done, info = env.step(action)
    return reward


def train_trajectories(agent: Memory, env, trajectories: List[List[int]], verbose: bool):
    reward_reached = 0
    for trajectory_actions in trajectories:
        trace(verbose, '>')
        reward_reached += train_trajectory(agent, env, trajectory_actions, verbose)
        agent.tm.reset()
        trace(verbose, '<')

    trace(True, f'Reward reached: {reward_reached} of {len(trajectories)}')


def train_trajectory(agent: Memory, env, actions: List[int], verbose: bool):
    observation, reward, done = env.reset(), 0, False

    max_steps = len(actions)
    actions = actions + [0]
    for step in range(max_steps + 1):
        action = actions[step]
        agent.train(Sar(observation, action, reward), verbose)

        if step == max_steps or done:
            break

        next_observation, reward, done, info = env.step(action)
        observation = next_observation
    return reward


def plot_anomalies(anomalies):
    fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(12, 6))
    xs = np.arange(len(anomalies))
    ax1.plot(xs, anomalies)
    ax2.hist(anomalies)
    plt.show()
