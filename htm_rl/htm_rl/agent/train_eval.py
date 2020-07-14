from typing import List

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from htm_rl.agent.agent import Agent
from htm_rl.agent.memory import Memory
from htm_rl.common.base_sar import Sar
from htm_rl.common.utils import trace


def train(agent: Agent, env, n_episodes: int, max_steps: int, verbose: bool):
    episodes_len = np.empty(n_episodes, dtype=np.int)
    rewards = np.empty(n_episodes, dtype=np.float)
    for episode in trange(n_episodes):
        episode_len, reward = train_episode(agent, env, max_steps, verbose)

        episodes_len[episode] = episode_len
        rewards[episode] = reward
        trace(verbose, '')
    return episodes_len, rewards


def train_episode(agent: Agent, env, max_steps: int, verbose: bool):
    agent.reset()
    state, reward, done = env.reset(), 0, False
    for step in range(max_steps + 1):
        action = agent.make_step(state, reward, verbose)
        if step == max_steps or done:
            return step, reward
        state, reward, done, info = env.step(action)


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


def plot_train_results(episode_lens, rewards, plot_title, episodes_title, rewards_title):
    fig, [ax1, ax2] = plt.subplots(nrows=2, figsize=(12, 10), sharex=True)

    xs = np.arange(episode_lens.size)

    fig.suptitle(plot_title)
    ax1.plot(xs, episode_lens)
    ax1.axhline(episode_lens.mean(), color='r')
    ax1.set_title(episodes_title)

    ax2.plot(xs, rewards)
    ax2.axhline(rewards.mean(), color='r')
    ax2.set_title(rewards_title)
    plt.show()

