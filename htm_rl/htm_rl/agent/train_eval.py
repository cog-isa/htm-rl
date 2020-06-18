import numpy as np
import matplotlib.pyplot as plt

from htm_rl.agent.agent import Agent
from htm_rl.common.base_sar import Sar
from htm_rl.common.utils import trace


def train(agent: Agent, env, n_episodes: int, max_steps: int, verbose: bool):
    reward_reached = 0
    for episode in range(n_episodes):
        reward_reached += train_episode(agent, env, max_steps, verbose)
        agent.tm.reset()
        trace(verbose, '')

    trace(verbose, f'Reward reached: {reward_reached}')


def train_episode(agent: Agent, env, max_steps: int, verbose: bool):
    observation, reward, done = env.reset(), 0, False

    for step in range(max_steps + 1):
        action = np.random.choice(env.n_actions)
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
