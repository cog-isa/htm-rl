import matplotlib.pyplot as plt
import numpy as np


class RunStats:
    steps: np.array
    rewards: np.array
    times: np.array

    def __init__(self, n_episodes):
        self.steps = np.array(n_episodes, dtype=np.int)
        self.rewards = np.array(n_episodes, dtype=np.float)
        self.times = np.array(n_episodes, dtype=np.float)
        self._i = 0

    def append_stats(self, steps, total_reward, elapsed_time):
        i = self._i
        self.steps[i] = steps
        self.rewards[i] = total_reward
        self.times[i] = elapsed_time
        self._i += 1


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

