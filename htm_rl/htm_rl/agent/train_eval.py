import os
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from htm_rl.common.utils import trace


class RunStats:
    steps: np.array
    rewards: np.array
    times: np.array

    def __init__(self, n_episodes):
        self.steps = np.empty(n_episodes, dtype=np.int)
        self.rewards = np.empty(n_episodes, dtype=np.float)
        self.times = np.empty(n_episodes, dtype=np.float)
        self._i = 0

    def append_stats(self, steps, total_reward, elapsed_time):
        i = self._i
        self.steps[i] = steps
        self.rewards[i] = total_reward
        self.times[i] = elapsed_time
        self._i += 1


class RunResultsProcessor:
    env_name: str
    optimal_len: int
    test_dir: str
    moving_average: int
    verbose: bool

    data_ext = '.csv.tar.gz'

    def __init__(self, env_name: str, optimal_len: int, test_dir: str, moving_average: int, verbose: bool):
        self.env_name = env_name
        self.optimal_len = optimal_len
        self.test_dir = test_dir
        self.moving_average = moving_average
        self.verbose = verbose

    def store_result(self, run_stats: RunStats, agent_info: str):
        result_table = pd.DataFrame({
            'steps': run_stats.steps,
            'times': run_stats.times,
            'rewards': run_stats.rewards
        })
        file_name = f'{self.env_name}_{agent_info}{self.data_ext}'
        out_file_path = os.path.join(self.test_dir, file_name)
        result_table.to_csv(out_file_path)

        self.print_results(run_stats)

    def aggregate_results(self):
        csv_files_mask = f'*{self.data_ext}'
        dfs = [
            (file_path, pd.read_csv(file_path))
            for file_path in sorted(glob(os.path.join(self.test_dir, csv_files_mask)))
        ]
        col_names = self._get_names(dfs)
        df_steps: pd.DataFrame = pd.concat(self._get_columns(dfs, 'steps'), axis=1, keys=col_names)
        df_times: pd.DataFrame = pd.concat(self._get_columns(dfs, 'times'), axis=1, keys=col_names)
        df_rewards: pd.DataFrame = pd.concat(self._get_columns(dfs, 'rewards'), axis=1, keys=col_names)
        df_rewards *= self.optimal_len / df_steps

        ma = self.moving_average
        self._plot_figure(df_rewards, f'Episode rewards, fraction of optimal, MA={ma}', 'rewards')
        self._plot_figure(df_steps, f'Episode lengths, steps, MA={ma}', 'steps')
        self._plot_figure(df_times, f'Episode durations, sec, MA={ma}', 'times')
        plt.show()

    def _plot_figure(self, df: pd.DataFrame, title: str, fname):
        fig: plt.Figure
        ax: plt.Axes
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        df_ma = df.rolling(window=self.moving_average).mean()
        df_ma.plot(use_index=True, ax=ax)

        ax.legend(loc='right', bbox_to_anchor=(1.2, .5))
        ax.set_title(title)
        fig.tight_layout(h_pad=4.)
        fig.show()

        save_path = os.path.join(self.test_dir, f'fig_{fname}.png')
        fig.savefig(save_path, dpi=120)

    def _get_names(self, dfs):
        env_name_len = len(self.env_name) + 1
        ext_len = len(self.data_ext)

        def path_to_short_name(file_path):
            p = Path(file_path)
            return p.name[env_name_len:-ext_len]

        return [path_to_short_name(file_path) for file_path, _ in dfs]

    def _get_columns(self, dfs, col):
        return [df[col] for _, df in dfs]

    def print_results(self, run_stats):
        avg_len = run_stats.steps.mean()
        avg_reward = (run_stats.rewards * (self.optimal_len / run_stats.steps)).mean()
        avg_time = run_stats.times.mean()
        elapsed = run_stats.times.sum()
        trace(
            self.verbose,
            f'AvgLen: {avg_len: .2f}  AvgReward: {avg_reward: .5f}  '
            f'AvgTime: {avg_time: .6f}  TotalTime: {elapsed: .6f}'
        )


def plot_anomalies(anomalies):
    fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(12, 6))
    xs = np.arange(len(anomalies))
    ax1.plot(xs, anomalies)
    ax2.hist(anomalies)
    plt.show()

