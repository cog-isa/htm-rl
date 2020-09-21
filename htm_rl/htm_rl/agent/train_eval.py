import os
from glob import glob
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from htm_rl.common.utils import trace


class RunStats:
    steps: List[int]
    rewards: List[float]
    times: List[float]

    def __init__(self):
        self.steps = []
        self.rewards = []
        self.times = []

    def append_stats(self, steps, total_reward, elapsed_time):
        self.steps.append(steps)
        self.rewards.append(total_reward)
        self.times.append(elapsed_time)


class RunResultsProcessor:
    env_name: str
    test_dir: str
    moving_average: int
    verbosity: int

    data_ext = '.csv.tar.gz'

    def __init__(
            self, env_name: str, test_dir: str, moving_average: int, verbosity: int
    ):
        self.env_name = env_name
        self.test_dir = test_dir
        self.moving_average = moving_average
        self.verbosity = verbosity

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

    def aggregate_results(self, file_masks, report_suffix, silent: bool):
        if not file_masks:
            file_masks = ['*']

        csv_files_masks = [
            os.path.join(self.test_dir, f'{self.env_name}_{file_mask}{self.data_ext}')
            for file_mask in file_masks
        ]
        trace(self.verbosity, 2, '\n'.join(csv_files_masks))
        dfs = [
            (file_path, pd.read_csv(file_path))
            for csv_file_mask in csv_files_masks
            for file_path in sorted(glob(csv_file_mask))
        ]
        col_names = self._get_names(dfs)
        df_steps: pd.DataFrame = pd.concat(self._get_columns(dfs, 'steps'), axis=1, keys=col_names)
        df_times: pd.DataFrame = pd.concat(self._get_columns(dfs, 'times'), axis=1, keys=col_names)
        df_rewards: pd.DataFrame = pd.concat(self._get_columns(dfs, 'rewards'), axis=1, keys=col_names)

        rel_col = df_steps.columns[0]
        if 'htm_0' in df_steps:
            rel_col = 'htm_0'
        df_steps_rel = df_steps.div(df_steps[rel_col], axis=0)
        df_steps_rel = np.log(df_steps_rel)

        report_name = f'{self.env_name}'
        if report_suffix:
            report_name += f'__{report_suffix}'

        ma = self.moving_average
        self._plot_figure(df_times, f'episode execution time, sec, MA={ma}', report_name, 'times')
        self._plot_figure(df_rewards, f'episode reward, MA={ma}', report_name, 'rewards')
        self._plot_figure(df_steps, f'episode duration, steps, MA={ma}', report_name, 'steps')
        self._plot_figure(
            df_steps_rel,
            f'episode duration in steps log-relative to `{rel_col}`, MA={ma}', report_name,
            f'steps_rel_{rel_col}'
        )
        if not silent:
            plt.show()

    def _plot_figure(self, df: pd.DataFrame, title: str, report_name, fname):
        fig: plt.Figure
        ax: plt.Axes
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        df_ma = df.rolling(window=self.moving_average).mean()
        df_ma.plot(use_index=True, ax=ax)

        ax.legend(loc='right', bbox_to_anchor=(1.2, .5))
        ax.set_title(f'{report_name}: {title}')
        fig.tight_layout(h_pad=4.)

        save_path = os.path.join(self.test_dir, f'{report_name}__{fname}.svg')
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
        avg_len = np.array(run_stats.steps).mean()
        avg_reward = np.array(run_stats.rewards).mean()
        avg_time = np.array(run_stats.times).mean()
        elapsed = np.array(run_stats.times).sum()
        trace(
            self.verbosity, 1,
            f'AvgLen: {avg_len: .2f}  AvgReward: {avg_reward: .5f}  '
            f'AvgTime: {avg_time: .6f}  TotalTime: {elapsed: .6f}'
        )

    def store_environment_maps(self, maps):
        for i, (env_map, seed) in enumerate(maps):
            n = env_map.shape[0]

            fig: plt.Figure
            ax: plt.Axes
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax.set_xticks(np.arange(-.5, n, 1))
            ax.set_yticks(np.arange(-.5, n, 1))
            ax.set_xticklabels(np.arange(n))
            ax.set_yticklabels(np.arange(n))
            ax.grid(color='grey', linestyle='-', linewidth=1)
            ax.set_title(f'{self.env_name}, seed={seed}')

            ax.imshow(env_map)
            save_path = os.path.join(self.test_dir, f'{self.env_name}_map_{i}_{seed}.svg')
            fig.savefig(save_path, dpi=120)


def plot_anomalies(anomalies):
    fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(12, 6))
    xs = np.arange(len(anomalies))
    ax1.plot(xs, anomalies)
    ax2.hist(anomalies)
    plt.show()

