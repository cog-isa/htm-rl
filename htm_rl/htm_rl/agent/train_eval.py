import os
from glob import glob
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from htm_rl.common.utils import trace


class RunStats:
    name: str
    steps: List[int]
    rewards: List[float]
    times: List[float]

    def __init__(self, name: str):
        self.name = name
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

    def store_result(self, run_stats: RunStats):
        agent_info = run_stats.name
        result_table = pd.DataFrame({
            'steps': run_stats.steps,
            'times': run_stats.times,
            'rewards': run_stats.rewards
        })
        file_name = f'{self.env_name}_{agent_info}{self.data_ext}'
        out_file_path = os.path.join(self.test_dir, file_name)
        result_table.to_csv(out_file_path)

        self.print_results(run_stats)

    def aggregate_results(self, file_masks, report_suffix, silent: bool, for_paper: bool):
        if not for_paper:
            self._aggregate_results(file_masks, report_suffix, silent)
        else:
            self._aggregate_results_for_paper(file_masks, report_suffix, silent)

    def _aggregate_results_for_paper(self, file_masks, report_suffix, silent: bool):
        plt.rc('font', size=14)  # legend fontsize

        dfs = self._read_data_files(file_masks)
        col_names = self._get_names(dfs)
        df_steps: pd.DataFrame = pd.concat(self._get_columns(dfs, 'steps'), axis=1, keys=col_names)
        report_name = f'{self.env_name}'
        if report_suffix:
            report_name += f'__{report_suffix}'

        ma = self.moving_average
        self._plot_figure(
            df_steps, f'episode duration (steps), moving average: {ma}', report_name, 'steps',
            transparent=True
        )
        if not silent:
            plt.show()

    def _aggregate_results(self, file_masks, report_suffix, silent: bool):
        dfs = self._read_data_files(file_masks)
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
        # self._plot_figure(df_times, f'episode execution time, sec, MA={ma}', report_name, 'times')
        self._plot_figure(df_rewards, f'episode reward, MA={ma}', report_name, 'rewards')
        # self._plot_figure(df_steps, f'episode duration, steps, MA={ma}', report_name, 'steps')
        # self._plot_figure(
        #     df_steps_rel,
        #     f'episode duration in steps log-relative to `{rel_col}`, MA={ma}', report_name,
        #     f'steps_rel_{rel_col}'
        # )
        if not silent:
            plt.show()

    def _plot_figure(self, df: pd.DataFrame, title: str, report_name, ylabel, transparent=False):
            fig: plt.Figure
            ax: plt.Axes
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            df_ma = df.rolling(window=self.moving_average).mean()
            df_ma.plot(use_index=True, ax=ax)

            ax.legend(loc='right', bbox_to_anchor=(1.38, .8))
            ax.set_title(f'{report_name}: {title}')
            ax.set_xlabel('episode')
            ax.set_ylabel(ylabel)
            fig.tight_layout(h_pad=4.)

            ylabel = ylabel.replace(' ', '_')
            save_path = os.path.join(self.test_dir, f'{report_name}__{ylabel}.svg')
            fig.savefig(save_path, dpi=120, transparent=transparent)

    def _read_data_files(self, file_masks):
        if not file_masks:
            file_masks = ['*']

        csv_files_masks = [
            os.path.join(self.test_dir, f'{self.env_name}_{file_mask}{self.data_ext}')
            for file_mask in file_masks
        ]
        trace(self.verbosity, 2, '\n'.join(csv_files_masks))
        return [
            (file_path, pd.read_csv(file_path))
            for csv_file_mask in csv_files_masks
            for file_path in sorted(glob(csv_file_mask))
        ]

    def _get_names(self, dfs):
        env_name_len = len(self.env_name) + 1
        ext_len = len(self.data_ext)

        def path_to_short_name(file_path):
            p = Path(file_path)
            name = p.name[env_name_len:-ext_len]
            if name == 'htm_0':
                name = 'random'
            elif 'htm' in name:
                parts = name.split('_')
                if len(parts) == 2:
                    parts.append('16g')
                name, pl_hor, goals = parts
                goals = goals[:-1]
                name = f'horizon={pl_hor}, goals={goals}'
            elif 'dqn' in name:
                name, greedy = name.split('_')
                if greedy == 'eps':
                    greedy = f'$\epsilon$-greedy'
                name = f'dqn: {greedy}'
            return name

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
            self.store_environment_map(i, env_map, seed)

    def store_environment_map(self, ind, env_map, seed):
        h, w = env_map.shape

        fig: plt.Figure
        ax: plt.Axes
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.set_xticks(np.arange(-.5, w, 1))
        ax.set_yticks(np.arange(-.5, h, 1))
        ax.set_xticklabels(np.arange(w))
        ax.set_yticklabels(np.arange(h))
        ax.grid(color='grey', linestyle='-', linewidth=1)
        ax.set_title(f'{self.env_name}, seed={seed}')

        ax.imshow(env_map)
        save_path = os.path.join(self.test_dir, f'{self.env_name}_map_{ind}_{seed}.svg')
        fig.savefig(save_path, dpi=120)


def plot_anomalies(anomalies):
    fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(12, 6))
    xs = np.arange(len(anomalies))
    ax1.plot(xs, anomalies)
    ax2.hist(anomalies)
    plt.show()

