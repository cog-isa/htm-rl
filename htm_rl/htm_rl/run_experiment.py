import argparse
import os
from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import numpy as np
import tqdm
from tqdm import trange

from htm_rl.common.utils import ensure_list
from htm_rl.config import FileConfig
from htm_rl.experiment import Experiment, RunStats


class RunConfig(FileConfig):
    def __init__(self, path):
        super(RunConfig, self).__init__(path)

    def run(self, agents_filter: list[str], envs_filter: list[str]):
        experiment = Experiment(**self)
        env_seeds = self['env_seeds']
        agent_seeds = self['agent_seeds']
        env_names = self.filter_by(self.read_subconfigs('envs', prefix='env'), envs_filter)
        agent_names = self.filter_by(self.read_subconfigs('agents', prefix='agent'), agents_filter)

        experiment_setups = list(product(env_names, agent_names, env_seeds))
        for env_name, agent_name, env_seed in experiment_setups:
            self['env'] = env_name
            self['env_seed'] = env_seed
            self['agent'] = agent_name
            agent_results = []
            for agent_seed in agent_seeds:
                self['agent_seed'] = agent_seed
                print(f'AGENT: {agent_name}     SEED: {env_seed} {agent_seed}')
                results: RunStats = experiment.run(self)
                results.print_results()
                agent_results.append(results)

            if len(agent_seeds) > 1:
                results = self.aggregate_stats(agent_results)
                results.print_results()
            print('')

    def read_subconfigs(self, key, prefix):
        if not isinstance(self[key], dict):
            # has to be the name/names
            self[key] = ensure_list(self[key])
            d = dict()
            for name in self[key]:
                d[name] = name
            self[key] = d

        configs = dict()
        for name, val in self[key].items():
            if isinstance(val, dict):
                continue
            config_path = self.path.with_name(f'{prefix}_{val}.yml')
            config = FileConfig(config_path, name=name)
            configs[name] = config

        self[key] = configs
        return self[key].keys()

    @staticmethod
    def filter_by(names, names_filter):
        if names_filter is None:
            return names

        names_filter = set(ensure_list(names_filter))
        return [
            name
            for name in names
            if name in names_filter
        ]

    @staticmethod
    def aggregate_stats(agent_results):
        results = RunStats()
        results.steps = np.mean([res.steps for res in agent_results], axis=0)
        results.times = np.mean([res.times for res in agent_results], axis=0)
        results.rewards = np.mean([res.rewards for res in agent_results], axis=0)
        return results


def register_arguments(parser: ArgumentParser):
    # todo: comment arguments with examples
    parser.add_argument('-c', '--config', dest='config', required=True)
    parser.add_argument('-a', '--agent', dest='agent', default=None, nargs='+')
    parser.add_argument('-e', '--env', dest='env', default=None, nargs='+')
    parser.add_argument('-m', '--print_maps', dest='print_maps', default=None, type=int)
    parser.add_argument('-t', '--print_heatmaps', dest='print_heatmaps', default=None, nargs='+', type=int)
    parser.add_argument('-d', '--print_debug', dest='print_debug', default=None, nargs='+', type=int)
    parser.add_argument('-w', '--wandb_enabled', dest='wandb_enabled', action='store_true', default=False)


def main():
    parser = argparse.ArgumentParser()
    register_arguments(parser)
    args = parser.parse_args()

    config_filename: str = args.config
    if not config_filename.startswith('expm_') or not config_filename.endswith('.yml'):
        config_filename = f'expm_{config_filename}.yml'
    config_path = Path(config_filename)
    config = RunConfig(config_path)

    base_dir: Path = config.path.parent
    results_dir: Path = base_dir.joinpath('results')
    results_dir.mkdir(exist_ok=True)

    config['base_dir'] = base_dir
    config['results_dir'] = results_dir

    config['print.maps'] = args.print_maps
    config['print.heatmaps'] = args.print_heatmaps
    config['print.debug'] = args.print_debug

    config['wandb.enabled'] = args.wandb_enabled
    if False:
        os.environ['WANDB_MODE'] = 'dryrun'
        # os.environ['WANDB_SILENT'] = 'true'

    config.run(args.agent, args.env)


if __name__ == '__main__':
    main()