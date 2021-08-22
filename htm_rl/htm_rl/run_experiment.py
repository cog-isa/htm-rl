import argparse
import os
from argparse import ArgumentParser
from ast import literal_eval
from itertools import product
from pathlib import Path

import numpy as np

from htm_rl.common.utils import ensure_list
from htm_rl.scenarios.config import FileConfig
from htm_rl.scenarios.standard.experiment import Experiment, RunStats


class RunConfig(FileConfig):
    def __init__(self, path):
        super(RunConfig, self).__init__(path)

    def run(self,):
        self.read_subconfigs('envs', prefix='env')
        self.read_subconfigs('agents', prefix='agent')
        for key, agent in self['agents'].items():
            agent.read_subconfig('sa_encoder', 'sa_encoder')
        self.add_overwrite_attributes(self['overwrites'])

        env_seeds = self['env_seeds']
        agent_seeds = self['agent_seeds']
        env_names = self.get_filtered_names_for('envs')
        agent_names = self.get_filtered_names_for('agents')

        experiment_setups = list(product(env_names, agent_names, env_seeds))
        for env_name, agent_name, env_seed in experiment_setups:
            self['env'] = env_name
            self['env_seed'] = env_seed
            self['agent'] = agent_name
            agent_results = []
            for agent_seed in agent_seeds:
                self['agent_seed'] = agent_seed
                print(f'AGENT: {agent_name}     SEED: {env_seed} {agent_seed}')

                results: RunStats = Experiment(self).run()
                results.print_results()
                agent_results.append(results)

            if len(agent_seeds) > 1:
                results = self.aggregate_stats(agent_results)
                results.print_results()
            print('')

    def add_overwrite_attributes(self, overwrites: list[str]):
        i = 0
        while i < len(overwrites):
            key = overwrites[i]
            if not key.startswith('--'):
                raise ValueError(
                    f'Config attribute name started with `--` is expected, got `{key}`!'
                )
            key = key[2:]
            value = overwrites[i+1]
            if value.startswith('--'):
                raise ValueError(
                    f'Config attribute value is expected, got `{value}`, which looks like an attribute name!'
                )
            value = self.parse_str(value)
            self[key] = value
            i += 2

    @staticmethod
    def parse_str(val):
        def boolify(s):
            if s in ['True', 'true']:
                return True
            if s in ['False', 'false']:
                return False
            raise ValueError('Not Boolean Value!')

        for caster in (boolify, int, float, literal_eval):
            try:
                return caster(val)
            except ValueError:
                pass
        return val

    def get_filtered_names_for(self, key):
        filter_key = f'{key}_filter'
        names = self[key].keys()
        names_filter = self[filter_key]

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
    parser.add_argument('-e', '--envs_filter', dest='envs_filter', default=None, nargs='+')
    parser.add_argument('-a', '--agents_filter', dest='agents_filter', default=None, nargs='+')
    parser.add_argument('-d', '--print_debug', dest='debug_enabled', action='store_true', default=False)
    parser.add_argument('-w', '--wandb_enabled', dest='wandb_enabled', action='store_true', default=False)


def main():
    parser = argparse.ArgumentParser()
    register_arguments(parser)
    args, overwrites = parser.parse_known_args()

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
    config['envs_filter'] = args.envs_filter
    config['agents_filter'] = args.agents_filter
    config['debug.enabled'] = args.debug_enabled
    config['wandb.enabled'] = args.wandb_enabled
    config['overwrites'] = overwrites
    config.run()


if __name__ == '__main__':
    main()
