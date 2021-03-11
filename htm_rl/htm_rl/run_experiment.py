import argparse
import os
from argparse import ArgumentParser
from itertools import product
from pathlib import Path
from typing import Dict

import numpy as np

from htm_rl.agent.train_eval import RunResultsProcessor
from htm_rl.config import read_config
from htm_rl.experiment import Experiment


class RunConfig:
    config: Dict

    def __init__(self, config):
        self.config = config

    def run(self, agent_key: str, run: bool, aggregate: bool):
        silent_run = self.config['silent']
        report_name_suffix = self.config['report_name']

        run_results_processor: RunResultsProcessor = self.config['run_results_processor']
        if run_results_processor.test_dir is None:
            run_results_processor.test_dir = self.config['test_dir']

        experiment = Experiment(**self.config['experiment'])

        dry_run = self.config['dry_run']
        if experiment is not None and dry_run:
            n_envs_to_render = self.config['n_environments_to_render']
            maps = experiment.get_environment_maps(n_envs_to_render)
            run_results_processor.store_environment_maps(maps)
            return

        # by default, if nothing specified then at least `run`
        if not dry_run and (run or not aggregate):
            experiment: Experiment

            seeds = self.get_seeds()
            env_configs = self.read_configs('env')
            agent_configs = self.read_configs('agent')

            experiment_setups = list(product(env_configs, agent_configs, seeds))
            for env_config, agent_config, seed in experiment_setups:
                results = experiment.run(
                    seed=seed, agent_config=agent_config, env_config=env_config
                )
                run_results_processor.store_result(results)
                print('')

        if aggregate:
            aggregate_file_masks = self.config['aggregate_masks']
            for_paper = self.config['paper']
            run_results_processor.aggregate_results(
                aggregate_file_masks, report_name_suffix, silent_run, for_paper
            )

    def get_seeds(self):
        seeds = self.config['seed']
        if isinstance(seeds, list):
            ...
        elif 'n_seeds' in self.config:
            seed = seeds
            n_seeds = self.config['n_seeds']
            rng = np.random.default_rng(seed)
            seeds = rng.integers(1_000_000, size=n_seeds).tolist()
        else:
            seeds = [seeds]
        return seeds

    def read_configs(self, key):
        path: Path = self.config['config_path']
        names = self.config[key]

        if not isinstance(names, list):
            names = [names]

        configs = []
        for name in names:
            config_path = path.with_name(f'{key}_{name}.yml')
            config = read_config(config_path, verbose=False)
            configs.append(config)

        return configs


def register_arguments(parser: ArgumentParser):
    # todo: comment arguments with examples
    parser.add_argument('-c', '--config', dest='config', required=True)
    parser.add_argument('-a', '--agent', dest='agent', default=None, nargs='+')
    parser.add_argument('-r', '--run', dest='run', action='store_true', default=False)
    parser.add_argument('-g', '--aggregate', dest='aggregate', default=None, nargs='*')
    parser.add_argument('-n', '--name', dest='report_name', default='')
    parser.add_argument('-d', '--dry', dest='dry', default=None, type=int)
    parser.add_argument('-s', '--silent', dest='silent', action='store_true', default=False)
    parser.add_argument('-p', '--paper', dest='paper', action='store_true', default=False)


def main():
    parser = argparse.ArgumentParser()
    register_arguments(parser)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = read_config(config_path, verbose=False)
    config['config_path'] = config_path

    test_dir = os.path.dirname(config_path)
    config['test_dir'] = test_dir

    config['silent'] = args.silent
    config['paper'] = args.paper

    dry_run = args.dry is not None
    config['dry_run'] = dry_run
    if dry_run:
        config['n_environments_to_render'] = args.dry
    config['report_name'] = args.report_name

    aggregate = args.aggregate is not None
    if aggregate:
        config['aggregate_masks'] = args.aggregate

    runner = RunConfig(config)
    runner.run(args.agent, args.run, aggregate)


if __name__ == '__main__':
    main()