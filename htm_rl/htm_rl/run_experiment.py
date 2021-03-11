import argparse
import os
from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import numpy as np

from htm_rl.agent.train_eval import RunResultsProcessor
from htm_rl.config import FileConfig
from htm_rl.experiment import Experiment


class RunConfig(FileConfig):
    def __init__(self, path):
        super(RunConfig, self).__init__(path)

    def run(self, agent_key: str, run: bool, aggregate: bool):
        silent_run = self['silent']
        report_name_suffix = self['report_name']

        run_results_processor: RunResultsProcessor = self['run_results_processor']
        if run_results_processor.test_dir is None:
            run_results_processor.test_dir = self['test_dir']

        experiment = Experiment(
            base_dir=self.path.parent, use_wandb=self['wandb'],
            **self['experiment']
        )

        dry_run = self['dry_run']
        if experiment is not None and dry_run:
            # doesn't work at the moment
            n_envs_to_render = self['n_environments_to_render']
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
            aggregate_file_masks = self['aggregate_masks']
            for_paper = self['paper']
            run_results_processor.aggregate_results(
                aggregate_file_masks, report_name_suffix, silent_run, for_paper
            )

    def get_seeds(self):
        seeds = self['seed']
        if isinstance(seeds, list):
            ...
        elif 'n_seeds' in self:
            seed = seeds
            n_seeds = self['n_seeds']
            rng = np.random.default_rng(seed)
            seeds = rng.integers(1_000_000, size=n_seeds).tolist()
        else:
            seeds = [seeds]
        return seeds

    def read_configs(self, key):
        names = self[key]

        if not isinstance(names, list):
            names = [names]

        configs = []
        for name in names:
            config_path = self.path.with_name(f'{key}_{name}.yml')
            config = FileConfig(config_path, name=name)
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
    parser.add_argument('-w', '--wandb', dest='wandb', action='store_true', default=False)
    parser.add_argument('-W', '--wandb-dry-silent', dest='wandb_dry_silent', action='store_true', default=False)


def main():
    parser = argparse.ArgumentParser()
    register_arguments(parser)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = RunConfig(config_path)

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

    config['wandb'] = args.wandb
    if args.wandb_dry_silent:
        os.environ['WANDB_MODE'] = 'dryrun'
        os.environ['WANDB_SILENT'] = 'true'

    config.run(args.agent, args.run, aggregate)


if __name__ == '__main__':
    main()