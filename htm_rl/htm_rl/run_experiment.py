import argparse
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

from htm_rl.agent.train_eval import RunResultsProcessor
from htm_rl.agents.ucb.ucb_experiment_runner import UcbExperimentRunner
from htm_rl.config import read_config


class ExperimentRunner:
    config: Dict

    def __init__(self, config):
        self.config = config

    def run(self, agent_key: str, run: bool, aggregate: bool):
        silent_run = self.config['silent']
        report_name_suffix = self.config['report_name']

        run_results_processor: RunResultsProcessor = self.config['run_results_processor']
        if run_results_processor.test_dir is None:
            run_results_processor.test_dir = self.config['test_dir']

        experiment_runner = self.config.get('experiment')

        dry_run = self.config['dry_run']
        if experiment_runner is not None and dry_run:
            n_envs_to_render = self.config['n_environments_to_render']
            maps = experiment_runner.get_environment_maps(n_envs_to_render)
            run_results_processor.store_environment_maps(maps)
            return

        # by default, if nothing specified then at least `run`
        if not dry_run and (run or not aggregate):
            experiment_runner: UcbExperimentRunner

            print(f'AGENT: {experiment_runner.name}')
            env_config = self.config['environment']
            agent_config = self.config['agent']
            experiment_runner.run_experiment(env_config, agent_config)
            experiment_runner.store_results(run_results_processor)

        if aggregate:
            aggregate_file_masks = self.config['aggregate_masks']
            for_paper = self.config['paper']
            run_results_processor.aggregate_results(
                aggregate_file_masks, report_name_suffix, silent_run, for_paper
            )


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

    runner = ExperimentRunner(config)
    runner.run(args.agent, args.run, aggregate)


if __name__ == '__main__':
    main()