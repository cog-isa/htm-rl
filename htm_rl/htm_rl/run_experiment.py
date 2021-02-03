import argparse
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

from htm_rl.agent.agent import TransferLearningExperimentRunner, TransferLearningExperimentRunner2
from htm_rl.agent.train_eval import RunResultsProcessor
from htm_rl.config import (
    TestRunner, read_config, RandomSeedSetter,
)


class ExperimentRunner:
    config: Dict

    def __init__(self, config):
        self.config = config

    def run(self, agent_key: str, run: bool, aggregate: bool):
        silent_run = self.config['silent']
        random_seeder: RandomSeedSetter = self.config['random_seed_setter']
        report_name_suffix = self.config['report_name']

        run_results_processor: RunResultsProcessor = self.config['run_results_processor']
        if run_results_processor.test_dir is None:
            run_results_processor.test_dir = self.config['test_dir']

        transfer_learning_experiment_runner: TransferLearningExperimentRunner
        transfer_learning_experiment_runner = self.config.get('transfer_learning_experiment_runner', None)

        dry_run = self.config['dry_run']
        if transfer_learning_experiment_runner is not None and dry_run:
            transfer_learning_experiment_runner: TransferLearningExperimentRunner2
            n_envs_to_render = self.config['n_environments_to_render']
            maps = transfer_learning_experiment_runner.get_environment_maps(n_envs_to_render)
            run_results_processor.store_environment_maps(maps)
            return

        # by default, if nothing specified then at least `run`
        if not dry_run and (run or not aggregate):
            if agent_key is not None:
                agents = [agent_key] if not isinstance(agent_key, list) else agent_key
            else:
                agents = self.config['agent_runners'].keys()

            for agent in agents:
                random_seeder.reset()
                # TODO: make IAgentRunner
                runner = self.config['agent_runners'][agent]
                runner.name = agent
                print(f'AGENT: {agent}')
                if transfer_learning_experiment_runner is not None:
                    transfer_learning_experiment_runner.run_experiment(runner)
                else:
                    runner.run()
                runner.store_results(run_results_processor)

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