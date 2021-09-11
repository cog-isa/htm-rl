import argparse
import sys
from argparse import ArgumentParser
from pathlib import Path

from htm_rl.scenarios.config import FileConfig
from htm_rl.scenarios.factories import materialize_experiment


def register_arguments(parser: ArgumentParser):
    # todo: comment arguments with examples
    parser.add_argument('-c', '--config', dest='config', required=True)
    parser.add_argument('-e', '--envs_filter', dest='envs_filter', default=None, nargs='+')
    parser.add_argument('-a', '--agents_filter', dest='agents_filter', default=None, nargs='+')
    parser.add_argument('-d', '--print_debug', dest='debug_enabled', action='store_true', default=False)
    parser.add_argument('-w', '--wandb_enabled', dest='wandb_enabled', action='store_true', default=False)
    parser.add_argument('-o', '--output', dest='results_dir', default='results')


def main():
    parser = argparse.ArgumentParser()
    register_arguments(parser)
    args, overwrites = parser.parse_known_args()

    config_filename: str = args.config
    if not config_filename.startswith('expm_') or not config_filename.endswith('.yml'):
        config_filename = f'expm_{config_filename}.yml'
    config_path = Path(config_filename)
    config = FileConfig(config_path)
    experiment = materialize_experiment(config)

    base_dir: Path = config.path.parent
    results_dir: Path = base_dir.joinpath(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    autosave_dir: Path = base_dir.joinpath('autosaves')
    autosave_dir.mkdir(exist_ok=True)
    # Add auto cleaning old entries

    config['argv'] = sys.argv[1:]
    config['base_dir'] = base_dir
    config['results_dir'] = results_dir
    config['autosave_dir'] = autosave_dir
    config['envs_filter'] = args.envs_filter
    config['agents_filter'] = args.agents_filter
    config['debug.enabled'] = args.debug_enabled
    config['wandb.enabled'] = args.wandb_enabled
    config['overwrites'] = overwrites
    experiment.run()


if __name__ == '__main__':
    main()
