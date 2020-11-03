"""
Here you can test agent+planner on a simple handcrafted MDP envs.
"""
import argparse
import os
from pathlib import Path

from htm_rl.config import (
    TestRunner, read_config,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', required=True)
    parser.add_argument('-a', '--agent', dest='agent', default=None, nargs='+')
    parser.add_argument('-r', '--run', dest='run', action='store_true', default=False)
    parser.add_argument('-g', '--aggregate', dest='aggregate', default=None, nargs='*')
    parser.add_argument('-n', '--name', dest='report_name', default='')
    parser.add_argument('-d', '--dry', dest='dry', default=None, type=int)
    parser.add_argument('-s', '--silent', dest='silent', action='store_true', default=False)
    parser.add_argument('-p', '--paper', dest='paper', action='store_true', default=False)
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

    runner = TestRunner(config)
    runner.run(args.agent, args.run, aggregate)


if __name__ == '__main__':
    main()