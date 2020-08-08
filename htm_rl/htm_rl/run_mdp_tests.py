"""
Here you can test agent+planner on a simple handcrafted MDP envs.
"""
import argparse
import os

from htm_rl.config import (
    read_config, TestRunner,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config', default='./experiments/config.yml')
    parser.add_argument('-r', '--run', dest='run', action='store_true', default=False)
    parser.add_argument('-agg', '--aggregate', dest='aggregate', action='store_true', default=False)
    args = parser.parse_args()

    config = read_config(args.config)
    test_dir = os.path.dirname(args.config)
    config['test_dir'] = test_dir

    runner = TestRunner(config)
    if args.run or not args.aggregate:
        runner.run()

    if args.aggregate:
        runner.aggregate_results()


if __name__ == '__main__':
    main()