import random
from abc import abstractmethod
from dataclasses import dataclass
from inspect import getattr_static
from pathlib import Path
from pprint import pprint
from typing import Dict, List

import numpy as np
from ruamel.yaml import YAML, BaseLoader, SafeConstructor

from htm_rl.agent.agent import Agent, AgentRunner, TransferLearningExperimentRunner, TransferLearningExperimentRunner2
from htm_rl.agent.legacy_agent import LegacyAgent
from htm_rl.agent.legacy_memory import LegacyMemory
from htm_rl.agent.legacy_planner import LegacyPlanner
from htm_rl.agent.memory import Memory
from htm_rl.agent.planner import Planner
from htm_rl.agent.train_eval import RunResultsProcessor
from htm_rl.baselines.dqn_agent import DqnAgent, DqnAgentRunner
from htm_rl.common.int_sdr_encoder import IntSdrEncoder
from htm_rl.common.sa_sdr_encoder import SaSdrEncoder
from htm_rl.common.sar_sdr_encoder import SarSdrEncoder
from htm_rl.envs.gridworld_map_generator import GridworldMapGenerator
from htm_rl.envs.mdp import (
    SarSuperpositionFormatter, PovBasedGridworldMdpGenerator, Mdp, SaSuperpositionFormatter,
    GridworldMdpGenerator,
)
from htm_rl.envs.preset_mdp_cell_transitions import PresetMdpCellTransitions
from htm_rl.htm_plugins.temporal_memory import TemporalMemory


@dataclass
class BaseConfig:

    @classmethod
    @abstractmethod
    def path(cls):
        ...

    @classmethod
    @abstractmethod
    def apply_defaults(cls, config, global_config):
        ...

    @abstractmethod
    def make(self, verbose=False):
        ...


class RandomSeedSetter:
    seed: int

    def __init__(self, seed):
        self.seed = seed
        self.reset()

    def reset(self):
        random.seed(self.seed)
        np.random.seed(self.seed)


class TestRunner:
    config: Dict

    def __init__(self, config):
        self.config = config

    def run(self, agent_key: str, run: bool, aggregate: bool):
        random_seeder: RandomSeedSetter = self.config['random_seed_setter']

        run_results_processor: RunResultsProcessor = self.config['run_results_processor']
        if run_results_processor.test_dir is None:
            run_results_processor.test_dir = self.config['test_dir']

        transfer_learning_experiment_runner: TransferLearningExperimentRunner
        transfer_learning_experiment_runner = self.config.get('transfer_learning_experiment_runner', None)

        dry_run = self.config['dry_run']
        if transfer_learning_experiment_runner is not None and dry_run:
            transfer_learning_experiment_runner: TransferLearningExperimentRunner2
            transfer_learning_experiment_runner.show_environments()

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
            report_name_suffix = self.config['report_name']
            run_results_processor.aggregate_results(aggregate_file_masks, report_name_suffix)


class PatchedSafeConstructor(SafeConstructor):
    def construct_yaml_object_with_init(self, node, cls):
        kwargs = self.construct_mapping(node, deep=True)
        return cls(**kwargs)


class PatchedYaml(YAML):
    def __init__(self):
        super(PatchedYaml, self).__init__(typ='safe')
        self.Constructor = PatchedSafeConstructor

    def register_class_for_initialization(self, cls):
        tag = f'!{cls.__name__}'

        def from_yaml(constructor: PatchedSafeConstructor, node):
            return constructor.construct_yaml_object_with_init(node, cls)

        self.constructor.add_constructor(tag, from_yaml)


class TagProxies:
    @staticmethod
    def passage_transitions(loader: BaseLoader, node):
        kwargs = loader.construct_mapping(node, deep=True)
        return PresetMdpCellTransitions.passage(**kwargs)

    @staticmethod
    def preset_env_transitions(loader: BaseLoader, node):
        preset_name = loader.construct_scalar(node)
        transitions_builder = getattr(PresetMdpCellTransitions, preset_name)
        return transitions_builder()

    @staticmethod
    def env(loader: BaseLoader, node):
        kwargs = loader.construct_mapping(node, deep=True)
        kwargs['env_type'] = Mdp
        return GridworldMdpGenerator.generate_env(**kwargs)

    @staticmethod
    def pov_env(loader: BaseLoader, node):
        kwargs = loader.construct_mapping(node, deep=True)
        kwargs['env_type'] = Mdp
        return PovBasedGridworldMdpGenerator.generate_env(**kwargs)

    @staticmethod
    def property(loader: BaseLoader, node):
        seq: List = loader.construct_sequence(node, deep=True)
        if len(seq) == 2:
            obj, prop_name = seq
            return getattr(obj, prop_name)
        else:
            obj, prop_name, default = seq
            return getattr(obj, prop_name, default)

    @staticmethod
    def sar_superposition_formatter(loader: BaseLoader, node):
        return SarSuperpositionFormatter.format

    @staticmethod
    def sa_superposition_formatter(loader: BaseLoader, node):
        return SaSuperpositionFormatter.format


def read_config(file_path: Path, verbose=False):
    yaml: PatchedYaml = PatchedYaml()
    register_static_methods_as_tags(TagProxies, yaml)
    register_classes(yaml)
    config = yaml.load(file_path)
    if verbose:
        pprint(config)
    return config


def register_classes(yaml: PatchedYaml):
    classes = [
        RandomSeedSetter,
        DqnAgent, DqnAgentRunner,
        IntSdrEncoder, SarSdrEncoder, SaSdrEncoder,
        TemporalMemory, LegacyMemory, Memory,
        LegacyPlanner, Planner,
        LegacyAgent, Agent, AgentRunner,
        TransferLearningExperimentRunner,
        GridworldMapGenerator,
        TransferLearningExperimentRunner2,
        RunResultsProcessor,
    ]
    for cls in classes:
        yaml.register_class_for_initialization(cls)


def register_static_methods_as_tags(cls, yaml: PatchedYaml):
    constructor: PatchedSafeConstructor = yaml.constructor
    constructor.deep_construct = True

    all_funcs = [
        (func, getattr_static(cls, func))
        for func in dir(cls)
    ]
    funcs = [
        (tag, getattr(cls, tag))
        for tag, func in all_funcs
        if isinstance(func, staticmethod)
    ]
    for tag, func in funcs:
        constructor.add_constructor(f'!{tag}', func)
