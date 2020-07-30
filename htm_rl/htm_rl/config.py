import dataclasses
import random
from abc import abstractmethod
from dataclasses import dataclass
from inspect import getattr_static
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List

import numpy as np
from ruamel.yaml import YAML, BaseLoader, SafeConstructor

from htm_rl.agent.agent import Agent, AgentRunner
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
from htm_rl.common.utils import trace
from htm_rl.envs.mdp import SarSuperpositionFormatter, GridworldMdpGenerator, Mdp, SaSuperpositionFormatter
from htm_rl.htm_plugins.temporal_memory import TemporalMemory
from htm_rl.envs.testing_envs import PresetMdpCellTransitions


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

        random.seed(seed)
        np.random.seed(seed)


@dataclass
class MdpCellTransitionsGeneratorConfig(BaseConfig):
    preset_name: str
    path_directions: List[int]

    @classmethod
    def path(cls):
        return '.env_mdp_cell_transitions'

    @classmethod
    def apply_defaults(cls, config, global_config):
        config['preset_name'] = 'passage'
        config['path_directions'] = [0, 1]

    def make(self, verbose=False):
        preset_name = self.preset_name
        if preset_name == 'passage':
            return PresetMdpCellTransitions.passage(self.path_directions)
        elif preset_name == 'multi_way_v0':
            return PresetMdpCellTransitions.multi_way_v0()
        elif preset_name == 'multi_way_v1':
            return PresetMdpCellTransitions.multi_way_v1()
        elif preset_name == 'multi_way_v2':
            return PresetMdpCellTransitions.multi_way_v2()
        elif preset_name == 'multi_way_v3':
            return PresetMdpCellTransitions.multi_way_v3()


@dataclass
class TemporalMemoryConfig(BaseConfig):
    n_columns: int
    cells_per_column: int
    activation_threshold: int
    learning_threshold: int
    maxNewSynapseCount: int
    maxSynapsesPerSegment: int
    seed: int
    initial_permanence: float = .5
    connected_permanence: float = .4
    predictedSegmentDecrement: float = .0001
    permanenceIncrement: float = .1
    permanenceDecrement: float = .05

    @classmethod
    def path(cls):
        return '.agent_tm'

    @classmethod
    def apply_defaults(cls, config, global_config):
        if 'agent_encoder' in global_config:
            encoder: SarSdrEncoder = global_config['agent_encoder']

            activation_threshold = encoder.activation_threshold

            action_bits = encoder._encoders.action.value_bits
            reward_bits = encoder._encoders.reward.value_bits
            learning_threshold = action_bits + reward_bits + 2
            assert learning_threshold + 2 < encoder.activation_threshold, \
                f'{learning_threshold}, {encoder.activation_threshold}'

            config['n_columns'] = encoder.total_bits
            config['activation_threshold'] = activation_threshold
            config['learning_threshold'] = learning_threshold
            config['maxNewSynapseCount'] = encoder.value_bits
            config['maxSynapsesPerSegment'] = encoder.value_bits

        if 'seed' in global_config:
            config['seed'] = global_config['seed']

    def make(self, verbosity=False) -> TemporalMemory:
        trace(
            verbosity, 1,
            f'Cells: {self.n_columns}x{self.cells_per_column}; '
            f'activation: {self.activation_threshold}; '
            f'learn: {self.learning_threshold}'
        )
        tm_kwargs = dataclasses.asdict(self)
        return TemporalMemory(**tm_kwargs)


@dataclass
class AgentRunnerConfig(BaseConfig):
    env: Any
    agent: Agent
    n_episodes: int
    max_steps: int
    verbose: bool
    pretrain: int = 0

    @classmethod
    def path(cls):
        return '.agent_runner'

    @classmethod
    def apply_defaults(cls, config, global_config):
        config['env'] = global_config['env']
        config['agent'] = global_config['agent']
        config['verbose'] = global_config['verbosity'] > 1

    def make(self, verbose=False):
        return AgentRunner(
            self.agent, self.env, self.n_episodes, self.max_steps, self.pretrain, self.verbose
        )


@dataclass
class DqnAgentRunnerConfig(BaseConfig):
    env: Any
    agent: DqnAgent
    n_episodes: int
    max_steps: int
    verbose: bool

    @classmethod
    def path(cls):
        return '.dqn_agent_runner'

    @classmethod
    def apply_defaults(cls, config, global_config):
        config['env'] = global_config['env']
        config['agent'] = global_config['dqn_agent']
        config['verbose'] = global_config['verbosity'] > 1

    def make(self, verbose=False):
        kwargs = dataclasses.asdict(self)
        return DqnAgentRunner(**kwargs)


class TestRunner:
    config: Dict

    def __init__(self, config):
        self.config = config

    def run(self, agent_key: str, run: bool, aggregate: bool):
        run_results_processor: RunResultsProcessor = self.config['run_results_processor']
        if run_results_processor.test_dir is None:
            run_results_processor.test_dir = self.config['test_dir']

        # by default, if nothing specified then at least `run`
        if run or not aggregate:
            if agent_key is not None:
                agents = [agent_key]
            else:
                agents = self.config['agents'].keys()

            for agent in agents:
                # TODO: make IAgentRunner
                runner = self.config['agent_runners'][agent]
                print(f'AGENT: {agent}')
                runner.run()
                runner.store_results(run_results_processor)

        if aggregate:
            run_results_processor.aggregate_results()


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
