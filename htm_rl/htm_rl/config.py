import dataclasses
import random
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Dict, List, Type, Optional
from inspect import getattr_static

import numpy as np
from ruamel.yaml import YAML, BaseConstructor, BaseLoader, Constructor, SafeConstructor

from htm_rl.agent.agent import Agent, AgentRunner
from htm_rl.agent.memory import Memory
from htm_rl.agent.planner import Planner
from htm_rl.agent.train_eval import RunResultsProcessor
from htm_rl.baselines.dqn_agent import DqnAgent, DqnAgentRunner
from htm_rl.common.base_sar import SarRelatedComposition
from htm_rl.common.int_sdr_encoder import IntSdrEncoder
from htm_rl.common.sar_sdr_encoder import SarSdrEncoder
from htm_rl.common.utils import trace, project_to_type_fields, isnone
from htm_rl.envs.mdp import SarSuperpositionFormatter, GridworldMdpGenerator, Mdp
from htm_rl.htm_plugins.temporal_memory import TemporalMemory
from htm_rl.testing_envs import PresetMdpCellTransitions


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
class SarSdrEncoderConfig(BaseConfig):
    n_unique_states: int
    n_unique_actions: int
    n_unique_rewards: int = 2
    bits_per_state_value: int = 8
    bits_per_action_value: int = 8
    bits_per_reward_value: int = 8
    trace_format: str = 'short'

    @classmethod
    def path(cls):
        return '.agent_encoder'

    @classmethod
    def apply_defaults(cls, config, global_config):
        if 'env' not in global_config:
            return

        env = global_config['env']
        config['n_unique_states'] = env.n_states
        config['n_unique_actions'] = env.n_actions

    def make(self, verbose=False) -> SarSdrEncoder:
        # shortcuts
        bps, bpa, bpr = self.bits_per_state_value, self.bits_per_action_value, self.bits_per_reward_value
        n_states, n_actions, n_rewards = self.n_unique_states, self.n_unique_actions, self.n_unique_rewards
        trace_format = self.trace_format

        state_encoder = IntSdrEncoder('state', n_states, bps, bps - 1, trace_format)
        action_encoder = IntSdrEncoder('action', n_actions, bpa, bpa - 1, trace_format)
        reward_encoder = IntSdrEncoder('reward', n_rewards, bpr, bpr - 1, trace_format)

        return SarSdrEncoder(state_encoder, action_encoder, reward_encoder)


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

    def make(self, verbose=False) -> TemporalMemory:
        trace(
            verbose,
            f'Cells: {self.n_columns}x{self.cells_per_column}; '
            f'activation: {self.activation_threshold}; '
            f'learn: {self.learning_threshold}'
        )
        tm_kwargs = dataclasses.asdict(self)
        return TemporalMemory(**tm_kwargs)


@dataclass
class AgentConfig(BaseConfig):
    n_actions: int
    planning_horizon: int
    encoder: SarSdrEncoder
    tm: TemporalMemory
    sdr_formatter: Callable
    sar_formatter: Callable = SarSuperpositionFormatter.format

    collect_anomalies: bool = False
    use_cooldown: bool = False

    @classmethod
    def path(cls):
        return '.agent'

    @classmethod
    def apply_defaults(cls, config, global_config):
        env = global_config['env']
        config['n_actions'] = env.n_actions

        if 'agent_encoder' in global_config:
            encoder = global_config['agent_encoder']
            config['encoder'] = encoder
            config['sdr_formatter'] = encoder.format

        if 'agent_tm' in global_config:
            config['tm'] = global_config['agent_tm']

    def make(self, verbose=False):
        memory = Memory(
            self.tm, self.encoder, self.sdr_formatter, self.sar_formatter,
            collect_anomalies=self.collect_anomalies
        )
        planner = Planner(memory, self.planning_horizon)
        agent = Agent(memory, planner, self.n_actions, use_cooldown=self.use_cooldown)
        return agent


@dataclass
class DqnAgentConfig(BaseConfig):
    n_states: int
    n_actions: int
    seed: int
    epsilon: float = .15
    gamma: float = .975
    lr: float = .3e-3

    @classmethod
    def path(cls):
        return '.dqn_agent'

    @classmethod
    def apply_defaults(cls, config, global_config):
        if 'env' in global_config:
            env = global_config['env']
            config['n_states'] = env.n_states
            config['n_actions'] = env.n_actions
        if 'seed' in global_config:
            config['seed'] = global_config['seed']

    def make(self, verbose=False):
        kwargs = dataclasses.asdict(self)
        return DqnAgent(**kwargs)


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


@dataclass
class RunResultsProcessorConfig(BaseConfig):
    env_name: str
    optimal_len: int
    test_dir: str
    moving_average: int = 4
    verbose: bool = True

    @classmethod
    def path(cls):
        return '.run_results_processor'

    @classmethod
    def apply_defaults(cls, config, global_config):
        if 'test_dir' in global_config:
            config['test_dir'] = global_config['test_dir']

    def make(self, verbose=False):
        return RunResultsProcessor(
            self.env_name, self.optimal_len, self.test_dir, self.moving_average, self.verbose
        )


class ConfigBasedFactory:
    config: Dict[str, Any]

    def __init__(self, config: Dict):
        self.config = config
        self.verbose = config.get('verbosity', 0) > 0

    def make_(self, *config_type):
        return [
            self.make_obj_(config_type) for config_type in config_type
        ]

    def make_obj_(self, config_type: Type[BaseConfig]):
        config_path = config_type.path()
        config = self[config_path]

        try:
            custom_config = project_to_type_fields(config_type, config)
            config_type.apply_defaults(config, self.config)
            config.update(custom_config)
            config_obj = config_type(**config)

            obj = config_obj.make(self.verbose)
            name = self._get_name(config_path)
            self.config[name] = obj
            return obj
        except:
            print('\n\n_________________ DEBUG:')
            print(f'TYPE: {config_type.__name__}')
            print('CONFIG[TYPE]:')
            pprint(config)
            print('GLOBAL CONFIG:')
            pprint(self.config)
            print('_______________________')
            raise


    def __getitem__(self, path):
        nodes = path.split('.')[1:]
        config = self.config
        for node in nodes:
            config = config.setdefault(f'.{node}', dict())
        return config

    def _get_name(self, path):
        return path.split('.')[-1]


class TestRunner:
    config: Dict

    def __init__(self, config):
        self.config = config
        self.factory = ConfigBasedFactory(config)

    def run(self, which_agent=None):
        # make env
        self.factory.make_(
            MdpCellTransitionsGeneratorConfig,
            MdpEnvConfig,
        )

        run_results_processor: Optional[RunResultsProcessor] = None
        if '.run_results_processor' in self.config:
            self.factory.make_(RunResultsProcessorConfig)
            run_results_processor = self.config['run_results_processor']

        if '.agent_runner' in self.config and isnone(which_agent, 'htm') == 'htm':
            self._run_htm_agent(run_results_processor)

        if '.dqn_agent_runner' in self.config and isnone(which_agent, 'dqn') == 'dqn':
            self._run_dqn_agent(run_results_processor)

    def aggregate_results(self):
        self.factory.make_(RunResultsProcessorConfig)
        run_results_processor: RunResultsProcessor = self.config['run_results_processor']
        run_results_processor.aggregate_results()

    def _run_htm_agent(self, run_results_processor):
        self.factory.make_(
            SarSdrEncoderConfig,
            TemporalMemoryConfig,
            AgentConfig,
            AgentRunnerConfig,
        )
        agent_runner: AgentRunner = self.config['agent_runner']
        agent_runner.run()

        agent: Agent = self.config['agent']
        planning_horizon = agent.planner.planning_horizon

        if run_results_processor is not None:
            run_results_processor.store_result(agent_runner.train_stats, f'htm_{planning_horizon}')

    def _run_dqn_agent(self, run_results_processor):
        self.factory.make_(
            DqnAgentConfig,
            DqnAgentRunnerConfig,
        )
        agent_runner: DqnAgentRunner = self.config['dqn_agent_runner']
        agent_runner.run()

        if run_results_processor is not None:
            run_results_processor.store_result(agent_runner.train_stats, f'dqn_eps')
            run_results_processor.store_result(agent_runner.test_stats, f'dqn_greedy')


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
        IntSdrEncoder, SarSdrEncoder,
        TemporalMemory, Memory,
        Planner,
        Agent, AgentRunner,
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
