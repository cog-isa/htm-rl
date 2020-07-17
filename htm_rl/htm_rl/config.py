import dataclasses
import inspect
import random
from dataclasses import dataclass
from itertools import chain
from typing import Any, Callable, Dict

import numpy as np

from htm_rl.agent.agent import Agent
from htm_rl.agent.memory import Memory
from htm_rl.agent.planner import Planner
from htm_rl.common.base_sar import SarRelatedComposition
from htm_rl.common.int_sdr_encoder import IntSdrEncoder
from htm_rl.common.sar_sdr_encoder import SarSdrEncoder
from htm_rl.common.utils import trace
from htm_rl.envs.mdp import SarSuperpositionFormatter
from htm_rl.htm_plugins.temporal_memory import TemporalMemory


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class SarSdrEncoderConfig:
    n_unique_states: int
    n_unique_actions: int
    n_unique_rewards: int = 2
    bits_per_state_value: int = 8
    bits_per_action_value: int = 8
    bits_per_reward_value: int = 8
    trace_format: str = 'short'

    @classmethod
    def path(cls):
        return '.encoder'

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

        encoders = SarRelatedComposition(state_encoder, action_encoder, reward_encoder)
        return SarSdrEncoder(encoders)


@dataclass
class TemporalMemoryConfig:
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
        return '.tm'

    @classmethod
    def apply_defaults(cls, config, global_config):
        if 'encoder' in global_config:
            encoder: SarSdrEncoder = global_config['encoder']

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
class AgentConfig:
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

        if 'encoder' in global_config:
            encoder = global_config['encoder']
            config['encoder'] = encoder
            config['sdr_formatter'] = encoder.format

        if 'tm' in global_config:
            config['tm'] = global_config['tm']

    def make(self, verbose=False):
        memory = Memory(
            self.tm, self.encoder, self.sdr_formatter, self.sar_formatter,
            collect_anomalies=self.collect_anomalies
        )
        planner = Planner(memory, self.planning_horizon)
        agent = Agent(memory, planner, self.n_actions, use_cooldown=self.use_cooldown)
        return agent


class ConfigBasedFactory:
    config: Dict[str, Any]

    def __init__(self, config: Dict):
        self.config = config
        self.verbose = config.get('verbosity', 0) > 0

    def make(self, *config_type):
        return [
            self.make_obj(config_type) for config_type in config_type
        ]

    def make_obj(self, config_type):
        config_path = config_type.path()
        config = self[config_path]

        custom_config = self._project_to_type_fields(config_type, config)
        config_type.apply_defaults(config, self.config)
        config.update(custom_config)
        config_obj = config_type(**config)

        obj = config_obj.make(self.verbose)
        name = self._get_name(config_path)
        self.config[name] = obj
        return obj

    @staticmethod
    def _project_to_type_fields(config_type, config):
        projection = {
            field.name: config[field.name]
            for field in dataclasses.fields(config_type)
            if field.name in config
        }
        return projection

    @staticmethod
    def _project_to_method_params(func, config):
        argspec = inspect.getfullargspec(func)
        args = chain(argspec.args, argspec.kwonlyargs)

        projection = {
            arg_name: config[arg_name]
            for arg_name in args
            if arg_name in config
        }
        return projection

    def __getitem__(self, path):
        nodes = path.split('.')[1:]
        config = self.config
        for node in nodes:
            config = config.setdefault(f'.{node}', dict())
        return config

    def _get_name(self, path):
        return path.split('.')[-1]
