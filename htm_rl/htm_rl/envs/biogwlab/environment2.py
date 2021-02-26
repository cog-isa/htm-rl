from typing import Dict, List

import numpy as np
from numpy.random._generator import Generator

from htm_rl.common.sdr_encoders import IntBucketEncoder
from htm_rl.common.utils import isnone
from htm_rl.envs.biogwlab.dynamics2 import BioGwLabEnvDynamics
from htm_rl.envs.biogwlab.environment_state import EnvironmentState
from htm_rl.envs.biogwlab.generation.env_generator import EnvironmentGenerator


class BioGwLabEnvironment:
    supported_actions = {'stay', 'move left', 'move up', 'move right', 'move down'}

    state: EnvironmentState

    _init_state: EnvironmentState
    _actions: List[str]
    _episode_max_steps: int
    _rnd: Generator
    _generator: EnvironmentGenerator
    _dynamics: BioGwLabEnvDynamics

    _episode_step: int

    def __init__(
            self, generator: Dict,
            episode_max_steps: int, seed: int,
            state_encoder: Dict,
            actions: List[str] = None
    ):
        if actions is not None:
            self.ensure_all_actions_supported(actions)
        else:
            actions = list(self.supported_actions)

        self._actions = actions
        self._episode_max_steps = episode_max_steps
        self._rnd = np.random.default_rng(seed=seed)
        self._generator = EnvironmentGenerator(seed=seed, **generator)
        self._dynamics = BioGwLabEnvDynamics()
        self._episode_step = 0

        self._init_state = self._generator.generate()
        self.state = self._init_state.make_copy()
        self._state_encoder = IntBucketEncoder(n_values=self.state.n_cells, **state_encoder)

    @property
    def n_actions(self):
        return len(self._actions)

    @property
    def output_sdr_size(self):
        return self._state_encoder.output_sdr_size

    def reset(self):
        """ reset the game, return the initial state"""
        self._episode_step = 0
        self.state = self._init_state.make_copy()
        return self._get_agent_state()

    def is_terminal(self, state=None):
        """ return True if position is terminal or False if it isn't """
        state = isnone(state, self.state)
        return self._episode_step >= self._episode_max_steps or self._dynamics.is_terminal(state)

    def step(self, action):
        """ take action, return next_state, reward, is_done, empty_info """
        assert not self.is_terminal(), 'Episode is finished!'

        state = self.state
        action = self._actions[action]
        reward = 0
        if action == 'stay':
            reward += self._dynamics.stay(state)
        else:
            direction = action[5:]  # cut "move "

            reward += self._dynamics.move(state, direction)
            reward += self._dynamics.stay(state)

        agent_state = self._get_agent_state()
        self._episode_step += 1
        is_done = self.is_terminal()
        return agent_state, reward, is_done, {}

    def _get_agent_state(self):
        position = self.state.agent_position
        position_fl = position[0] * self.state.shape[1] + position[1]
        encoded_position = self._state_encoder.encode(position_fl)
        return encoded_position

    @classmethod
    def ensure_all_actions_supported(cls, actions):
        non_supported_actions = [
            action for action in actions
            if action not in cls.supported_actions
        ]
        assert not non_supported_actions, \
            f'{non_supported_actions} actions are not supported'