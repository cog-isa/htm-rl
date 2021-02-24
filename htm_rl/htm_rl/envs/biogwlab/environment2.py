from typing import Tuple, Dict, List

import numpy as np
from numpy.random._generator import Generator

from htm_rl.common.utils import isnone
from htm_rl.envs.biogwlab.dynamics import BioGwLabEnvDynamics
from htm_rl.envs.biogwlab.environment_state import BioGwLabEnvState
from htm_rl.envs.biogwlab.generation.map_generator import BioGwLabEnvGenerator


class BioGwLabEnvironment:
    supported_actions = {'stay', 'move left', 'move up', 'move right', 'move down'}

    state: BioGwLabEnvState

    _actions: List[str]
    _episode_max_steps: int
    _rnd: Generator
    _generator: BioGwLabEnvGenerator
    _dynamics: BioGwLabEnvDynamics

    _episode_step: int

    def __init__(
            self, generator: Dict, actions: List[str],
            episode_max_steps: int, seed: int
    ):
        self.ensure_all_actions_supported(actions)

        self._actions = actions
        self._episode_max_steps = episode_max_steps
        self._rnd = np.random.default_rng(seed=seed)
        self._generator = BioGwLabEnvGenerator(seed=seed, **generator)
        self._dynamics = BioGwLabEnvDynamics()
        self._episode_step = 0

        self.state = self._generator.generate()
        self.generate_initial_position()

    def generate_new_environment(self):
        self.state = self._generator.generate()

    def generate_initial_position(self):
        position = self._get_random_empty_position()
        direction = self._rnd.integers(len(self.state.directions))

        self.state.agent_initial_position = position
        self.state.agent_initial_direction = direction

    def reset(self):
        """ reset the game, return the initial state"""
        self._episode_step = 0
        self._reset_food()

        self.state.agent_position = self.state.agent_initial_position
        self.state.agent_direction = self.state.agent_initial_direction
        return self._get_agent_state()

    def is_terminal(self, state=None):
        """ return True if position is terminal or False if it isn't """
        state = isnone(state, self.state)
        return self._episode_step >= self._episode_max_steps or self._dynamics.is_terminal(state)

    def step(self, action):
        """ take action, return next_state, reward, is_done, empty_info """
        assert not self.is_terminal(), 'Episode is finished!'

        state = self.state
        action = self.actions[action]
        reward = 0
        if action == 'stay':
            reward += self._dynamics.stay(state)
        elif action == 'move':
            reward += self._dynamics.move_forward(state)
            reward += self._dynamics.stay(state)
        elif action == 'turn left':
            reward += self._dynamics.turn(-1, state)
            reward += self._dynamics.stay(state)
        elif action == 'turn right':
            reward += self._dynamics.turn(1, state)
            reward += self._dynamics.stay(state)
        else:
            raise NotImplementedError(f'Action {action} is not implemented')

        agent_state = self._get_agent_state()
        self._episode_step += 1
        is_done = self.is_terminal()
        return agent_state, reward, is_done, {}

    def _reset_food(self):
        # print('<=', self.state.n_rewarding_foods)
        self.state.n_rewarding_foods = self._dynamics.count_rewarding_food_items(self.state)
        # print('=>', self.state.n_rewarding_foods)
        for i, j, food_type in self.state.food_items:
            self.state.food_mask[i, j] = True
            self.state.food_map[i, j] = food_type

    def _get_agent_state(self):
        position = self.state.agent_position
        direction = self.state.directions[self.state.agent_direction]
        return position, direction

    def _unflatten(self, flatten_position):
        return divmod(flatten_position, self.state.size)

    def _get_random_empty_position(self):
        non_empty_mask = self.state.obstacle_mask | self.state.food_mask
        available_states = np.flatnonzero(non_empty_mask)
        position = self._rnd.choice(available_states)
        return self._unflatten(position)

    @classmethod
    def ensure_all_actions_supported(cls, actions):
        non_supported_actions = [
            action for action in actions
            if action not in cls.supported_actions
        ]
        assert not non_supported_actions, \
            f'{non_supported_actions} actions are not supported'