from typing import Tuple, List, Optional

import numpy as np

from htm_rl.common.utils import isnone, clip


class BioGwLabEnvState:
    directions = [(1, 0, 'right'), (0, -1, 'down'), (-1, 0, 'left'), (0, 1, 'up')]

    size: int
    seed: int
    n_area_types: int
    n_wall_colors: int
    n_food_types: int
    n_scent_channels: int
    obstacle_mask: np.ndarray
    areas_map: np.ndarray
    wall_colors: np.ndarray
    food_items: List[Tuple[int, int, int]]
    food_map: np.ndarray
    food_mask: np.ndarray
    food_scents: np.ndarray
    food_scent: np.ndarray

    agent_initial_position: Optional[Tuple[int, int]]
    agent_initial_direction: Optional[int]
    agent_position: Optional[Tuple[int, int]]
    agent_direction: Optional[int]

    def __init__(
            self, size: int, seed: int, n_area_types: int,
            n_wall_colors: int, n_food_types: int,
            n_scent_channels: int, obstacle_mask: np.ndarray,
            areas_map: np.ndarray, wall_colors: np.ndarray,
            food_items: List[Tuple[int, int, int]], food_mask: np.ndarray,
            food_map: np.ndarray, food_scents: np.ndarray
    ):
        self.size = size
        self.seed = seed
        self.n_area_types = n_area_types
        self.n_wall_colors = n_wall_colors
        self.n_food_types = n_food_types
        self.n_scent_channels = n_scent_channels
        self.obstacle_mask = obstacle_mask
        self.areas_map = areas_map
        self.wall_colors = wall_colors
        self.food_items = food_items
        self.food_map = food_map
        self.food_mask = food_mask
        self.food_scents = food_scents
        self.food_scent = food_scents.sum(axis=-1)

        self.agent_initial_position = None
        self.agent_initial_direction = None
        self.agent_position = None
        self.agent_direction = None


class BioGwLabDynamics:
    actions = {
        'stay': {
            'cost': .3
        },
        'move': {
            'cost': .7,
        },
        'turn': {
            'cost': .5,
        }
    }
    food_rewards = [1., -1., 5., -5.]
    time_cost = -.01

    def __init__(self):
        pass

    def is_terminal(self, state):
        return False

    def stay(self, state):
        i, j = state.agent_position
        reward = self.actions['stay']['cost'] * self.time_cost
        if state.food_map[i, j] >= 0:
            reward += self._eat_food(i, j, state)

        return reward

    def move_forward(self, state):
        i, j, success = self._move_forward(state)
        state.agent_position = (i, j)
        reward = self.actions['move']['cost'] * self.time_cost
        return reward

    def turn(self, turn_direction, state):
        state.agent_direction = self._turn(state.agent_direction, turn_direction)
        reward = self.actions['turn']['cost'] * self.time_cost
        return reward

    @staticmethod
    def _turn(view_direction, turn_direction):
        return (view_direction + turn_direction + 4) % 4

    @staticmethod
    def _move_forward(state: BioGwLabEnvState):
        i, j = state.agent_position
        view_direction = state.agent_direction
        directions = state.directions

        j += directions[view_direction][0]     # X axis
        i += directions[view_direction][1]     # Y axis

        i = clip(i, state.size)
        j = clip(j, state.size)
        success = (i, j) != state.agent_position and not state.obstacle_mask[i, j]
        if not success:
            i, j = state.agent_position
        return i, j, success

    def _eat_food(self, i, j, state):
        for k, (_i, _j, food_type) in enumerate(state.food_items):
            if i != _i or j != _j:
                continue

            state.food_scents[:, :, :, k] = 0
            state.food_mask[i, j] = False
            state.food_map[i, j] = -1
            return self.food_rewards[food_type]


class BioGwLabEnv:
    actions = ['stay', 'move', 'turn left', 'turn right']

    state: BioGwLabEnvState
    dynamics: BioGwLabDynamics

    def __init__(self, state: BioGwLabEnvState, dynamics: BioGwLabDynamics):
        self.state = state
        self.dynamics = dynamics
        self.rnd = np.random.default_rng(seed=state.seed)

    def generate_initial_position(self):
        position = self._get_random_empty_position()
        direction = self.rnd.integers(len(self.state.directions))

        self.state.agent_initial_position = position
        self.state.agent_initial_direction = direction

    def reset(self):
        """ reset the game, return the initial state"""
        self.state.agent_position = self.state.agent_initial_position
        self.state.agent_direction = self.state.agent_initial_direction
        return self._get_agent_state()

    def is_terminal(self, state=None):
        """ return True if position is terminal or False if it isn't """
        state = isnone(state, self.state)
        return self.dynamics.is_terminal(state)

    def step(self, action):
        """ take action, return next_state, reward, is_done, empty_info """
        assert not self.is_terminal(), 'Episode is finished!'

        state = self.state
        action = self.actions[action]
        reward = 0
        if action == 'stay':
            reward += self.dynamics.stay(state)
        elif action == 'move':
            reward += self.dynamics.move_forward(state)
            reward += self.dynamics.stay(state)
        elif action == 'turn left':
            reward += self.dynamics.turn(-1, state)
            reward += self.dynamics.stay(state)
        elif action == 'turn right':
            reward += self.dynamics.turn(1, state)
            reward += self.dynamics.stay(state)
        else:
            raise NotImplementedError(f'Action {action} is not implemented')

        agent_state = self._get_agent_state()
        is_done = self.is_terminal()
        return agent_state, reward, is_done, {}

    def _get_agent_state(self):
        position = self.state.agent_position
        direction = self.state.directions[self.state.agent_direction]
        return position, direction

    def _unflatten(self, flatten_position):
        return divmod(flatten_position, self.state.size)

    def _get_random_empty_position(self):
        non_empty_mask = self.state.obstacle_mask | self.state.food_mask
        available_states = np.flatnonzero(non_empty_mask)
        position = self.rnd.choice(available_states)
        return self._unflatten(position)
