from itertools import product
from typing import Tuple

import numpy as np
from numpy.random._generator import Generator

from htm_rl.common.utils import isnone, clip
from htm_rl.envs.biogwlab.env_state import BioGwLabEnvState


class BioGwLabEnvDynamics:
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
        state.agent_direction = self.rotate_direction(state.agent_direction, turn_direction)
        reward = self.actions['turn']['cost'] * self.time_cost
        return reward

    @staticmethod
    def rotate_direction(view_direction, turn_direction):
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
            state.food_scent = state.get_food_scent(state.food_scents)
            state.food_mask[i, j] = False
            state.food_map[i, j] = -1
            return self.food_rewards[food_type]

    def generate_scent_map(self, state: BioGwLabEnvState, rnd: Generator):
        channels = state.n_scent_channels
        size = state.size
        scent_map = np.zeros((channels, size, size), dtype=np.int8)
        for channel in range(channels):
            scent = state.food_scent[channel].ravel()
            n_cells = scent.size()
            n_active = int(.2 * n_cells)
            activations = rnd.choice(n_cells, p=scent, size=n_active)
            scent_map[channel, np.divmod(activations, size)] = 1
        return scent_map

class BioGwLabEnv:
    rnd: Generator
    actions = ['stay', 'move', 'turn left', 'turn right']

    state: BioGwLabEnvState
    dynamics: BioGwLabEnvDynamics

    def __init__(self, state: BioGwLabEnvState, dynamics: BioGwLabEnvDynamics):
        self.state = state
        self.dynamics = dynamics
        self.rnd = np.random.default_rng(seed=state.seed)

        self.generate_initial_position()

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


class BioGwLabStateVisualRepresenter:
    repr_len: int

    def __init__(self, state: BioGwLabEnvState):
        self.repr_len = (
                1 + state.n_wall_colors + state.n_area_types + state.n_food_types
        )

    def get_representation(self, state: BioGwLabEnvState):
        size = state.size
        vis_repr = np.zeros((size, size, self.repr_len), dtype=np.int8)
        for i, j in product(range(size), range(size)):
            vis_repr[i, j] = self.get_cell_representation(i, j, state)
        return vis_repr

    def get_cell_representation(self, i, j, state: BioGwLabEnvState):
        result = np.zeros(self.repr_len, dtype=np.int8)
        shift = 1
        if state.obstacle_mask[i, j]:
            result[shift + state.wall_colors[i, j]] = 1
        else:
            shift += state.n_wall_colors
            result[shift + state.areas_map[i, j]] = 1
            if state.food_mask[i, j]:
                shift += state.n_area_types
                result[shift + state.food_map[i, j]] = 1
        return result

    def init_outer_cells(self, repr):
        repr[:, :, 0] = 1


class BioGwLabStateScentRepresenter:
    rnd: Generator

    def __init__(self, seed: int):
        self.rnd = np.random.default_rng(seed=seed)

    def generate_scent_map(self, state: BioGwLabEnvState):
        channels = state.n_scent_channels
        size = state.size
        scent_map = np.zeros((channels, size, size), dtype=np.int8)
        for channel in range(channels):
            scent = state.food_scent[channel].ravel()
            n_cells = scent.size
            n_active = int(.2 * n_cells)
            activations = self.rnd.choice(n_cells, p=scent, size=n_active)
            scent_map[channel, np.divmod(activations, size)] = 1
        return scent_map

    def init_outer_cells(self, repr):
        pass


class BioGwLabEnvRepresentationWrapper(BioGwLabEnv):
    visual_representer: BioGwLabStateVisualRepresenter
    scent_representer: BioGwLabStateScentRepresenter

    visual_representation: np.ndarray

    def __init__(self, state: BioGwLabEnvState, dynamics: BioGwLabEnvDynamics):
        super().__init__(state, dynamics)

        self.visual_representer = BioGwLabStateVisualRepresenter(self.state)
        self.scent_representer = BioGwLabStateScentRepresenter(self.state.seed)

    def reset(self):
        _ = super().reset()

        self.visual_representation = self.visual_representer.get_representation(
            self.state
        )
        visual_representation = self.visual_representation
        scent_representation = self.scent_representer.generate_scent_map(self.state)
        return visual_representation, scent_representation

    def step(self, action):
        _, reward, is_done, _ = super().step(action)

        # update current cell representation, because only this one could be changed
        i, j = self.state.agent_position
        self.visual_representation[i, j] = self.visual_representer.get_cell_representation(
            i, j, self.state
        )

        visual_representation = self.visual_representation
        scent_representation = self.scent_representer.generate_scent_map(self.state)
        return visual_representation, scent_representation


class BioGwLabEnvObservationWrapper(BioGwLabEnvRepresentationWrapper):
    i_transformation = [1, -1, -1, 1, ]
    j_transformation = [1, 1, -1, -1, ]

    view_rect: Tuple[Tuple[int, int], Tuple[int, int]]
    scent_rect: Tuple[Tuple[int, int], Tuple[int, int]]

    def __init__(
            self, state: BioGwLabEnvState, dynamics: BioGwLabEnvDynamics,
            view_rect: Tuple[Tuple[int, int], Tuple[int, int]],
            scent_rect: Tuple[Tuple[int, int], Tuple[int, int]]
    ):
        super().__init__(state, dynamics)

        self.view_rect = self._view_to_machine(view_rect)
        self.scent_rect = self._view_to_machine(scent_rect)

    def reset(self):
        vis_repr, scent_repr = super().reset()

        vis_observation = self._clip_observation(
            vis_repr, self.view_rect, self.visual_representer.init_outer_cells
        )
        # scent_observation = self._clip_observation(
        #     scent_repr, self.scent_rect, self.scent_representer.init_outer_cells
        # )
        #
        # observation = np.concatenate(vis_observation, scent_observation, axis=None)
        # return observation
        return vis_observation

    def _clip_observation(self, full_repr, obs_rect, init_obs):
        state = self.state
        i, j = state.agent_position
        forward_dir = state.agent_direction
        repr_len = full_repr.shape[-1]

        i_high, j_high = self.state.size, self.state.size
        # print(i, j, forward_dir, i_high, j_high)
        # print(obs_rect)

        (bi, lj), (ui, rj) = obs_rect
        obs = np.empty((ui-bi+1, rj-lj+1, repr_len), dtype=np.int8)
        init_obs(obs)

        # print('orig', bi, ui, lj, rj)
        # print('====')
        # for i in range(5):
        #     _bi, _ui, _lj, _rj = self._rotate(bi, ui, lj, rj, i)
        #     print('rot', _bi, _ui, _lj, _rj)
        # print('====')

        _bi, _ui, _lj, _rj = self._rotate(bi, ui, lj, rj, forward_dir)
        # print('rot', _bi, _ui, _lj, _rj)

        _bi = clip(i + _bi, i_high)
        _ui = clip(_ui + i, i_high) + 1
        _lj = clip(j + _lj, j_high)
        _rj = clip(_rj + j, j_high) + 1
        # print('repr', _bi, _ui, _lj, _rj)
        repr = full_repr[_bi:_ui, _lj:_rj].copy()
        # print(repr.shape)

        _bi, _ui = _bi - i, _ui - i - 1
        _lj, _rj = _lj - j, _rj - j - 1

        # print('cli', _bi, _ui, _lj, _rj)
        _bi, _ui, _lj, _rj = self._rotate(_bi, _ui, _lj, _rj, 2 - forward_dir)
        # print('clip', _bi, _ui, _lj, _rj)

        bi -= _bi
        ui = bi - _bi + _ui + 1
        lj -= _lj
        rj = lj - _lj + _rj + 1
        # print('res', bi, ui, lj, rj)

        if forward_dir == 0:
            repr = np.swapaxes(repr, 0, 1)
            repr = np.flip(repr, 0)
        elif forward_dir == 1:
            repr = repr
        elif forward_dir == 2:
            repr = np.swapaxes(repr, 0, 1)
            repr = np.flip(repr, 1)
        else:
            repr = np.flip(repr, (0, 1))

        # print('obs shape', obs.shape)
        # print(obs[__bi:__ui, __lj:__rj].shape)
        obs[bi:ui, lj:rj] = repr
        return obs

    def _rotate(self, bi, ui, lj, rj, direction):
        bi *= self.i_transformation[direction - 1]
        ui *= self.i_transformation[direction - 1]
        lj *= self.j_transformation[direction - 1]
        rj *= self.j_transformation[direction - 1]
        if direction % 2 == 0:
            bi, lj = lj, bi
            ui, rj = rj, ui
        if bi > ui:
            bi, ui = ui, bi
        if lj > rj:
            lj, rj = rj, lj
        return bi, ui, lj, rj

    def _view_to_machine(self, rect):
        (lj, bi), (rj, ui) = rect
        return (bi, -rj), (ui, -lj)
