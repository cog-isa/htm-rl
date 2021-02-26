import numpy as np
from numpy.random._generator import Generator

from htm_rl.common.utils import clip
from htm_rl.envs.biogwlab0.environment_state import BioGwLabEnvState


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
    time_cost = -.05

    def __init__(self):
        pass

    def is_terminal(self, state):
        return state.n_rewarding_foods <= 0

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

    def count_rewarding_food_items(self, state):
        return sum(
            1
            for _, _, food_type in state.food_items
            if self.food_rewards[food_type] >= 0
        )

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

            state.food_mask[i, j] = False
            state.food_map[i, j] = -1
            reward = self.food_rewards[food_type]
            if reward >= 0:
                state.n_rewarding_foods = 0

            # state.food_scents[:, :, :, k] = 0
            # state.food_scent = state.get_food_scent(state.food_scents)
            return reward

    def generate_scent_map(self, state: BioGwLabEnvState, rnd: Generator):
        channels = state.n_scent_channels
        size = state.size
        scent_map = np.zeros((channels, size, size), dtype=np.int8)
        for channel in range(channels):
            scent = state.food_scent[channel].ravel()
            n_cells = scent._shape_xy()
            n_active = int(.2 * n_cells)
            activations = rnd.choice(n_cells, p=scent, size=n_active)
            scent_map[channel, np.divmod(activations, size)] = 1
        return scent_map
