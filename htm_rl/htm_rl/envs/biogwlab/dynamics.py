import numpy as np
from numpy.random._generator import Generator

from htm_rl.common.utils import clip
from htm_rl.envs.biogwlab.environment_state import BioGwLabEnvState, EnvironmentState


class BioGwLabEnvDynamics:
    actions = {
        'stay': {
            'cost': .3
        },
        'move': {
            'cost': .7,
        }
    }
    food_reward = 4.
    time_cost = -.05

    def __init__(self):
        pass

    def is_terminal(self, state: EnvironmentState):
        return state.n_foods <= 0

    def stay(self, state):
        i, j = state.agent_position
        reward = self.actions['stay']['cost'] * self.time_cost
        if state.food_mask[i, j]:
            reward += self._eat_food(i, j, state)

        return reward

    def move(self, state, direction):
        i, j, success = self._move(state, direction)
        state.agent_position = (i, j)
        reward = self.actions['move']['cost'] * self.time_cost
        return reward

    @staticmethod
    def _move(state: EnvironmentState, move_direction):
        i, j = state.agent_position
        directions = state.directions

        i += directions[move_direction][0]
        j += directions[move_direction][1]

        i = clip(i, state.shape[0])
        j = clip(j, state.shape[1])
        success = (i, j) != state.agent_position and not state.obstacle_mask[i, j]
        if not success:
            i, j = state.agent_position
        return i, j, success

    def _eat_food(self, i, j, state: EnvironmentState):
        state.food_mask[i, j] = False
        state.n_foods -= 1

        reward = self.food_reward
        return reward
