import numpy as np

from htm_rl.common.utils import isnone
from htm_rl.envs.gridworld_mdp import GridworldMdp


class GridworldPomdp(GridworldMdp):
    view_radius: int

    def __init__(self, view_radius, gridworld_map: np.ndarray, seed: int):
        super(GridworldPomdp, self).__init__(gridworld_map, seed)

        assert view_radius >= 1
        self.view_radius = view_radius

    def reset(self):
        """ reset the game, return the initial state"""
        super(GridworldPomdp, self).reset()
        return self._to_observation()

    def step(self, action):
        """ take action, return next_state, reward, is_done, empty_info """
        _, reward, is_done, _ = super(GridworldPomdp, self).step(action)
        observation = self._to_observation()
        return observation, reward, is_done, {}

    def get_observation_representation(self, state=None):
        state = isnone(state, self._current_state)
        state_repr, seed = self.get_representation(state, mode='img')

        cell = self._to_cell(state)
        obs_repr = self._to_observation(cell, state_repr)
        return obs_repr, seed

    def _to_observation(self, cell=None, repr=None):
        x, y = isnone(cell, self._current_cell)
        d = self.view_radius
        xhigh, yhigh = self.shape

        lx, rx = self._clip(x - d, xhigh), self._clip(x + d, xhigh)
        by, uy = self._clip(y - d, yhigh), self._clip(y + d, yhigh)

        size = 2 * d + 1
        obs = np.zeros((size, size), dtype=np.int8)

        repr = isnone(repr, self.gridworld_map)
        obs[lx:rx, by:uy] = repr[lx:rx, by:uy]
        return obs
