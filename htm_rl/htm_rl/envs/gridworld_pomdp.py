import numpy as np

from htm_rl.common.utils import isnone
from htm_rl.envs.gridworld_mdp import GridworldMdp


class GridworldPomdp(GridworldMdp):
    view_radius: int

    def __init__(self, view_radius, gridworld_map: np.ndarray, seed: int):
        assert view_radius >= 1
        self.view_radius = view_radius
        super(GridworldPomdp, self).__init__(gridworld_map, seed)

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

    def state_to_obs(self, state):
        return self._to_observation(self._to_cell(state))

    def _to_observation(self, cell=None, repr=None):
        x, y = isnone(cell, self._current_cell)
        d = self.view_radius
        xhigh, yhigh = self.shape
        # print(x, y, d, self.shape)

        lx, rx = self._clip(x - d, xhigh), self._clip(x + d, xhigh) + 1
        by, uy = self._clip(y - d, yhigh), self._clip(y + d, yhigh) + 1

        _lx, _rx = d + (lx - x), d + (rx - x)
        _by, _uy = d + (by - y), d + (uy - y)
        # print(lx, rx, by, uy)
        # print(_lx, _rx, _by, _uy)

        size = 2 * d + 1
        obs = np.zeros((size, size), dtype=np.int8)

        repr = isnone(repr, self.gridworld_map)
        obs[_lx:_rx, _by:_uy] = repr[lx:rx, by:uy]
        return obs
