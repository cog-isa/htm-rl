import numpy as np
import matplotlib.pyplot as plt

from htm_rl.common.utils import isnone


class GridworldMdp:
    directions = [(1, 0), (0, -1), (-1, 0), (0, 1)]

    def __init__(self, gridworld_map: np.ndarray, seed: int):
        """
        Defines an MDP. Compatible with gym Env.
        :param gridworld_map: TODO
        """
        assert gridworld_map.ndim == 2, f'Provided map has {gridworld_map.ndim} dimensions; required: 2'
        assert np.count_nonzero(gridworld_map) >= 2, \
            f'Provided map does not have 2 available cells\n{gridworld_map}'

        self.gridworld_map = gridworld_map
        self.n_states = self.gridworld_map.size
        self.n_actions = 4
        self.shape = self.gridworld_map.shape

        self.rnd_generator = np.random.default_rng(seed=seed)
        self.initial_state = None
        self.terminal_state = None
        self._current_state, self._current_cell = None, None

        self.set_initial_state()
        self.set_terminal_state()
        self.reset()

    def set_initial_state(self):
        while True:
            initial_state = self._get_random_available_state()
            if isnone(self.terminal_state, -1) != initial_state:
                self.initial_state = initial_state

    def set_terminal_state(self):
        while True:
            terminal_state = self._get_random_available_state()
            if isnone(self.initial_state, -1) != terminal_state:
                self.terminal_state = terminal_state

    def is_terminal(self, state):
        """ return True if state is terminal or False if it isn't """
        return state == self.terminal_state

    def reset(self):
        """ reset the game, return the initial state"""
        self._current_state = self.initial_state
        self._current_cell = self._to_cell(self._current_state)
        return self._current_state

    def step(self, action):
        """ take action, return next_state, reward, is_done, empty_info """
        assert not self.is_terminal(self._current_state), 'Episode is finished!'

        i, j = self._current_cell
        i = self._clip(i + self.directions[action][0], self.shape[0])
        j = self._clip(j + self.directions[action][1], self.shape[1])

        if (i, j) != self._current_state and self.gridworld_map[i][j]:
            self._current_cell = i, j
            self._current_state = self._to_state(i, j)

        reward = self._reward(self._current_state)
        is_done = self.is_terminal(self._current_state)
        return self._current_state, reward, is_done, {}

    def render(self, mode: str = None):
        if mode == 'img':
            a = np.zeros_like(self.gridworld_map, dtype=np.int8)
            a[self.gridworld_map] = 20
            agent_i, agent_j = self._current_cell
            a[agent_i, agent_j] = 100
            goal_i, goal_j = self._to_cell(self.terminal_state)
            a[goal_i, goal_j] = 255

            plt.imshow(a)
            plt.show(block=True)
        else:
            print(f'State: {self._current_state}')

    def _to_state(self, i, j):
        return i * self.shape[0] + j

    def _to_cell(self, state):
        return divmod(state, self.shape[0])

    def _get_random_available_state(self):
        available_states = np.flatnonzero(self.gridworld_map)
        state = self.rnd_generator.choice(available_states)
        return state

    @staticmethod
    def _clip(x, high):
        if x >= high:
            return high - 1
        if x < 0:
            return 0
        return x

    def _reward(self, state):
        """ return the reward you get for taking action in state and landing on next_state"""
        return 1.0 if self.is_terminal(state) else -1e-2
