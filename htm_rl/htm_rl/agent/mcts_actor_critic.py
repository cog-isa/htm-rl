from typing import List

import numpy as np
from numpy.random._generator import Generator

from htm_rl.common.sdr import SparseSdr


class MctsActorCritic:
    discount_factor: float
    update_cell_ratio: float
    online_learning: bool
    cell_visited_count: np.array
    cell_value: np.array
    _total_steps: int
    _cell_eligibility_trace: np.array
    _rng: Generator

    def __init__(
            self, cells_sdr_size: int, update_cell_ratio: float,
            discount_factor: float, seed: int, online: bool
    ):
        self.discount_factor = discount_factor
        self.update_cell_ratio = update_cell_ratio
        self.online_learning = online
        self._rng = np.random.default_rng(seed)

        self._cell_visited_count = np.full(cells_sdr_size, 1e-5, dtype=np.float)
        self._cell_value = np.zeros(cells_sdr_size, dtype=np.float)
        self._cell_eligibility_trace = np.zeros(cells_sdr_size, dtype=np.float)
        self._total_steps = 0

        if self.online_learning:
            self.cell_visited_count = self._cell_visited_count
            self.cell_value = self._cell_value
        else:
            self.cell_visited_count = self._cell_visited_count.copy()
            self.cell_value = self._cell_value.copy()

    def add_step(self, active_cells_sdr: SparseSdr, reward: float):
        # NB: each step the stats are updated only for the fraction of active cells,
        # which is defined by `update_cell_ratio`. This's an **average** ratio!

        active_cells_sdr = np.array(active_cells_sdr)
        # chooses cells with specified average rate ==> get indicator [0 1 0 0 0 1 ... ]
        active_cells_update_voting = self._rng.binomial(
            1, p=self.update_cell_ratio, size=active_cells_sdr.size
        )
        active_cells_update_indices = np.nonzero(active_cells_update_voting)[0]
        updated_cells = active_cells_sdr[active_cells_update_indices]

        self._cell_visited_count[updated_cells] += 1

        self._cell_eligibility_trace *= self.discount_factor
        self._cell_eligibility_trace[updated_cells] += 1.
        self._cell_value += self._cell_eligibility_trace * reward

    def choose(self, options: List[SparseSdr]):
        ucb1_values = np.array([
            self.value(cells_sdr) for cells_sdr in options
        ])
        return np.argmax(ucb1_values)

    def value(self, cells_sdr: SparseSdr):
        T = self._total_steps

        N = self._cell_visited_count[cells_sdr]
        # similar to importance sampling - each step even an active cell is updated with some probability
        N *= self.update_cell_ratio

        R = self._cell_value[cells_sdr]
        Q = R / N
        U = (2 * np.log(T) / N)**.5
        V = Q + U
        return V.mean()

    def reset(self):
        self._cell_eligibility_trace.fill(0.)

        if not self.online_learning:
            self.cell_visited_count = self._cell_visited_count.copy()
            self.cell_value = self._cell_value.copy()
