from typing import Tuple, List

import numpy as np
from numpy.random import Generator

from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import exp_decay


class QValueNetwork:
    discount_factor: float
    learning_rate: Tuple[float, float]

    cell_value: np.ndarray
    _rng: Generator

    def __init__(
            self, cells_sdr_size: int, seed: int,
            discount_factor: float,
            learning_rate: Tuple[float, float],
    ):
        self._rng = np.random.default_rng(seed)

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        # self.cell_value = np.full(cells_sdr_size, 0., dtype=np.float)
        self.cell_value = self._rng.uniform(-1e-4, 1e-4, size=cells_sdr_size)

    def values(self, xs: List[SparseSdr]) -> np.ndarray:
        return np.array([self.value(x) for x in xs])

    # noinspection PyPep8Naming
    def update(
            self, sa: SparseSdr, reward: float, sa_next: SparseSdr,
            E_traces: np.ndarray
    ):
        lr, _ = self.learning_rate
        Q = self.cell_value
        TD_error = self.td_error(sa, reward, sa_next)

        if E_traces is not None:
            Q += lr * TD_error * E_traces
        else:
            Q[sa] += lr * TD_error

    # noinspection PyPep8Naming
    def td_error(self, sa: SparseSdr, reward: float, sa_next: SparseSdr):
        # in general it could be s instead sa and V instead of Q
        gamma = self.discount_factor
        R = reward
        Q = self.cell_value

        Q_sa = Q[sa].mean()
        Q_sa_next = Q[sa_next].mean()

        TD_error = R + gamma * Q_sa_next - Q_sa
        return TD_error

    def value(self, x) -> float:
        if len(x) == 0:
            return np.infty
        return self.cell_value[x].mean()

    def decay_learning_factors(self):
        self.learning_rate = exp_decay(self.learning_rate)
