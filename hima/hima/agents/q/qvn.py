from typing import Tuple, List

import numpy as np
from numpy.random import Generator

from hima.common.sdr import SparseSdr
from hima.common.utils import exp_decay


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

        self.cell_value = self._rng.uniform(-1e-5, 1e-5, size=cells_sdr_size)

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
        gamma = self.discount_factor
        R = reward
        Q_sa = self.value(sa)
        Q_sa_next = self.value(sa_next)
        return R + gamma * Q_sa_next - Q_sa

    def value(self, x) -> float:
        if len(x) == 0:
            return np.infty
        # noinspection PyTypeChecker
        return np.median(self.cell_value[x])

    def decay_learning_factors(self):
        self.learning_rate = exp_decay(self.learning_rate)
