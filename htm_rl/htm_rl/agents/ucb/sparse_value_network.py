from typing import List, Tuple

import numpy as np
from numpy.random._generator import Generator

from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import update_exp_trace, exp_decay


class UcbEstimator:
    visit_decay: float
    discount_factor: float
    learning_rate: Tuple[float, float]
    ucb_exploration_factor: Tuple[float, float]

    # you should not to read it literally cause it's affected by exp MA window
    cell_visit_count: np.ndarray

    def __init__(
            self, cells_sdr_size: int,
            visit_decay: float,
            discount_factor: float,
            learning_rate: Tuple[float, float],
            ucb_exploration_factor: Tuple[float, float]
    ):
        self.visit_decay = visit_decay
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.ucb_exploration_factor = ucb_exploration_factor

        self.cell_visit_count = np.full(cells_sdr_size, 1., dtype=np.float)

    def ucb_terms(self, xs: List[SparseSdr]) -> np.ndarray:
        total_visits = self.total_visits(xs)
        return np.array([self.ucb_term(x, total_visits) for x in xs])

    def update(self, sa: SparseSdr):
        update_exp_trace(self.cell_visit_count, sa, self.visit_decay)

    # noinspection PyPep8Naming
    def ucb_term(self, x, total_visits) -> float:
        # NB: T regards to a total options (xs) visits, N - to just a cells in x
        # but we consider here as if each cell was a single
        # representative of an option
        T = total_visits
        N = self.cell_visit_count[x]
        cp = self.ucb_exploration_factor[0]

        ucb = cp * np.sqrt(2 * np.log(T + 1) / (N + 1))
        return ucb.mean()

    def total_visits(self, xs: List[SparseSdr]) -> int:
        # option visit count: avg visit count of its cells
        return sum(
            self.cell_visit_count[x].mean()
            for x in xs
            if len(x) > 0
        )

    def decay_learning_factors(self):
        self.ucb_exploration_factor = exp_decay(self.ucb_exploration_factor)
        self.learning_rate = exp_decay(self.learning_rate)


class SparseValueNetwork:
    trace_decay: float
    visit_decay: float
    discount_factor: float
    learning_rate: Tuple[float, float]
    ucb_exploration_factor: Tuple[float, float]

    # you should not to read it literally cause it's affected by exp MA window
    cell_visit_count: np.ndarray
    cell_value: np.ndarray

    cell_eligibility_trace: np.ndarray

    _rng: Generator

    def __init__(
            self, cells_sdr_size: int, seed: int,
            trace_decay: float, visit_decay: float,
            discount_factor: float,
            learning_rate: Tuple[float, float],
            ucb_exploration_factor: Tuple[float, float]
    ):
        self._rng = np.random.default_rng(seed)

        self.trace_decay = trace_decay
        self.visit_decay = visit_decay
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.ucb_exploration_factor = ucb_exploration_factor

        self.cell_visit_count = np.full(cells_sdr_size, 1., dtype=np.float)
        # self.cell_value = np.full(cells_sdr_size, 0., dtype=np.float)
        self.cell_value = self._rng.uniform(-1e-4, 1e-4, size=cells_sdr_size)
        self.cell_eligibility_trace = np.zeros(cells_sdr_size, dtype=np.float)

    # noinspection PyTypeChecker
    def choose(self, options: List[SparseSdr], greedy=False) -> int:
        option_values = self.evaluate_options(options)
        if not greedy:
            option_values += self.evaluate_options_ucb_term(options)

        option_index: int = np.argmax(option_values)
        return option_index

    def evaluate_options(self, options: List[SparseSdr]) -> np.ndarray:
        return np.array([self._evaluate_option(option) for option in options])

    def evaluate_options_ucb_term(self, options: List[SparseSdr]) -> np.ndarray:
        total_visits = self._total_visits(options)
        return np.array([self._ucb_term(option, total_visits) for option in options])

    # noinspection PyPep8Naming
    def update(
            self, sa: SparseSdr, reward: float, sa_next: SparseSdr, td_lambda=True,
            update_visit_count=True
    ):
        if update_visit_count:
            self._update_cell_visit_counter(sa)

        lr, _ = self.learning_rate
        Q = self.cell_value
        TD_error = self._td_error(sa, reward, sa_next)

        if td_lambda:
            self._update_eligibility_trace(sa)
            E = self.cell_eligibility_trace
            Q += lr * TD_error * E
        else:
            Q[sa] += lr * TD_error

    # noinspection PyPep8Naming
    def _td_error(self, sa: SparseSdr, reward: float, sa_next: SparseSdr):
        gamma = self.discount_factor
        R = reward
        Q = self.cell_value

        Q_sa = Q[sa].mean()
        Q_sa_next = Q[sa_next].mean()

        TD_error = R + gamma * Q_sa_next - Q_sa
        return TD_error

    # noinspection PyPep8Naming
    def _update_eligibility_trace(self, sa: SparseSdr):
        E = self.cell_eligibility_trace
        lambda_, gamma = self.trace_decay, self.discount_factor
        update_exp_trace(E, sa, lambda_ * gamma)

    def _update_cell_visit_counter(self, sa: SparseSdr):
        update_exp_trace(self.cell_visit_count, sa, self.visit_decay)

    def _evaluate_option(self, option) -> float:
        if len(option) == 0:
            return np.infty
        return self.cell_value[option].mean()

    # noinspection PyPep8Naming
    def _ucb_term(self, option, total_visits) -> float:
        # NB: T regards to a total options visits, N - to just a cell
        # but we consider here as if each cell was a single
        # representative of an option
        T = total_visits
        N = self.cell_visit_count[option]
        cp = self.ucb_exploration_factor[0]

        ucb = cp * np.sqrt(2 * np.log(T + 1) / (N + 1))
        return ucb.mean()

    def _total_visits(self, options: List[SparseSdr]) -> int:
        # option visit count: avg visit count of its cells
        return sum(
            np.mean(self.cell_visit_count[option])
            for option in options
            if len(option) > 0
        )

    def reset(self):
        self.cell_eligibility_trace.fill(0.)

    def decay_learning_factors(self):
        self.ucb_exploration_factor = exp_decay(self.ucb_exploration_factor)
        self.learning_rate = exp_decay(self.learning_rate)

