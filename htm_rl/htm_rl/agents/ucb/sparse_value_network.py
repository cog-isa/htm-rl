from typing import List, Tuple, Optional

import numpy as np
from numpy.random._generator import Generator

from htm_rl.common.sdr import SparseSdr


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
        total_visits = self._total_visits(options) if not greedy else None
        option_values = []
        for option in options:
            option_values.append(
                self._value_option(option, greedy, total_visits)
            )

        option_index: int = np.argmax(option_values)
        return option_index, option_values

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

    def _value_option(self, option, greedy: bool, total_visits: Optional[float] = None) -> float:
        if len(option) == 0:
            return np.infty

        value = self.cell_value[option]
        if not greedy:
            value += self._ucb_upper_bound_term(option, total_visits)
        return value.mean()

    # noinspection PyPep8Naming
    def _ucb_upper_bound_term(self, cells_sdr, total_visits):
        # NB: T regards to a total options visits, N - to just a cell
        # but we consider here as if each cell was a single
        # representative of an option
        T = total_visits
        N = self.cell_visit_count[cells_sdr]
        cp = self.ucb_exploration_factor[0]

        return cp * np.sqrt(2 * np.log(T + 1) / (N + 1))

    def _total_visits(self, options: List[SparseSdr]) -> int:
        # option visit count: avg visit count of its cells
        return sum(
            np.mean(self.cell_visit_count[state])
            for state in options
            if len(state) > 0
        )

    def reset(self):
        self.cell_eligibility_trace.fill(0.)

    def decay_learning_factors(self):
        self.ucb_exploration_factor = exp_decay(self.ucb_exploration_factor)
        self.learning_rate = exp_decay(self.learning_rate)


def exp_sum(s, decay, val):
    return s * decay + val


def lin_sum(s, lr, val):
    return s + lr * (val - s)


def update_slice_exp_sum(s, ind, decay, val):
    s[ind] *= decay
    s[ind] += val


def update_slice_lin_sum(s, ind, lr, val):
    s[ind] *= (1 - lr)
    s[ind] += lr * val


def update_exp_trace(traces, tr, decay, val=1.):
    traces *= decay
    traces[tr] += val


def exp_decay(factor_decay_tuple):
    factor, decay = factor_decay_tuple
    return factor * decay, decay


def modify_factor_tuple(factor_decay_tuple, alpha):
    factor, decay = factor_decay_tuple
    return factor * alpha, decay
