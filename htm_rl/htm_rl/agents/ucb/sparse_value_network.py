from typing import List, Tuple

import numpy as np
from numpy.random._generator import Generator

from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import trace


class SparseValueNetwork:
    trace_decay: float
    discount_factor: float
    learning_rate: float

    # you should not to read it literally cause it's affected by exp MA window
    cell_visit_count: np.ndarray
    cell_value: np.ndarray
    visit_decay: float
    reward_bounds: Tuple[float, float]

    cell_eligibility_trace: np.ndarray
    ucb_exploration_factor: Tuple[float, float]

    _rng: Generator

    def __init__(
            self, cells_sdr_size: int, seed: int,
            trace_decay: float, visit_decay: float,
            discount_factor: float, learning_rate: float,
            ucb_exploration_factor: Tuple[float, float]
    ):
        self._rng = np.random.default_rng(seed)

        self.trace_decay = trace_decay
        self.visit_decay = visit_decay
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        self.cell_visit_count = np.full(cells_sdr_size, 1., dtype=np.float)
        # self.cell_value = np.full(cells_sdr_size, 0., dtype=np.float)
        self.cell_value = self._rng.uniform(-1e-4, 1e-4, size=cells_sdr_size)
        self.cell_eligibility_trace = np.zeros(cells_sdr_size, dtype=np.float)
        self.reward_bounds = 0., 1.
        self.ucb_exploration_factor = ucb_exploration_factor

    # noinspection PyTypeChecker
    def choose(self, options: List[SparseSdr], greedy=False) -> int:
        total_visits, reward_range = None, None
        if not greedy:
            total_visits = self._total_visits(options)
            reward_range = self.reward_bounds[1] - self.reward_bounds[0]

        option_values = []
        for option in options:
            if len(option) == 0:
                option_values.append(np.infty)
                continue

            value = self.cell_value[option]
            if not greedy:
                value += self._ucb_upper_bound_term(option, total_visits) * reward_range

            option_values.append(value.mean())

        option_index: int = np.argmax(option_values)
        return option_index

    def update(self, sa: SparseSdr, reward: float, sa_next: SparseSdr, td_lambda=True):
        # self._update_reward_bounds(reward)
        if td_lambda:
            self._update_cell_visit_counter(sa)
            self._update_cell_value(sa, reward, sa_next)
        else:
            self._update_cell_value_td_0(sa, reward, sa_next)

    # noinspection PyPep8Naming
    def _update_cell_value(self, sa: SparseSdr, reward: float, sa_next: SparseSdr):
        lr = self.learning_rate
        lambda_, gamma = self.trace_decay, self.discount_factor
        R = reward
        V = self.cell_value
        E = self.cell_eligibility_trace

        V_sa = V[sa].mean()
        V_sa_next = V[sa_next].mean()

        E *= lambda_ * gamma
        E[sa] += 1.

        TD_error = R + gamma * V_sa_next - V_sa
        V += lr * TD_error * E

    # noinspection PyPep8Naming
    def _update_cell_value_td_0(self, sa: SparseSdr, reward: float, sa_next: SparseSdr):
        lr = self.learning_rate
        gamma = self.discount_factor
        R = reward
        V = self.cell_value

        V_sa = V[sa].mean()
        V_sa_next = V[sa_next].mean()

        TD_error = R + gamma * V_sa_next - V_sa
        V[sa] += lr * TD_error

    def _update_cell_visit_counter(self, sa: SparseSdr):
        self.cell_visit_count *= self.visit_decay
        self.cell_visit_count[sa] += 1.

    def _update_reward_bounds(self, reward: float):
        low, high = self.reward_bounds
        if low <= reward <= high:
            return

        if reward < low:
            low = reward
        else:
            high = reward
        self.reward_bounds = low, high

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

        alpha, decay = self.ucb_exploration_factor
        self.ucb_exploration_factor = alpha * decay, decay
        # self.learning_rate *= self.ucb_exploration_factor[1]
