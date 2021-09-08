from typing import Optional

import numpy as np

from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import update_exp_trace, exp_decay, DecayingValue


class UcbEstimator:
    visit_decay: float
    ucb_exploration_factor: DecayingValue

    # you should not read it literally cause it's affected by exp MA window
    cell_visit_count: Optional[np.ndarray]

    def __init__(
            self, cells_sdr_size: int = None, visit_decay: float = None,
            ucb_exploration_factor: DecayingValue = (0., 0.)
    ):
        self.visit_decay = visit_decay
        self.ucb_exploration_factor = ucb_exploration_factor
        self.cell_visit_count = None
        if self.enabled:
            self.cell_visit_count = np.full(cells_sdr_size, 1., dtype=np.float)

    @property
    def enabled(self):
        return self.ucb_exploration_factor[0] > 1e-5

    def ucb_terms(self, xs: list[SparseSdr]) -> np.ndarray:
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

    def total_visits(self, xs: list[SparseSdr]) -> int:
        # state/action visit count: avg visit count of its cells
        return sum(
            self.cell_visit_count[x].mean()
            for x in xs
            if len(x) > 0
        )

    def decay_learning_factors(self):
        self.ucb_exploration_factor = exp_decay(self.ucb_exploration_factor)
