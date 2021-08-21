from typing import Optional

import numpy as np

from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import update_exp_trace, isnone, safe_ith, exp_decay


class EligibilityTraces:
    trace_decay: tuple[float, float]
    discount_factor: float
    E: Optional[np.ndarray]

    cells_sdr_size: Optional[int]

    def __init__(
            self, cells_sdr_size: int = None,
            trace_decay: tuple[float, float] = None,
            discount_factor: float = None
    ):
        self.cells_sdr_size = cells_sdr_size
        self.trace_decay = isnone(trace_decay, [.0, .0])
        self.discount_factor = discount_factor
        self.E = None
        self.reset()

    @property
    def enabled(self) -> bool:
        return self.trace_decay[0] > .05

    def update(self, sa: SparseSdr):
        if self.enabled:
            lambda_, gamma = self.trace_decay[0], self.discount_factor
            update_exp_trace(self.E, sa, lambda_ * gamma)

    def reset(self, decay: bool = True):
        if not self.enabled:
            self.E = None
            return

        if self.E is None:
            self.E = np.empty(self.cells_sdr_size, dtype=np.float)

        self.E.fill(0.)
        if decay:
            self.trace_decay = exp_decay(self.trace_decay)
