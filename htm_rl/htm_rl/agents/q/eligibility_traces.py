from typing import Optional

import numpy as np

from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import update_exp_trace, isnone


class EligibilityTraces:
    trace_decay: float
    discount_factor: float
    E: Optional[np.ndarray]
    enabled: bool

    def __init__(self, cells_sdr_size: int = None, trace_decay: float = None, discount_factor: float = None):
        self.trace_decay = trace_decay
        self.discount_factor = discount_factor
        self.enabled = isnone(trace_decay, .0) == .0
        if self.enabled:
            self.E = np.empty(cells_sdr_size, dtype=np.float)
        else:
            self.E = None

        self.reset()

    # noinspection PyPep8Naming
    def update(self, sa: SparseSdr):
        if self.enabled:
            lambda_, gamma = self.trace_decay, self.discount_factor
            update_exp_trace(self.E, sa, lambda_ * gamma)

    def reset(self):
        if self.enabled:
            self.E.fill(0.)
