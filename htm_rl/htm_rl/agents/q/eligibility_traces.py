import numpy as np

from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import update_exp_trace


class EligibilityTraces:
    trace_decay: float
    discount_factor: float
    E: np.ndarray

    def __init__(self, cells_sdr_size: int, trace_decay: float, discount_factor: float):
        self.trace_decay = trace_decay
        self.discount_factor = discount_factor
        self.E = np.empty(cells_sdr_size, dtype=np.float)

        self.reset()

    # noinspection PyPep8Naming
    def update(self, sa: SparseSdr):
        E = self.E
        lambda_, gamma = self.trace_decay, self.discount_factor
        update_exp_trace(E, sa, lambda_ * gamma)

    def reset(self):
        self.E.fill(0.)