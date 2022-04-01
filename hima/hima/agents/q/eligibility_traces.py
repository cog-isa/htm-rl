from typing import Optional

import numpy as np

from hima.common.sdr import SparseSdr
from hima.common.utils import update_exp_trace, exp_decay, DecayingValue


class EligibilityTraces:
    trace_decay: DecayingValue
    discount_factor: float
    with_reset: bool
    E: Optional[np.ndarray]

    cells_sdr_size: Optional[int]

    def __init__(
            self, cells_sdr_size: int = None,
            trace_decay: DecayingValue = (.0, .0),
            discount_factor: float = None,
            with_reset: bool = False
    ):
        self.cells_sdr_size = cells_sdr_size
        self.trace_decay = trace_decay
        self.discount_factor = discount_factor
        self.with_reset = with_reset
        self.E = None
        self.reset()

    @property
    def enabled(self) -> bool:
        return self.trace_decay[0] > .05

    def update(self, sa: SparseSdr, with_reset: bool = False):
        if self.enabled:
            lambda_, gamma = self.trace_decay[0], self.discount_factor
            update_exp_trace(
                self.E, sa,
                decay=lambda_ * gamma,
                with_reset=self.with_reset or with_reset
            )

    def reset(self, decay: bool = True):
        if not self.enabled:
            self.E = None
            return

        if self.E is None:
            self.E = np.empty(self.cells_sdr_size, dtype=np.float)

        self.E.fill(0.)
        if decay:
            self.decay_trace_decay()

    def decay_trace_decay(self):
        self.trace_decay = exp_decay(self.trace_decay)
