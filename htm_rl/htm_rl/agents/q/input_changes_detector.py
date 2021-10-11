import numpy as np

from htm_rl.common.sdr import SparseSdr
from htm_rl.common.utils import update_exp_trace


class InputChangesDetector:
    input_freq: np.ndarray

    def __init__(self, input_sdr_size: int):
        self.input_freq = np.zeros(input_sdr_size, dtype=np.float)

    def reset(self):
        self.input_freq.fill(.0)

    def changed(self, x: SparseSdr, train: bool) -> float:
        if not train:
            return True

        update_exp_trace(self.input_freq, x, .5)
        freq = self.input_freq[x].mean()
        # 1 + .5 + .25 + ... ==> > 3 times avg repeat
        return freq < 1.8
